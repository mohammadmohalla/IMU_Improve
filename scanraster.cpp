
#include "scanraster.h"

#include <opencv2/core/types_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <math.h>
#include <vector>
#include <stddef.h>


using namespace cv;
using namespace std;

#define INPUT_THRESHOLD     128.0       // threshold for input image

#define RANDOM_LINES        500         // number of random test lines in the first pass
#define RANDOM_QUADS        20          // max. number of randomly detected quads
#define MAX_ITERATIONS      10          // max. number of iterations to find additional quads


#define ADAPTIVE_DIST                   // use relative ratios for pattern matching

struct ScanRaster
{
    int regionId;               // current regionId, incremented with each quad

    Quad *quads;                // result array of detected quads
    int quad_count;             // current number of detected quads
    int max_quad_count;         // maximum number of detected quads

    Mat *input;                 // camera input image
    Mat *image;                 // working image, thresholded
    Mat *temp;                  // temp image for floodfill
};

/**
 * Checks if the given rectangle intersects the border of the image.
 * These quads are incomplete and cannot be detected properly.
 * @return  true if rectangle r intersects to border
 */
static bool intersects_border(Rect *r, Mat *image)
{
    // intersects top-level border
    if ((r->x == 0) || (r->y == 0))
        return true;

    // intersects right border
    if ((r->x + r->width >= image->cols - 1))
        return true;

    // intersects bottom border
    if ((r->y + r->height >= image->rows - 1))
        return true;

    // no intersection => return false
    return false;
}

/**
 * Finds the corners of a quad using the OpenCV contours functions.
 *
 * @param Rect      rectangle of the image to search
 * @param id        id = greyscale color of the object
 * @param image     OpenCV image to search
 * @param temp      temporal mask image, must be 2 pixel wider, taller than image
 * @param corners   pointer to an array of the 4 resulting corners
 * @return          true if the object is a quad, false otherwise
 */
static bool find_corners(Rect *r, int id, cv::Mat *image, cv::Mat *temp, Point2d *corners)
{
    Mat rect = (*temp)(Rect(0, 0, r->width, r->height));

    // compute mask of the polygon with the given id.
    rect = (*image)(*r) == id;

    vector<vector<Point> > contours;
    vector<Vec4i> hierachy;

    // extract external contour of the polygon.
    findContours(rect, contours, hierachy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(r->tl()));
    if (contours.size() != 1)
        return false;

    vector<Point> polygon;

    // Approximate and reduce the contour until the polygon
    // has four corners and might be a valid quad.
    // Each iteration increases the threshold.
    for (int i = 1; i < 8; i++)
    {
        approxPolyDP(contours[0], polygon, i, true);
        if (polygon.size() == 4)
            break;
    }

    // Proceed only if the polygon has exactly four corners.
    if (polygon.size() == 4)
    {
        for (int i = 0; i < 4; i++)
        {
            corners[i] = polygon[i];
        }
        return true;
    }

    // The polygon has more or less than four corners and could not be a quad.

    return false;
}

/**
 * Refines the corners of an already for sub-pixel accuracy by
 * using the OpenCV cornerSubPix function.
 * @param points    pointer to an array of corners, which are modified
 * @param count     number of points in the array
 * @param input     input image, must be greyscale
 */
static void refine_corners(Point2d *points, int count, Mat *input)
{
    vector<Point2f> polygon2f;

    // copy coordinates into a std::vector, OpenCV requires it.
    for (int i = 0; i < count; i++)
    {
        polygon2f.push_back(points[i]);
    }

    // utilize the color gradient to refine the coordinates of the corners.
    cornerSubPix(*input, polygon2f, Size(5,5), Size(-1,-1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 15, 0.1));

    // copy refined coordinates back into the parameter array.
    for (int i = 0; i < count; i++)
    {
        points[i] = polygon2f[i];
    }
}

/**
 * Performs a bilinear interpolation on the quad to
 * return the coordinates of an inner point.
 *
 * Correspondance between the coordinates (u,v) and the point of the quad
 * Point 0 has coordinates (0,0)
 * Point 1 has coordinates (1,0)
 * Point 2 has coordinates (1,1)
 * Point 3 has coordinates (0,1)
 *
 * @param quad  the quad to interpolate
 * @param u     u coordinate for interpolation, must be in range 0..1
 * @param vu    v coordinate for interpolation, must be in range 0..1*
 * @return      the interpolated point
 */
static Point2d lerp(Quad *quad, float u, float v)
{
    // Compute weights for 4 points.
    // w0 w1
    // w3 w2
    double w0 = (1-u) * (1-v);
    double w1 = u * (1-v);
    double w2 = u * v;
    double w3 = (1-u) * v;

    // read points of the quad
    // a b
    // d c
    Point2d a = quad->points[0];
    Point2d b = quad->points[1];
    Point2d c = quad->points[2];
    Point2d d = quad->points[3];

    // interpolate xy coordinates as weighted sum
    double x = a.x * w0 + b.x * w1 + c.x * w2 + d.x * w3;
    double y = a.y * w0 + b.y * w1 + c.y * w2 + d.y * w3;
    return Point2d(x,y);
}

/**
 * Aligns a quad, so that the internal marker is a point 0 by
 * checking the relative brightness in the image.
 * The internal marker is white, the other corners are black.
 *
 * @param quad      the quad to align, is modified
 * @param input     the greyscale image from the camera
 */
static void align_quad(Quad *quad, Mat *input)
{
    int max_color = 0;
    int max_index = 0;

    // These are the relative coordinates of the four corners.
    // If the quad is valid, one of these is white and the other
    // three are black.
    // The coordinates are derived by construction from the raster.
    static Point2d corners[4] =
    {
        Point2d(1.5f/6.0f, 1.5f/6.0f),
        Point2d(4.5f/6.0f, 1.5f/6.0f),
        Point2d(4.5f/6.0f, 4.5f/6.0f),
        Point2d(1.5f/6.0f, 4.5f/6.0f)
    };

    // find brightest corner.
    for (int i = 0; i < 4; i++)
    {
        Point2d p = lerp(quad, corners[i].x, corners[i].y);

        int color = *input->ptr(p.y, p.x);

        if (color > max_color)
        {
            max_color = color;
            max_index = i;
        }
    }

    // rotate quad accordingly.
    int steps = max_index;

    Point2d p[4];
    for (int i = 0; i < 4; i++)
    {
        p[i] = quad->points[(i + steps) % 4];
    }

    // copy points back.
    for (int i = 0; i < 4; i++)
    {
        quad->points[i] = p[i];
    }
}

/**
 * Marks and examines the object at coordinates (x,y) in the input image.
 * The region is checked further and may end up as a detected quad.
 * If the object is a quad, it is added to the list in the ScanRaster object.
 *
 * @param scan  the current ScanRaster object storing the context
 * @param x     x coordinate of the pixel in the region
 * @param y     y coordinate of the pixel in the region
 */
static void mark_region(ScanRaster *scan, int x, int y)
{
    Quad quad;
    Rect rect;

    // quads are black, if the pixel is not black, return immediately.
    if (*scan->image->ptr(y,x))
        return;

    // allocate a new region id.
    int id = scan->regionId++;

    // allocate a new rectangle with the floodFill to apply*/
    if (!cv::floodFill(*scan->image, Point(x, y), Scalar(id), &rect, Scalar(0), Scalar(0))) {
        return;
    }

    // ignore possible incomplete quads, which are touching the border, they cannot evaluated correctly.
    if (intersects_border(&rect, scan->image))
    {
        //log("scan_raster: quad at (%i, %i) intersects the border of the image", x, y);
        return;
    }

    // try to reduce the pixel region into a polygon with four corners.
    if (!find_corners(&rect, id, scan->image, scan->temp, quad.points))
    {
        //log("scan_raster: object at (%i, %i) is not a quad", x, y);
        return;
    }

    // at this position, the number of corners is always four.

    // adjust coordinates of the corners according to the color gradient.
    refine_corners(quad.points, 4, scan->input);

    // if necessary, rotate the quad, so that the internal white marker is at position 0.
    align_quad(&quad, scan->input);

    // add quad to the list of detected quads.
    // check for overflow and ignore additional quads.
    if (scan->quad_count < scan->max_quad_count)
    {
        scan->quads[scan->quad_count++] = quad;
    }
}

/***
 * Checks if the ratio between a and b is equal to r with some tolerance of 25%.
 * This function is used to check the ratio of line segments to detect a quad.
 * @param a     length of the first segment in pixels
 * @param b     length of the second segment in pixels
 * @param r     expected ratio
 * @return      true if the ratio of a and b matches r, false otherwise
 */
static inline bool is_ratio(int a, int b, float r)
{
    float f = (float)a/(float)b*r;
    return (f > 0.8f) && (f < 1.25f);
}

/**
 * Scans a line in the image for the quad pattern. In particular, this
 * function searched for a pattern of 2x (black and white) pixels with a ratio of 3:1.

 * The test utilizes a state machine, which looks for a pattern of black and white
 * segments along a line. If the ratio between black and white segments corresponds
 * to the reference quad, the region is examined further.

 * If adaptive testing (ADAPTIVE_DIST) is enabled, the ratio is not fixed at 3:1
 * but instead depends on the ratio of the previously detected part of the line.
 * This addition also allows to detect quads at very steep angles.
 *
 * If a >> possible << quad is detected, this function calls mark_region to
 * examine the object in more detail.
 *
 * The scan is started at pixel (x,y) and moved into direction (dx, dy)
 * until the border of the image is reached.
 *
 * HINT: The direction (dx, dy) is given in 16.16 fixed-point format.
 *       Fixed point has been shown to be significant faster on
 *       certain ARM architectures.
 *
 * @param scan      ScanRaster object holding the input image
 * @param x         x coordinate of the starting point
 * @param y         y coordinate of the starting point
 * @param dx        x direction of the line as 16:16 fixed point
 * @param dy        y direction of the line as 16:16 fixed point
 * @param fast      the scanning aborts when enough quads were found
  */
static void scan_line(ScanRaster *scan, int x, int y, int dx, int dy, bool fast)
{
    // Convert width and height from int to 16.16 bit fixed-point.
    int width = scan->image->cols << 16;
    int height = scan->image->rows << 16;

    // Convert x,y coordinates into 16.16 fixed-point.
    x = x << 16;
    y = y << 16;

    // Length of the last black and white segments along the test line.
    // These variables act similar to a shift-register with size 5.
    int count_0 = 1;    // length of the current segment.
    int count_1 = 1;    // length of the previous segment.
    int count_2 = 1;    // length of the 2nd previous segment.
    int count_3 = 1;    // length of the 3rd previous segment.
    int count_4 = 1;    // length of the 4th previous segment.

    bool state = false;  // color of the current segment, true = white, false = black.
    bool first = true;   // first


    bool running = true;
    // Iterate along a line within the bounds of the image.
    // This is the inner loop of the image processing algorithm.
    while (running && (x >= 0) && (y >= 0) && (x < width) && (y < height))
    {
        // Retrieve pointer to the pixel at position x,y.
        // Shift by 16 to convert from 16.16 fixed-point to int.
        uchar* ptr = scan->image->ptr(y >> 16,x >> 16);

        bool pixel = *ptr == 255; // color of the pixel, true = white, false = black.

        // Check if there is a transition between black and white segments.
        if (pixel != state)
        {
            // Avoid noise by requiring at least 20 pixels in the current segment
            // of a valid quad. This value has been choosen experimentally but works
            // for all tested resolutions.
            if (!state && (count_0 > 20))
            {
                /* Here, we are at a transition from black (! state= false) to white (pixel=true),
                 * while the black segment has a size of at least 20 pixels.
                 * In case of a valid quad, the ray is currently existing the quad.
                 *
                 * The following rations indicate, that the ray has passed two quads.
                 * 3:   r_0: black segment within the quad
                 * 1:   r_1: white segment between two quads
                 * 3:   r_2: black segment within the previous quad
                 * 3:   r_3: white segment between two quads
                 *
                 * In the following figure, the ray has passed from left to right and
                 * the X marks the current position (dx = 0x10000, dy = 0).
                 *
                 *          |-------------|         |-------------|
                 *          |             |         |             |
                 *  <-r3->  |  <- r_2 ->  | <-r1->  |  <- r_0 ->  X
                 * (white)  |  (black)    | (white) |  (black)    |
                 *          |-------------|         |-------------|
                 */

                /* ADAPTIVE_DIST produces more accurate results by considering the
                 * perspective distortion. With the exception of the first segment,
                 * the ratio between black and white is no longer fixed at 3:1
                 * but corresponds to the ratio of the previous segment.
                 */

#ifdef ADAPTIVE_DIST
                bool r_0 = is_ratio(count_0,  count_1, 1.0f/3.0f);
                bool r_1 = is_ratio(count_1,  count_2, (float)count_0/(float)count_1);
                bool r_2 = is_ratio(count_2,  count_3, (float)count_1/(float)count_2);
                bool r_3 = is_ratio(count_3,  count_4, (float)count_2/(float)count_3);
#else
                bool r_0 = is_ratio(count_0,  count_1, 1.0f/3.0f);
                bool r_1 = is_ratio(count_0,  count_2, 1);
                bool r_2 = is_ratio(count_0,  count_3, 1.0f/3.0f);
                bool r_3 = is_ratio(count_0,  count_4, 1);
#endif

                // Check if all ratios are valid */
                if (r_0 && r_1 && r_2 && r_3)
                {
                    if (first)
                    {
                        // The first time, a potentially quad region is detected on the ray,
                        // we can evaluate the two previous quads as well.
                        int dist1 = count_0 + count_1 + count_2 + count_3 + count_4/2;
                        mark_region(scan, (x-dx*dist1) >> 16, (y-dy*dist1) >> 16);

                        int dist0 = count_0 + count_1 + count_2/2;
                        mark_region(scan, (x-dx*dist0) >> 16, (y-dy*dist0) >> 16);

                        // All subsequently detected quad region are on the same ray
                        // and have been therefore already evaluated.
                        first = false;
                    }

                    // Shift coordinates (x,y) from the border of the quad into the quad,
                    // convert from fixed-point to int, and evaluate region.
                    // Optimally (x-dx*count_0/2, y-dy*count_0/2) is in the middle of the quad.
                    mark_region(scan, (x-dx*count_0/2) >> 16, (y-dy*count_0/2) >> 16);
                }
            }

            // Append segment to the list and overwrite the last one.
            count_4 = count_3;
            count_3 = count_2;
            count_2 = count_1;
            count_1 = count_0;
            // The new segments is set to a length of 1 pixel, since we are
            // currently at the beginning. It will be increased in further iterations.
            count_0 = 1;

            // Update the state (black, white) to the color of the current pixel.
            state = pixel;
        }

        //set the flag to abort when enough quads are found and fast mode is active
        if ((fast) &&(scan->quad_count == scan->max_quad_count)) {
            running = false;
        }
        else {
            // Increase the length of the current line segment.
            count_0++;

            // Advance pixel coordinates into direction dx, dy.
            x += dx;
            y += dy;
        }
    }
}

/**
 * Scans the line through (x0, y0) and (x1, y1) for quads.
 * The line is extended and clipped at the bounds of the image.
 *
 * Valid quads are added to the list in the scan object.
 *
 * @param ScanRaster    scan object holding the image
 * @param               x coordinate of the first point
 * @param               y coordinate of the first point
 * @param               x coordinate of the second point
 * @param               y coordinate of the second point
 * @param               true when a fast scan should be made, else false
 *
 *  */
static void scan_extended_line(ScanRaster *scan, int x0, int y0, int x1, int y1, bool fast)
{
    // Compute direction vector of the line.
    float dx = x1 - x0;
    float dy = y1 - y0;

    // Extend the line beyond the image.
    // The constant factor 1000 might be suspicious, but since we are dealing with random lines,
    // overflows should not cause any problem.
    cv::Point_<int> a(x0 - 1000*dx, y0 - 1000*dy);
    cv::Point_<int> b(x0 + 1000*dx, y0 + 1000*dy);
    cv::Size_<int> size(scan->image->cols, scan->image->rows);

    // Clip line to the image bounding rectangle.
    if (cv::clipLine(size, a, b))
    {
        // normalize direction vector.
        dx = b.x - a.x;
        dy = b.y - a.y;
        float l = 1.0/sqrtf(dx*dx+dy*dy);
        dx = dx * l;
        dy = dy * l;

        // Convert direction into fixed-point.
        scan_line(scan, a.x, a.y, dx * 65536.0, dy * 65536.0, fast);
    }
}

/**
 * Initializes the scan object and create the neccessary temporary
 * images and buffers.
 *
 * @param scan              The scan object to initialize.
 * @param input             Input image from the camera.
 * @param quads             Pointer to an array, which will store the resulting quads.
 * @param max_quad_count    The maximum number of quads to detect, capacity of the array.
 */
static void scan_init(ScanRaster *scan, Mat *input, Quad *quads, int max_quad_count)
{
    scan->quads = quads;
    scan->max_quad_count = max_quad_count;
    scan->quad_count = 0;
    scan->regionId = 1;

    scan->input = input;
    scan->image = new Mat(input->rows, input->cols, DataType<uchar>::type);
    scan->temp = new Mat(input->rows, input->cols, DataType<uchar>::type);
}

/**
 * Deletes a scan object and frees all dynamically allocated buffers.
 * @param scan the scan object to free
 */
static void scan_free(ScanRaster *scan)
{
    delete scan->image;
    delete scan->temp;
}

/**
 * Scans the image for 2D quads of the raster.
 *
 * @param input             camera image, will be modified
 * @param quads             pointer to an array, which will be filled with the detected quads
 * @param max_quad_count    maximum number of quads to detect, capacity of array quads
 */
int scan_raster(Mat *input, Quad *quads, int max_quad_count)
{
    ScanRaster scan;
    scan_init(&scan, input, quads, max_quad_count);

    //log("scan_raster: max_quad_count = %i", max_quad_count);

    // binary threshold on the image
    // no adaptive threshold requires due to color balancing in smartphone camera
    cv::threshold(*input, *scan.image, 128.0, 255.0, CV_THRESH_BINARY);

    int random_lines = 0;

    // Generate random lines within the image and look for quads along these lines.
    for (int i = 0; i < RANDOM_LINES; i++)
    {
        // Generate random line (x0, y0) - (x1, y1).
        int x0 = random() % input->cols;
        int y0 = random() % input->rows;

        int x1 = random() % input->cols;
        int y1 = random() % input->rows;

        // If both coordinates are equal, adjust the second one.
        if ((x0 == x1) && (y0 == y1))
        {
            /* increase and wrap x1 */
            x1 = (x0 + 1) % input->cols;
        }

        scan_extended_line(&scan, x0, y0, x1, y1, false);
        random_lines++;

        // Exit if the maximum number of quads has been already detected.
        if (scan.quad_count >= max_quad_count)
            break;

        // arbitrary value, larger enough to cover multiple rows and to
        // ensure that the subsequent tests will find the entire grid.
        if (scan.quad_count > RANDOM_QUADS)
            break;
    }

    // Report the result of the first pass.
    //log("scan_raster: %i random lines tested, %i quads found", random_lines, scan.quad_count);

    // After several quads have been found by testing random lines, construct lines passing through
    // the already detected lines to find the remaining quads.
    // This approach is highly efficient since the quads are arranged in a regular grid.
    int count;
    int start = 0;      // start of the recently added quads.
    int iteration = 0;  // iteration number, breaks if maximum has been reached.

    do
    {
        // Save quad count to distuinguish between old and new quads.
        count = scan.quad_count;

        // Construct two orthogonal lines through every old quad.
        for (int i = start; i < count; i++)
        {
            Point2d a = lerp(quads + i, 0.5, 0.0);
            Point2d b = lerp(quads + i, 0.5, 1.0);
            Point2d c = lerp(quads + i, 0.0, 0.5);
            Point2d d = lerp(quads + i, 1.0, 0.5);


            scan_extended_line(&scan, a.x, a.y, b.x, b.y, false);
            scan_extended_line(&scan, c.x, c.y, d.x, d.y, false);
        }

        start = count; // Start in the next iteration with the recently added quads.
        iteration++;   // Increase iteration number.
    }
    while ((count != scan.quad_count) && (iteration < MAX_ITERATIONS) && (scan.quad_count < max_quad_count));

    //log("scan_raster: %i additional iterations, %i quads found", iteration, scan.quad_count);

    scan_free(&scan);

    //log("scan_raster: exit");

    return scan.quad_count;
}


