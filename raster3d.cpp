#include <math.h>
#include <vector>
#include <stddef.h>

#include <opencv2/core/types_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "raster3d.h"

#define QUAD_SIZE               2.0
#define GAP_SIZE                (2.0/3.0)


#define log(...) __android_log_print(ANDROID_LOG_INFO, "SensorTest", __VA_ARGS__)

using namespace cv;
using namespace std;

/**
 * Calculates the distance between two points.
 * @param a     first point
 * @param b     second point
 * @return      the distance between a and b
 */
static inline double distance(Point2d *a, Point2d *b)
{
    double dx = b->x-a->x;
    double dy = b->y-a->y;
    return sqrt(dx*dx+dy*dy);
}

/**
 * Estimates the relative position between two quads in the raster.
 * possible locations for (dx, dy)
 *
 *          (0,-1)
 *            b
 *  (-1,0)  b a b (1,0)
 *            b
 *          (0,1)
 *
 * @param a     reference quad
 * @param b     quad, whose relation to a should be determined
 * @param dx    resulting column difference, x coordinate in the raster
 * @param dx    resulting row difference, x coordinate in the raster
 * @result      returns true if a relation could be determined, false otherwise
 */
static bool get_relative_position(Quad *a, Quad *b, int *dx, int *dy)
{
    int best_side = -1;
    double best_dist = 10000000;

    // Check all four sides.
    for (int i = 0; i < 4; i++)
    {
        // j is the starting point of the side of the neighbour quad.
        int j = (i+2)%4;

        double dist_a = distance(&a->points[i], &b->points[(j+1)%4]);
        double dist_b = distance(&a->points[(i+1)%4], &b->points[j]);
        double dist = dist_a + dist_b;

        /* Compute pair of sides with shortest distance. */
        if (dist < best_dist)
        {
            best_side = i;
            best_dist = dist;
        }
    }

    if (best_side < 0)
        return false;

    double side_a = distance(&a->points[best_side], &a->points[(best_side+1)%4]);
    double side_b = distance(&b->points[(best_side+2)%4], &b->points[(best_side+3)%4]);

    //  If the shortest distance between two sides is longer than the length of each side, the quads are not neighboured.
    if ((best_dist > 1.5 * side_a) || (best_dist > 1.5 * side_b))
    {
        return false;
    }

    // Compute relative direction from index of the side. This value depends on the orientation of the quads.
    switch (best_side)
    {
        case 0: *dx = 0; *dy = -1; break;
        case 1: *dx = 1; *dy = 0; break;
        case 2: *dx = 0; *dy = 1; break;
        case 3: *dx = -1; *dy = 0; break;
    }

    return true;
}

/**
 * Determined relative raster coordinates of the quads by
 * iteratively attaching neighboring quads.
 * @param quads         array of quads
 * @param quad_count    number of quads
 */
static void assign_raster_coords(Quad *quads, int quad_count)
{
    //  If there are not quads, exit immediatly. 
    if (quad_count == 0)
        return;

    // Insert the first quad into the list and assign coordinates (0,0).
    // We are only interested in relative coordinates and the raster
    // also does not allow to identify quads with global coordinates.
    quads[0].raster_pos.x = 0;
    quads[0].raster_pos.y = 0;
    quads[0].raster_valid = true;

    // All other quads do not have valid raster coordinates.
    for (int i = 1; i < quad_count; i++)
    {
        quads[i].raster_valid = false;
    }

    int connect_count = 1;
    int iteration = 0;

    // Iterate over the array until no progress can be made.
    bool changed;
    do
    {
        changed = false;

        for (int i = 0; i < quad_count; i++)
        {
            Quad *a = quads + i;
            if (!a->raster_valid) continue;

            // The quad a with valid raster coordinates is in the array. Find a neighboring quads.
            for (int j = 0; j < quad_count; j++)
            {
                // Check is quad b is a neighbor of quad a.
                Quad *b = quads + j;

                // Check unassigned quads only.
                if (!b->raster_valid)
                {
                    int dx, dy;

                    // Determine realtive position.
                    if (get_relative_position(a, b, &dx, &dy))
                    {
                        // Compute coordinates of quad b from the coordinates of a and the relation between the two quads a and b.
                        b->raster_pos.x = a->raster_pos.x + dx;
                        b->raster_pos.y = a->raster_pos.y + dy;
                        b->raster_valid = true;
                        connect_count++;
                        changed = true;
                    }
                }
            }

        }
        iteration++;
    } while (changed && (iteration < 100));
}

/**
 * Removes all quads from the array without a valid raster position.
 * Since we require global coordinates to calculate the 3D rotation,
 * the relation between quads in the raster must be known.
 * @param array     array of quads
 * @param count     number of quads in the array, will be modified if a quad is removed
 */
static void remove_unassigned_quads(Quad *quads, int *count)
{
    int initial = *count;

    for (int i = *count - 1; i >= 0; i--)
    {
        if (!quads[i].raster_valid)
        {
            for (int j = i + 1; j < *count; j++)
            {
                quads[j-1] = quads[j];
            }
            *count = *count - 1;
        }
    }

    //log("raster3d: removed %i unconnected of %i quads", initial - *count, initial);
}

/**
 * Adds the coordinates of the detected 2D to the vector.
 * @param quads     array of quads
 * @param count     number of quads
 * @param points    pointer to vector
 */
static void add_quads_2d(Quad *quads, int count, vector<Point2f> *points)
{
    for (int j = 0; j < count; j++)
    {
        Quad *q = quads + j;

        // Each quad has always four corners.
        for (int j = 0; j < 4; j++)
        {
            points->push_back(Point2f(q->points[j].x, q->points[j].y));
        }
    }
}

static void remove_offset(Quad *quads, int count) {
    int offsetX = 0;
    int offsetY = 0;
    
    //get the offset of the quads
    for (int j = 0; j < count; j++) {
        Quad *q = quads + j;
        offsetX += q->raster_pos.x;
        offsetY += q->raster_pos.y;
    }
    offsetX /= 9;
    offsetY /= 9;
    
    //assign the offset to the quads (the one in the middle is 0,0)
    for (int j = 0; j < count; j++) {
        Quad *q = quads + j;
        q->raster_pos.x -= offsetX;
        q->raster_pos.y -= offsetY;
    }
}


/**
 * Adds the 3D coordinates of the given quads to the vector.
 * @param quads     array of quads
 * @param count     number of quads
 * @param points    pointer to vector
 */
static void add_quads_3d(Quad *quads, int count, vector<Point3f> *points, float quadSize)
{
    float gapSize = quadSize/3.0f;
    
    for (int j = 0; j < count; j++)
    {
        Quad *q = quads + j;

        // Compute the center of the quad from its position in the raster.

        float offset_x = q->raster_pos.x * (quadSize + gapSize);
        float offset_y = q->raster_pos.y * (quadSize + gapSize);

        // The relative coordinates of the corners.
        float quad[4][2] = { {-1,-1}, {1,-1}, {1,1},{-1,1}};

        for (int i = 0; i < 4;i++)
        {
            float px = quad[i][0] * quadSize/2 + offset_x;
            float py = quad[i][1] * quadSize/2 + offset_y;

            points->push_back(Point3f(px, py, 0));
        }
    }
}

/**
 * Sets the values of 3x3 matrix from a given array in row major order.
 *
 * [0] [1] [2]
 * [3] [4] [5]
 * [6] [7] [8]
 *
 * @param p         the 3x3 double matrix to be updated
 * @param items     pointer to array of 9 double values
 */
static void set_3x3_matrix(Mat *p, double *items)
{
    p->at<double>(0,0) = items[0]; p->at<double>(0,1) = items[1]; p->at<double>(0,2) = items[2];
    p->at<double>(1,0) = items[3]; p->at<double>(1,1) = items[4]; p->at<double>(1,2) = items[5];
    p->at<double>(2,0) = items[6]; p->at<double>(2,1) = items[7]; p->at<double>(2,2) = items[8];
}

/**
 * Initializes the orientation of the 3D Raster from an array of quads.
 * @param quads         array of quads
 * @param count         number of quads in the array
 * @param projection    3x3 projection matrix, row major layout
 * @param raster        raster3d to be initialized
 * @param result        true if sucessful, false otherwise
 * @param fast          when set to true, a faster but less precise version is used
 * @param first         indicator whether this is the first image, only used when fast==true
 */
bool get_raster_3d(Quad *quads, int count, double *projection, Raster3D *raster, float quadSize)
{
    if (count == 0)
        return false;

    // Associate raster coordinates (x,y) with each quad, ignore unassigned quads
    assign_raster_coords(quads, count);
    remove_unassigned_quads(quads, &count);

    // If the requires minimum number of quads has not been found, return an invalid result.
    if(count < 9){
        return false;
    }

    vector<Point2f> image_points;
    vector<Point3f> obj_points;

    //remove the offset from the quads 
    remove_offset(quads, count);
    
    // Initialize 2D screen coordinates of the quads
    add_quads_2d(quads, count, &image_points);

    // Initialize 3D relative coordinates of the quads
    add_quads_3d(quads, count, &obj_points, quadSize);

    Mat p = Mat(3, 3, CV_64F);
    set_3x3_matrix(&p, projection);

    Mat rvec = Mat(3,1 , CV_64F);
    Mat tvec = Mat(3,1 , CV_64F);


    bool result = false;

    result = solvePnP(Mat(obj_points), Mat(image_points), p, Mat(), rvec, tvec, false, CV_ITERATIVE);
    

    // Store rotation (rvec) and translation (tvec) vectors in the raster.
    for (int i = 0; i < 3; i++)
    {
        raster->rvec[i] = rvec.at<double>(i, 0);
        raster->tvec[i] = tvec.at<double>(i, 0);
    }

    return result;
}

/**
 * Calculates the rotation matrix of a 3d raster.
 * @param raster    the 3D raster
 * @param matrix    3x3 matrix in row major, will contain rotation matrix
 */
void get_pose(Raster3D *raster, double *matrix)
{
    Mat rmat = Mat(3, 3, CV_64F);
    Mat rvec = Mat(3, 1, CV_64F);
    Mat tvec = Mat(3, 1, CV_64F);

    for (int i = 0; i < 3; i++)
    {
        rvec.at<double>(i,0) = raster->rvec[i];
    }
    for (int i = 0; i < 3; i++) {
        tvec.at<double>(i,0) = raster->tvec[i];
    }

    // OpenCV function to convert rotation vector into rotation matrix.
    Rodrigues(rvec, rmat);

    // the orientation
    matrix[0] = rmat.at<double>(0, 0); matrix[1] = rmat.at<double>(0, 1); matrix[2] = rmat.at<double>(0, 2);
    matrix[3] = rmat.at<double>(1, 0); matrix[4] = rmat.at<double>(1, 1); matrix[5] = rmat.at<double>(1, 2);
    matrix[6] = rmat.at<double>(2, 0); matrix[7] = rmat.at<double>(2, 1); matrix[8] = rmat.at<double>(2, 2);
    
    // convert the position onto object space and
    Mat cmT = -rmat.t()*tvec;
    matrix[9] = cmT.at<double>(0,0); matrix[10] = cmT.at<double>(1,0); matrix[11] = cmT.at<double>(2,0);
}

/**
 * Reprojects and draws the array of quads onto the image
 * @param quads         array of quads to be drawm
 * @param count         number of quads in the array
 * @param projection    3x3 projection matrix in row major
 * @param raster        raster3d used for orientation
 * @param image         target image
 */
void draw_quads_3d (Quad *quads, int count, double *projection, Raster3D *raster, Mat *image, float quadSize)
{
    vector<Point3f> obj_points;
    add_quads_3d(quads, count, &obj_points, quadSize);

    Mat rvec = Mat(3, 1, CV_64F);
    Mat tvec = Mat(3, 1, CV_64F);

    for (int i = 0; i < 3; i++)
    {
        rvec.at<double>(i,0) = raster->rvec[i];
        tvec.at<double>(i,0) = raster->tvec[i];
    }

    Mat p = Mat(3, 3, CV_64F);
    set_3x3_matrix(&p, projection);

    vector<Point2f> reprojected;
    projectPoints(obj_points, rvec, tvec, p, Mat(), reprojected);

    Scalar black = Scalar(0,0,0,255);
    Scalar white = Scalar(255,255,255,255);

    for (int i = 0; i < reprojected.size(); i++)
    {
        circle(*image, reprojected.at(i), 3, ((i%4)==0) ? black : white , -1);
    }
}
