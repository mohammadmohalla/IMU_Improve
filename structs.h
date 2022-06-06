#ifndef STRUCTS_H_
#define STRUCTS_H_


struct Raster3D
{
    double rvec[3];     // rotation vector of the raster
    double tvec[3];     // translation vector of the raster
};

struct Quad
{
    cv::Point2i raster_pos;     // X,Y coordinates of the quad within the raster
    bool raster_valid;          // X,Y coordinates within the raster are valid
    cv::Point2d points[4];      // Coordinates of the four corners
};

#endif /* STRUCTS_H_ */
