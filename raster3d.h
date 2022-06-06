#ifndef RASTER3D_H_
#define RASTER3D_H_

#include <opencv2/core/core.hpp>
#include "structs.h"
#include "scanraster.h"
#include <stddef.h>

bool    get_raster_3d       (Quad *quads, int count, double *projection, Raster3D *raster, float quadSize);
void    get_pose            (Raster3D *raster, double *matrix);
void    draw_quads_3d       (Quad *quads, int count, double *projection, Raster3D *raster, cv::Mat *image);


#endif /* RASTER3D_H_ */
