#ifndef SCANRASTER_H_
#define SCANRASTER_H_

#include <opencv2/core/core.hpp>
#include "structs.h"

int scan_raster(cv::Mat *input, Quad *quads, int max_quad_count);

#endif /* SCANRASTER_H_ */
