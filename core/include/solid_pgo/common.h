#pragma once

#include <cmath>
#include <pcl/point_types.h>

// PointType: PointXYZ (SOLiD-compatible, no intensity channel needed)
#ifndef SOLID_PGO_POINTTYPE_DEFINED
#define SOLID_PGO_POINTTYPE_DEFINED
typedef pcl::PointXYZ PointType;
#endif

inline double rad2deg(double radians) { return radians * 180.0 / M_PI; }
inline double deg2rad(double degrees) { return degrees * M_PI / 180.0; }

struct Pose6D {
    double x;
    double y;
    double z;
    double roll;
    double pitch;
    double yaw;
};
