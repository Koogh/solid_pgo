#pragma once

#include <cmath>
#include <vector>
#include <iostream>
#include <Eigen/Dense>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

// Adapted from SOLiD (https://github.com/sparolab/solid)
// Paper: "SOLID: Spatially Organized and Lightweight Global Descriptor for FOV-constrained LiDAR Place Recognition"

typedef pcl::PointXYZI PointType;

inline float calc_dist(float x, float y, float z) { return sqrtf(x*x + y*y + z*z); }
inline float rad2deg(float rad) { return rad * 180.0f / M_PI; }
inline float deg2rad(float deg) { return deg * M_PI / 180.0f; }

struct RAH
{
    int idx_range  = 0;
    int idx_angle  = 0;
    int idx_height = 0;
};

class SOLiDModule
{
public:
    // LiDAR vertical FOV (degrees) — tune to your sensor
    float FOV_u = 2.0f;     // upper bound (VLP-16: 15, OS1-64: 22.5)
    float FOV_d = -24.8f;   // lower bound (VLP-16: -15, OS1-64: -22.5)

    int   NUM_ANGLE    = 60;
    int   NUM_RANGE    = 40;
    int   NUM_HEIGHT   = 32;
    int   MIN_DISTANCE = 3;
    int   MAX_DISTANCE = 80;
    float VOXEL_SIZE   = 0.4f;

    // ---- descriptor ----
    Eigen::VectorXd makeSolid(pcl::PointCloud<PointType>& scan_down);

    // ---- loop detection: cosine similarity on range part, in [0,1] ----
    double loop_detection(const Eigen::VectorXd& query, const Eigen::VectorXd& candidate);

    // ---- pose estimation: returns yaw offset in degrees [0, 360) ----
    double pose_estimation(const Eigen::VectorXd& query, const Eigen::VectorXd& candidate);

    // ---- point cloud utilities ----
    void remove_far_points(pcl::PointCloud<PointType>& scan_raw,
                           pcl::PointCloud<PointType>::Ptr scan_out);
    void remove_closest_points(pcl::PointCloud<PointType>& scan_raw,
                                pcl::PointCloud<PointType>::Ptr scan_out);
    void down_sampling(pcl::PointCloud<PointType>& scan_raw,
                       pcl::PointCloud<PointType>::Ptr scan_down);

private:
    RAH pt2rah(PointType& point, float gap_angle, float gap_range, float gap_height);

    static float xy2theta(float x, float y)
    {
        if (x == 0.0f) x = 0.001f;
        if (y == 0.0f) y = 0.001f;
        if (x >= 0 && y >= 0) return rad2deg(atan2f(y, x));
        if (x <  0 && y >= 0) return 180.0f - rad2deg(atan2f(y, -x));
        if (x <  0 && y <  0) return 180.0f + rad2deg(atan2f(-y, -x));
        /* x >= 0 && y <  0 */ return 360.0f - rad2deg(atan2f(-y, x));
    }
};

// ============================================================
//  Inline implementation (header-only for easy inclusion)
// ============================================================

inline RAH SOLiDModule::pt2rah(PointType& point,
                                float gap_angle, float gap_range, float gap_height)
{
    RAH rah;
    float px = point.x, py = point.y, pz = point.z;

    float theta   = xy2theta(px, py);
    float dist_xy = sqrtf(px*px + py*py);
    float phi     = rad2deg(atan2f(pz, dist_xy));

    rah.idx_range  = std::min((int)(dist_xy / gap_range),  NUM_RANGE  - 1);
    rah.idx_angle  = std::min((int)(theta   / gap_angle),  NUM_ANGLE  - 1);
    rah.idx_height = std::min((int)((phi - FOV_d) / gap_height), NUM_HEIGHT - 1);

    if (rah.idx_height < 0)           rah.idx_height = 0;
    if (rah.idx_height >= NUM_HEIGHT) rah.idx_height = NUM_HEIGHT - 1;

    return rah;
}

inline Eigen::VectorXd SOLiDModule::makeSolid(pcl::PointCloud<PointType>& scan_down)
{
    Eigen::MatrixXd range_matrix = Eigen::MatrixXd::Zero(NUM_RANGE,  NUM_HEIGHT);
    Eigen::MatrixXd angle_matrix = Eigen::MatrixXd::Zero(NUM_ANGLE,  NUM_HEIGHT);

    float gap_angle  = 360.0f / NUM_ANGLE;
    float gap_range  = static_cast<float>(MAX_DISTANCE) / NUM_RANGE;
    float gap_height = (FOV_u - FOV_d) / NUM_HEIGHT;

    for (auto& pt : scan_down.points)
    {
        RAH rah = pt2rah(pt, gap_angle, gap_range, gap_height);
        range_matrix(rah.idx_range,  rah.idx_height) += 1.0;
        angle_matrix(rah.idx_angle,  rah.idx_height) += 1.0;
    }

    // weight vector: normalized column sums of range_matrix
    Eigen::VectorXd weight = Eigen::VectorXd::Zero(NUM_HEIGHT);
    for (int c = 0; c < NUM_HEIGHT; ++c)
        weight(c) = range_matrix.col(c).sum();

    double wmin = weight.minCoeff();
    double wmax = weight.maxCoeff();
    if (wmax - wmin > 1e-9)
        weight = (weight.array() - wmin) / (wmax - wmin);

    Eigen::VectorXd solid(NUM_RANGE + NUM_ANGLE);
    solid.head(NUM_RANGE) = range_matrix * weight;
    solid.tail(NUM_ANGLE) = angle_matrix * weight;
    return solid;
}

inline double SOLiDModule::loop_detection(const Eigen::VectorXd& query,
                                           const Eigen::VectorXd& candidate)
{
    // Cosine similarity on the RANGE part
    Eigen::VectorXd r_q = query.head(NUM_RANGE);
    Eigen::VectorXd r_c = candidate.head(NUM_RANGE);
    double denom = r_q.norm() * r_c.norm();
    if (denom < 1e-9) return 0.0;
    return r_q.dot(r_c) / denom;
}

inline double SOLiDModule::pose_estimation(const Eigen::VectorXd& query,
                                            const Eigen::VectorXd& candidate)
{
    // L1-norm shift search on the ANGLE part → yaw offset estimate
    Eigen::VectorXd a_q = query.tail(NUM_ANGLE);
    Eigen::VectorXd a_c = candidate.tail(NUM_ANGLE);

    double minDist = std::numeric_limits<double>::max();
    int    minIdx  = 0;

    for (int shift = 0; shift < NUM_ANGLE; ++shift)
    {
        Eigen::VectorXd shifted = Eigen::VectorXd::Zero(NUM_ANGLE);
        for (int i = 0; i < NUM_ANGLE; ++i)
            shifted((i + shift) % NUM_ANGLE) = a_q(i);
        double dist = (a_c - shifted).cwiseAbs().sum();
        if (dist < minDist) { minDist = dist; minIdx = shift; }
    }

    return (minIdx + 1) * (360.0 / NUM_ANGLE);
}

inline void SOLiDModule::remove_far_points(pcl::PointCloud<PointType>& scan_raw,
                                            pcl::PointCloud<PointType>::Ptr scan_out)
{
    for (auto& pt : scan_raw.points)
        if (calc_dist(pt.x, pt.y, pt.z) < MAX_DISTANCE)
            scan_out->points.push_back(pt);
}

inline void SOLiDModule::remove_closest_points(pcl::PointCloud<PointType>& scan_raw,
                                                 pcl::PointCloud<PointType>::Ptr scan_out)
{
    for (auto& pt : scan_raw.points)
        if (calc_dist(pt.x, pt.y, pt.z) > MIN_DISTANCE)
            scan_out->points.push_back(pt);
}

inline void SOLiDModule::down_sampling(pcl::PointCloud<PointType>& scan_raw,
                                        pcl::PointCloud<PointType>::Ptr scan_down)
{
    pcl::VoxelGrid<PointType> vg;
    vg.setInputCloud(scan_raw.makeShared());
    vg.setLeafSize(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE);
    vg.filter(*scan_down);
}
