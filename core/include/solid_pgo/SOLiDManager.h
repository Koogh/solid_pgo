#pragma once

// ============================================================
// SOLiDManager.h  —  ROS-free SOLiD loop closure wrapper
// ============================================================
// Replaces SCManager from SC-PGO.
// Same external interface:
//   makeAndSaveDescriptor(cloud)  ← replaces makeAndSaveScancontextAndKeys()
//   detectLoopClosureID()         ← same name, same return type std::pair<int,float>
// NO ROS headers included here.
// ============================================================

#include <vector>
#include <utility>
#include <limits>
#include <iostream>
#include <cmath>

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// SOLiD core (pure C++, no ROS)
#include "solid_module.h"

#include "solid_pgo/common.h"

class SOLiDManager
{
public:
    // ----- Parameters (public for direct access like SCManager.NUM_EXCLUDE_RECENT) -----
    int    NUM_EXCLUDE_RECENT = 50;    // exclude latest N frames from loop search
    double loop_threshold     = 0.7;   // cosine similarity threshold (higher = stricter)

    // SOLiD sensor parameters (setters below)
    // These are passed down to SOLiDModule which has const members.
    // SOLiDModule uses compile-time constants; runtime overrides via subclass
    // or direct member modification before first use.
    // For runtime parameter support, we provide a configurable wrapper.

    // ----- API matching SCManager -----------------------------------------

    // Replaces: scManager.makeAndSaveScancontextAndKeys(*cloud)
    // Input type matches PointType (pcl::PointXYZ), same as SOLiDModule
    void makeAndSaveDescriptor(pcl::PointCloud<PointType>& scan_down)
    {
        // SOLiDModule::makeSolid expects pcl::PointCloud<pcl::PointXYZ>
        // PointType = pcl::PointXYZ, so no conversion needed
        Eigen::VectorXd descriptor = solidModule_.makeSolid(scan_down);
        descriptors_.push_back(descriptor);
    }

    // Replaces: scManager.detectLoopClosureID()
    // Returns: {matched_node_index, yaw_diff_radians}
    //          {-1, 0.0f} if no loop found
    std::pair<int, float> detectLoopClosureID(void)
    {
        int curr_idx = static_cast<int>(descriptors_.size()) - 1;
        if (curr_idx < NUM_EXCLUDE_RECENT)
            return {-1, 0.0f};

        const Eigen::VectorXd& query = descriptors_[curr_idx];

        // Brute-force search over all frames excluding recent NUM_EXCLUDE_RECENT
        // TODO: KD-tree acceleration for large-scale deployment
        double best_sim = -std::numeric_limits<double>::max();
        int best_idx = -1;

        int search_end = curr_idx - NUM_EXCLUDE_RECENT;  // inclusive upper bound
        for (int i = 0; i <= search_end; ++i)
        {
            double sim = solidModule_.loop_detection(query, descriptors_[i]);
            if (sim > best_sim)
            {
                best_sim = sim;
                best_idx = i;
            }
        }

        if (best_idx >= 0 && best_sim > loop_threshold)
        {
            // Estimate yaw difference (degrees → radians)
            double yaw_deg = solidModule_.pose_estimation(query, descriptors_[best_idx]);
            float  yaw_rad = static_cast<float>(yaw_deg * M_PI / 180.0);

            std::cout << "[SOLiD] Loop detected!  curr=" << curr_idx
                      << "  match=" << best_idx
                      << "  cosine_sim=" << best_sim
                      << "  yaw_diff=" << yaw_deg << " deg" << std::endl;

            return {best_idx, yaw_rad};
        }

        return {-1, 0.0f};
    }

    // ----- Parameter setters -----------------------------------------------

    // Equivalent to scManager.setSCdistThres()
    // Note: SOLiD uses cosine similarity (higher = more similar),
    //       so higher threshold means stricter matching.
    void setLoopThreshold(double t)    { loop_threshold     = t; }
    void setNumExcludeRecent(int n)    { NUM_EXCLUDE_RECENT = n; }

    // SOLiD-specific parameters are baked into SOLiDModule as const members.
    // If you need runtime-configurable sensor params, subclass SOLiDModule.
    // The defaults (FOV_u=2, FOV_d=-24.8, NUM_RANGE=40, NUM_ANGLE=60)
    // correspond to a 32-channel VLP-style LiDAR at 80 m range.

    // ----- Utility -----------------------------------------------------------

    int size() const { return static_cast<int>(descriptors_.size()); }

private:
    SOLiDModule solidModule_;
    std::vector<Eigen::VectorXd> descriptors_;
};
