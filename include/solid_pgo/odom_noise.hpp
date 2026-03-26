#pragma once

#include <string>
#include <cmath>
#include <array>

#include <Eigen/Dense>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/linear/NoiseModel.h>

#include <nav_msgs/msg/odometry.hpp>

namespace solid_pgo {

/**
 * OdomNoiseProvider
 *
 * Computes a GTSAM noise model for each odometry BetweenFactor.
 * Strategy is selected at construction time via the "mode" string:
 *
 *   "fixed"      — constant diagonal noise (default, same as SC-PGO)
 *   "distance"   — scale translational noise by traveled distance
 *   "covariance" — use pose.covariance from nav_msgs/Odometry directly
 *
 * All modes share the same base sigma parameters so you can tune once
 * and switch mode without re-tuning from scratch.
 */
class OdomNoiseProvider
{
public:
    struct Config
    {
        std::string mode = "fixed";  // "fixed" | "distance" | "covariance"

        // Base sigmas (used by all modes as a floor / reference)
        // Order: [rot_x, rot_y, rot_z, tx, ty, tz]  (GTSAM Pose3 convention)
        double sigma_rot   = 1e-3;   // [rad]
        double sigma_trans = 1e-2;   // [m]

        // "distance" mode: noise grows linearly with distance
        //   sigma_trans_actual = sigma_trans + dist_scale * distance
        double dist_scale_rot   = 1e-4;  // [rad/m]
        double dist_scale_trans = 5e-3;  // [m/m]

        // "covariance" mode: minimum sigma floor
        //   prevents degenerate (near-zero) covariance from LIO
        double min_sigma_rot   = 1e-4;
        double min_sigma_trans = 1e-3;
    };

    explicit OdomNoiseProvider(const Config& cfg = Config{}) : cfg_(cfg) {}

    /**
     * noise_for_odom()
     *
     * Returns a GTSAM noise model for one sequential odom factor.
     *
     * @param odom        The nav_msgs/Odometry of the CURRENT keyframe
     *                    (pose.covariance used in "covariance" mode)
     * @param distance    Euclidean distance traveled since the PREVIOUS keyframe
     *                    (used in "distance" mode)
     */
    gtsam::noiseModel::Base::SharedPtr noise_for_odom(
        const nav_msgs::msg::Odometry& odom,
        double distance) const
    {
        if (cfg_.mode == "distance")
            return distance_noise(distance);
        if (cfg_.mode == "covariance")
            return covariance_noise(odom);
        return fixed_noise();
    }

    /** Convenience: fixed noise (no odom msg needed) */
    gtsam::noiseModel::Diagonal::shared_ptr fixed_noise() const
    {
        gtsam::Vector6 s;
        s << cfg_.sigma_rot,   cfg_.sigma_rot,   cfg_.sigma_rot,
             cfg_.sigma_trans, cfg_.sigma_trans, cfg_.sigma_trans;
        return gtsam::noiseModel::Diagonal::Sigmas(s);
    }

private:
    Config cfg_;

    gtsam::noiseModel::Diagonal::shared_ptr distance_noise(double dist) const
    {
        double sr = cfg_.sigma_rot   + cfg_.dist_scale_rot   * dist;
        double st = cfg_.sigma_trans + cfg_.dist_scale_trans * dist;
        gtsam::Vector6 s;
        s << sr, sr, sr, st, st, st;
        return gtsam::noiseModel::Diagonal::Sigmas(s);
    }

    gtsam::noiseModel::Base::SharedPtr covariance_noise(
        const nav_msgs::msg::Odometry& odom) const
    {
        // nav_msgs/Odometry pose.covariance is row-major 6×6
        // order: [x, y, z, rot_x, rot_y, rot_z]
        // GTSAM Pose3 sigma order: [rot_x, rot_y, rot_z, x, y, z]
        const auto& c = odom.pose.covariance;  // std::array<double,36>

        // Check if covariance is all-zero (not provided by the LIO)
        bool all_zero = true;
        for (double v : c) { if (v != 0.0) { all_zero = false; break; } }
        if (all_zero)
            return fixed_noise();  // fall back gracefully

        // Build 6×6 Eigen matrix from row-major array
        // ROS covariance order: [x,y,z, rx,ry,rz]
        // GTSAM order:          [rx,ry,rz, x,y,z]
        // Extract diagonal variances and re-order
        std::array<int,6> ros_diag_idx = {0, 7, 14, 21, 28, 35};
        //  ROS indices:    x=0  y=7  z=14  rx=21  ry=28  rz=35
        double var_x  = c[ros_diag_idx[0]];
        double var_y  = c[ros_diag_idx[1]];
        double var_z  = c[ros_diag_idx[2]];
        double var_rx = c[ros_diag_idx[3]];
        double var_ry = c[ros_diag_idx[4]];
        double var_rz = c[ros_diag_idx[5]];

        // Apply minimum floor
        auto safe_sigma = [](double var, double min_s) {
            double s = std::sqrt(std::max(var, 0.0));
            return std::max(s, min_s);
        };

        gtsam::Vector6 sigmas;
        // GTSAM Pose3: [rx, ry, rz, x, y, z]
        sigmas << safe_sigma(var_rx, cfg_.min_sigma_rot),
                  safe_sigma(var_ry, cfg_.min_sigma_rot),
                  safe_sigma(var_rz, cfg_.min_sigma_rot),
                  safe_sigma(var_x,  cfg_.min_sigma_trans),
                  safe_sigma(var_y,  cfg_.min_sigma_trans),
                  safe_sigma(var_z,  cfg_.min_sigma_trans);

        return gtsam::noiseModel::Diagonal::Sigmas(sigmas);
    }
};

}  // namespace solid_pgo
