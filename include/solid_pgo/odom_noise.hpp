#pragma once

#include <string>
#include <cmath>
#include <array>

#include <gtsam/linear/NoiseModel.h>
#include <nav_msgs/msg/odometry.hpp>

namespace solid_pgo {

// ---- Config is defined outside the class to avoid GCC nested-struct
//      default-initializer + default-argument interaction bug ----
struct OdomNoiseConfig
{
    std::string mode = "fixed";  // "fixed" | "distance" | "covariance"

    // Base sigmas shared by all modes
    // GTSAM Pose3 convention: [rot_x, rot_y, rot_z, tx, ty, tz]
    double sigma_rot   = 1e-3;   // [rad]
    double sigma_trans = 1e-2;   // [m]

    // "distance" mode: sigma += scale * distance_since_last_keyframe
    double dist_scale_rot   = 1e-4;  // [rad/m]
    double dist_scale_trans = 5e-3;  // [m/m]

    // "covariance" mode: minimum sigma floor (guards against near-zero LIO output)
    double min_sigma_rot   = 1e-4;
    double min_sigma_trans = 1e-3;
};

/**
 * OdomNoiseProvider
 *
 * Returns a per-keyframe GTSAM noise model for odometry BetweenFactors.
 * Strategy is selected at construction time via OdomNoiseConfig::mode:
 *
 *   "fixed"      — constant diagonal noise (default, always works)
 *   "distance"   — noise grows linearly with distance traveled
 *   "covariance" — use pose.covariance from nav_msgs/Odometry;
 *                  falls back to "fixed" if covariance is all-zero
 */
class OdomNoiseProvider
{
public:
    // Keep Config as a type alias for backward compatibility
    using Config = OdomNoiseConfig;

    OdomNoiseProvider() : cfg_(OdomNoiseConfig{}) {}
    explicit OdomNoiseProvider(const OdomNoiseConfig& cfg) : cfg_(cfg) {}

    /**
     * noise_for_odom()
     *
     * @param odom      Current keyframe's raw nav_msgs/Odometry
     *                  (pose.covariance used in "covariance" mode)
     * @param distance  Distance [m] traveled since previous keyframe
     *                  (used in "distance" mode)
     */
    gtsam::noiseModel::Base::shared_ptr noise_for_odom(
        const nav_msgs::msg::Odometry& odom,
        double distance) const
    {
        if (cfg_.mode == "distance")
            return distance_noise(distance);
        if (cfg_.mode == "covariance")
            return covariance_noise(odom);
        return fixed_noise();
    }

    /** Convenience accessor — fixed noise regardless of odom msg */
    gtsam::noiseModel::Diagonal::shared_ptr fixed_noise() const
    {
        gtsam::Vector6 s;
        s << cfg_.sigma_rot,   cfg_.sigma_rot,   cfg_.sigma_rot,
             cfg_.sigma_trans, cfg_.sigma_trans, cfg_.sigma_trans;
        return gtsam::noiseModel::Diagonal::Sigmas(s);
    }

private:
    OdomNoiseConfig cfg_;

    gtsam::noiseModel::Diagonal::shared_ptr distance_noise(double dist) const
    {
        double sr = cfg_.sigma_rot   + cfg_.dist_scale_rot   * dist;
        double st = cfg_.sigma_trans + cfg_.dist_scale_trans * dist;
        gtsam::Vector6 s;
        s << sr, sr, sr, st, st, st;
        return gtsam::noiseModel::Diagonal::Sigmas(s);
    }

    gtsam::noiseModel::Base::shared_ptr covariance_noise(
        const nav_msgs::msg::Odometry& odom) const
    {
        // ROS covariance: row-major 6×6, order [x, y, z, rx, ry, rz]
        // GTSAM Pose3 order:                   [rx, ry, rz, x, y, z]
        const auto& c = odom.pose.covariance;  // std::array<double,36>

        // Fall back to fixed if LIO does not provide covariance
        bool all_zero = true;
        for (double v : c) { if (v != 0.0) { all_zero = false; break; } }
        if (all_zero)
            return fixed_noise();

        // Diagonal indices in row-major 6×6: x=0, y=7, z=14, rx=21, ry=28, rz=35
        auto safe_sigma = [](double var, double min_s) -> double {
            return std::max(std::sqrt(std::max(var, 0.0)), min_s);
        };

        gtsam::Vector6 sigmas;
        // GTSAM: [rx, ry, rz, x, y, z]
        sigmas << safe_sigma(c[21], cfg_.min_sigma_rot),
                  safe_sigma(c[28], cfg_.min_sigma_rot),
                  safe_sigma(c[35], cfg_.min_sigma_rot),
                  safe_sigma(c[0],  cfg_.min_sigma_trans),
                  safe_sigma(c[7],  cfg_.min_sigma_trans),
                  safe_sigma(c[14], cfg_.min_sigma_trans);

        return gtsam::noiseModel::Diagonal::Sigmas(sigmas);
    }
};

}  // namespace solid_pgo
