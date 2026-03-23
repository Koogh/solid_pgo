#pragma once

// ============================================================
// PoseGraphManager.h  —  ROS-free GTSAM pose graph wrapper
// ============================================================
// Encapsulates all GTSAM state management.
// No ROS headers. Used identically by ros1/ and ros2/ nodes.
// ============================================================

#include <mutex>
#include <iostream>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/ISAM2.h>

#include "solid_pgo/common.h"

// Note: 'using namespace gtsam' is intentionally NOT placed here (header file).
// Include this header, then add 'using namespace gtsam;' in your .cpp if needed.

class PoseGraphManager
{
public:
    // Public mutex — used by node threads to protect addFactor / runISAM2opt
    std::mutex mtxPosegraph;

    PoseGraphManager()
        : isam_(nullptr), gtSAMgraphMade_(false) {}

    ~PoseGraphManager()
    {
        if (isam_) delete isam_;
    }

    // Call once after construction (before any addXxxFactor calls)
    void init()
    {
        ISAM2Params params;
        params.relinearizeThreshold = 0.01;
        params.relinearizeSkip      = 1;
        isam_ = new ISAM2(params);
        initNoises();
    }

    // -----------------------------------------------------------------------
    // Factor insertion (NOT thread-safe — caller must lock mtxPosegraph)
    // -----------------------------------------------------------------------

    // Add prior factor for the very first node
    void addPriorFactor(int idx, const Pose6D& pose)
    {
        gtsam::Pose3 p = pose6DtoGTSAM(pose);
        gtSAMgraph_.add(gtsam::PriorFactor<gtsam::Pose3>(idx, p, priorNoise_));
        initialEstimate_.insert(idx, p);
        gtSAMgraphMade_ = true;
    }

    // Add odometry (between) factor between consecutive keyframes
    void addOdomFactor(int prev_idx, int curr_idx,
                       const Pose6D& from, const Pose6D& to)
    {
        gtsam::Pose3 pFrom = pose6DtoGTSAM(from);
        gtsam::Pose3 pTo   = pose6DtoGTSAM(to);
        gtSAMgraph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
            prev_idx, curr_idx, pFrom.between(pTo), odomNoise_));
        initialEstimate_.insert(curr_idx, pTo);
    }

    // Add loop closure factor (robust Cauchy noise)
    void addLoopFactor(int prev_idx, int curr_idx,
                       const gtsam::Pose3& relative_pose)
    {
        gtSAMgraph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
            prev_idx, curr_idx, relative_pose, robustLoopNoise_));
    }

    // Add GPS altitude factor (optional)
    void addGPSFactor(int node_idx, double x, double y, double z)
    {
        gtsam::Point3 gpsConstraint(x, y, z);
        gtSAMgraph_.add(gtsam::GPSFactor(node_idx, gpsConstraint, robustGPSNoise_));
    }

    // -----------------------------------------------------------------------
    // Optimization (NOT thread-safe — caller must lock mtxPosegraph)
    // -----------------------------------------------------------------------

    void runISAM2opt()
    {
        isam_->update(gtSAMgraph_, initialEstimate_);
        isam_->update();
        gtSAMgraph_.resize(0);
        initialEstimate_.clear();
        isamCurrentEstimate_ = isam_->calculateEstimate();
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    const gtsam::Values& getEstimate() const { return isamCurrentEstimate_; }
    bool isGraphMade()                 const { return gtSAMgraphMade_;       }

    // Convenience: convert pose index → Pose6D from optimized estimate
    Pose6D getOptimizedPose(int idx) const
    {
        const gtsam::Pose3& p = isamCurrentEstimate_.at<gtsam::Pose3>(idx);
        return Pose6D{
            p.translation().x(), p.translation().y(), p.translation().z(),
            p.rotation().roll(), p.rotation().pitch(), p.rotation().yaw()
        };
    }

    int optimizedPoseCount() const
    {
        return static_cast<int>(isamCurrentEstimate_.size());
    }

    // -----------------------------------------------------------------------
    // Static utility
    // -----------------------------------------------------------------------

    static gtsam::Pose3 pose6DtoGTSAM(const Pose6D& p)
    {
        return gtsam::Pose3(
            gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw),
            gtsam::Point3(p.x, p.y, p.z));
    }

private:
    gtsam::NonlinearFactorGraph gtSAMgraph_;
    gtsam::Values               initialEstimate_;
    gtsam::ISAM2*               isam_;
    gtsam::Values               isamCurrentEstimate_;
    bool                        gtSAMgraphMade_;

    gtsam::noiseModel::Diagonal::shared_ptr priorNoise_;
    gtsam::noiseModel::Diagonal::shared_ptr odomNoise_;
    gtsam::noiseModel::Base::shared_ptr     robustLoopNoise_;
    gtsam::noiseModel::Base::shared_ptr     robustGPSNoise_;

    void initNoises()
    {
        gtsam::Vector priorNoiseVector6(6);
        priorNoiseVector6 << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12;
        priorNoise_ = gtsam::noiseModel::Diagonal::Variances(priorNoiseVector6);

        gtsam::Vector odomNoiseVector6(6);
        odomNoiseVector6 << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
        odomNoise_ = gtsam::noiseModel::Diagonal::Variances(odomNoiseVector6);

        double loopNoiseScore = 0.5;
        gtsam::Vector robustNoiseVector6(6);
        robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore,
                              loopNoiseScore, loopNoiseScore, loopNoiseScore;
        robustLoopNoise_ = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Cauchy::Create(1),
            gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6));

        gtsam::Vector robustNoiseVector3(3);
        robustNoiseVector3 << 1e9, 1e9, 250.0;
        robustGPSNoise_ = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Cauchy::Create(1),
            gtsam::noiseModel::Diagonal::Variances(robustNoiseVector3));
    }
};
