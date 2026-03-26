/**
 * solid_pgo_node.cpp
 *
 * SLAM back-end for ROS 2 Humble.
 *  - Front-end  : any LIO that publishes PointCloud2 + nav_msgs/Odometry
 *                 (default: rko_lio topics)
 *  - Loop det.  : SOLiD descriptor (cosine sim on range part)
 *  - Pose estim.: SOLiD angle-shift (yaw initial guess for ICP)
 *  - PGO        : GTSAM iSAM2  (structure inspired by SC-PGO)
 *
 * Architecture (three threads):
 *   process_pg()   – keyframe management, sequential odom factors
 *   process_lcd()  – loop detection at configurable Hz
 *   process_icp()  – ICP verification + loop factor insertion
 */

#include <fstream>
#include <math.h>
#include <vector>
#include <deque>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include <optional>
#include <filesystem>

#include <Eigen/Dense>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>

#include "solid_pgo/solid_module.hpp"
#include "solid_pgo/odom_noise.hpp"

using namespace gtsam;

// ============================================================
//  Pose6D helper
// ============================================================
struct Pose6D
{
    double x, y, z;
    double roll, pitch, yaw;
};

inline gtsam::Pose3 pose6dToGtsam(const Pose6D& p)
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw),
                        gtsam::Point3(p.x, p.y, p.z));
}

inline Pose6D gtsamToPose6d(const gtsam::Pose3& p)
{
    return Pose6D{p.translation().x(), p.translation().y(), p.translation().z(),
                  p.rotation().roll(), p.rotation().pitch(), p.rotation().yaw()};
}

Pose6D odomTopose6d(const nav_msgs::msg::Odometry::ConstSharedPtr& odom)
{
    auto& pos = odom->pose.pose.position;
    auto& q   = odom->pose.pose.orientation;

    double roll, pitch, yaw;
    // quat → RPY
    double sinr = 2.0 * (q.w * q.x + q.y * q.z);
    double cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
    roll = atan2(sinr, cosr);

    double sinp = 2.0 * (q.w * q.y - q.z * q.x);
    if (fabs(sinp) >= 1.0)
        pitch = copysign(M_PI / 2.0, sinp);
    else
        pitch = asin(sinp);

    double siny = 2.0 * (q.w * q.z + q.x * q.y);
    double cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    yaw = atan2(siny, cosy);

    return Pose6D{pos.x, pos.y, pos.z, roll, pitch, yaw};
}

Pose6D diffTransformation(const Pose6D& p1, const Pose6D& p2)
{
    Eigen::Affine3f T1 = pcl::getTransformation((float)p1.x, (float)p1.y, (float)p1.z,
                                                  (float)p1.roll, (float)p1.pitch, (float)p1.yaw);
    Eigen::Affine3f T2 = pcl::getTransformation((float)p2.x, (float)p2.y, (float)p2.z,
                                                  (float)p2.roll, (float)p2.pitch, (float)p2.yaw);
    Eigen::Matrix4f delta = T1.matrix().inverse() * T2.matrix();
    Eigen::Affine3f dA; dA.matrix() = delta;
    float dx, dy, dz, dr, dp, dyw;
    pcl::getTranslationAndEulerAngles(dA, dx, dy, dz, dr, dp, dyw);
    return Pose6D{fabs(dx), fabs(dy), fabs(dz), fabs(dr), fabs(dp), fabs(dyw)};
}

pcl::PointCloud<PointType>::Ptr local2global(
    const pcl::PointCloud<PointType>::Ptr& cloud, const Pose6D& tf)
{
    pcl::PointCloud<PointType>::Ptr out(new pcl::PointCloud<PointType>());
    Eigen::Affine3f T = pcl::getTransformation((float)tf.x, (float)tf.y, (float)tf.z,
                                                (float)tf.roll, (float)tf.pitch, (float)tf.yaw);
    pcl::transformPointCloud(*cloud, *out, T);
    return out;
}

// ============================================================
//  SolidPgoNode
// ============================================================
class SolidPgoNode : public rclcpp::Node
{
public:
    SolidPgoNode() : Node("solid_pgo_node")
    {
        loadParams();
        initGTSAM();

        // Publishers
        pub_path_     = create_publisher<nav_msgs::msg::Path>("solid_pgo/path", 1);
        pub_map_      = create_publisher<sensor_msgs::msg::PointCloud2>("solid_pgo/map", 1);
        pub_odom_     = create_publisher<nav_msgs::msg::Odometry>("solid_pgo/odometry", 1);
        pub_loop_src_ = create_publisher<sensor_msgs::msg::PointCloud2>("solid_pgo/loop_scan", 1);
        pub_loop_tgt_ = create_publisher<sensor_msgs::msg::PointCloud2>("solid_pgo/loop_submap", 1);
        pub_loop_vis_ = create_publisher<visualization_msgs::msg::MarkerArray>("solid_pgo/loop_markers", 1);

        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        // Subscribers — separate queues, manual time sync (same pattern as SC-PGO)
        sub_odom_  = create_subscription<nav_msgs::msg::Odometry>(
            odom_topic_, 100,
            [this](const nav_msgs::msg::Odometry::ConstSharedPtr msg) {
                std::lock_guard<std::mutex> lk(buf_mutex_);
                odom_buf_.push(msg);
            });

        sub_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            cloud_topic_, 10,
            [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
                std::lock_guard<std::mutex> lk(buf_mutex_);
                cloud_buf_.push(msg);
            });

        // Worker threads
        thread_pg_  = std::thread(&SolidPgoNode::process_pg,  this);
        thread_lcd_ = std::thread(&SolidPgoNode::process_lcd, this);
        thread_icp_ = std::thread(&SolidPgoNode::process_icp, this);

        RCLCPP_INFO(get_logger(), "[solid_pgo] Started. cloud='%s'  odom='%s'",
                    cloud_topic_.c_str(), odom_topic_.c_str());
    }

    ~SolidPgoNode()
    {
        running_ = false;
        if (thread_pg_.joinable())  thread_pg_.join();
        if (thread_lcd_.joinable()) thread_lcd_.join();
        if (thread_icp_.joinable()) thread_icp_.join();
    }

private:
    // ---- parameters ----
    std::string odom_topic_;
    std::string cloud_topic_;
    std::string map_frame_;
    double keyframe_meter_gap_;
    double keyframe_rad_gap_;
    double loop_closure_frequency_;
    double solid_loop_threshold_;     // cosine similarity threshold (lower = stricter)
    int    solid_num_exclude_recent_; // exclude recent N frames from loop candidates
    int    loop_submap_size_;
    double icp_fitness_threshold_;
    double map_voxel_size_;
    double keyframe_voxel_size_;
    bool   save_map_;
    std::string save_dir_;

    // ---- SOLID ----
    SOLiDModule solid_;
    std::vector<Eigen::VectorXd> solid_keys_;  // one per keyframe

    // ---- keyframe data ----
    std::vector<Pose6D>  kf_poses_;
    std::vector<Pose6D>  kf_poses_updated_;
    std::vector<double>  kf_times_;
    std::vector<double>  kf_distances_;       // distance from prev keyframe [m]
    std::vector<nav_msgs::msg::Odometry> kf_odoms_;  // raw odom msg (for covariance)
    std::vector<pcl::PointCloud<PointType>::Ptr> kf_clouds_;
    std::mutex kf_mutex_;

    // ---- GTSAM ----
    NonlinearFactorGraph graph_;
    Values               initial_estimate_;
    ISAM2*               isam_        = nullptr;
    Values               isam_estimate_;
    bool                 graph_ready_ = false;

    noiseModel::Diagonal::shared_ptr prior_noise_;
    noiseModel::Base::shared_ptr     loop_noise_;

    solid_pgo::OdomNoiseProvider odom_noise_provider_;

    std::mutex pg_mutex_;
    int recent_idx_updated_ = 0;

    // ---- buffers ----
    std::queue<nav_msgs::msg::Odometry::ConstSharedPtr>      odom_buf_;
    std::queue<sensor_msgs::msg::PointCloud2::ConstSharedPtr> cloud_buf_;
    std::mutex buf_mutex_;

    // ---- ICP job queue ----
    std::queue<std::pair<int,int>> icp_job_buf_;
    std::mutex icp_mutex_;

    // ---- state ----
    Pose6D odom_prev_{0,0,0,0,0,0};
    Pose6D odom_curr_{0,0,0,0,0,0};
    double trans_accum_ = 1e9;
    double rot_accum_   = 1e9;
    std::atomic<bool> running_{true};
    std::atomic<bool> map_redraw_{false};

    // ---- loop visualization ----
    std::vector<std::pair<int,int>> accepted_loops_;

    // ---- publishers / subscribers ----
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr            pub_path_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr  pub_map_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr        pub_odom_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr  pub_loop_src_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr  pub_loop_tgt_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_loop_vis_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr         sub_odom_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr   sub_cloud_;

    // ---- threads ----
    std::thread thread_pg_;
    std::thread thread_lcd_;
    std::thread thread_icp_;

    // ---- voxel filters ----
    pcl::VoxelGrid<PointType> vf_keyframe_;  // used only in process_pg
    pcl::VoxelGrid<PointType> vf_icp_;       // used only in process_icp

    // ============================================================
    void loadParams()
    {
        declare_parameter("cloud_topic",             "rko_lio/frame");
        declare_parameter("odom_topic",              "rko_lio/odometry");
        declare_parameter("map_frame",               "map");
        declare_parameter("keyframe_meter_gap",      1.0);
        declare_parameter("keyframe_rad_gap",        0.4);
        declare_parameter("loop_closure_frequency",  1.0);
        declare_parameter("solid_loop_threshold",    0.85);
        declare_parameter("solid_num_exclude_recent", 50);
        declare_parameter("loop_submap_size",        10);
        declare_parameter("icp_fitness_threshold",   0.3);
        declare_parameter("map_voxel_size",          0.2);
        declare_parameter("keyframe_voxel_size",     0.4);
        declare_parameter("solid_fov_upper",         2.0);
        declare_parameter("solid_fov_lower",        -24.8);
        declare_parameter("solid_num_angle",         60);
        declare_parameter("solid_num_range",         40);
        declare_parameter("solid_num_height",        32);
        declare_parameter("solid_min_distance",      3);
        declare_parameter("solid_max_distance",      80);
        declare_parameter("solid_voxel_size",        0.4);
        declare_parameter("save_map",                false);
        declare_parameter("save_dir",                "/tmp/solid_pgo");

        // ---- odom noise mode ----
        declare_parameter("odom_noise_mode",          "fixed");   // "fixed"|"distance"|"covariance"
        declare_parameter("odom_sigma_rot",           1e-3);
        declare_parameter("odom_sigma_trans",         1e-2);
        declare_parameter("odom_dist_scale_rot",      1e-4);
        declare_parameter("odom_dist_scale_trans",    5e-3);
        declare_parameter("odom_min_sigma_rot",       1e-4);
        declare_parameter("odom_min_sigma_trans",     1e-3);

        cloud_topic_             = get_parameter("cloud_topic").as_string();
        odom_topic_              = get_parameter("odom_topic").as_string();
        map_frame_               = get_parameter("map_frame").as_string();
        keyframe_meter_gap_      = get_parameter("keyframe_meter_gap").as_double();
        keyframe_rad_gap_        = get_parameter("keyframe_rad_gap").as_double();
        loop_closure_frequency_  = get_parameter("loop_closure_frequency").as_double();
        solid_loop_threshold_    = get_parameter("solid_loop_threshold").as_double();
        solid_num_exclude_recent_= get_parameter("solid_num_exclude_recent").as_int();
        loop_submap_size_        = get_parameter("loop_submap_size").as_int();
        icp_fitness_threshold_   = get_parameter("icp_fitness_threshold").as_double();
        map_voxel_size_          = get_parameter("map_voxel_size").as_double();
        keyframe_voxel_size_     = get_parameter("keyframe_voxel_size").as_double();
        save_map_                = get_parameter("save_map").as_bool();
        save_dir_                = get_parameter("save_dir").as_string();

        solid_.FOV_u        = (float)get_parameter("solid_fov_upper").as_double();
        solid_.FOV_d        = (float)get_parameter("solid_fov_lower").as_double();
        solid_.NUM_ANGLE    = get_parameter("solid_num_angle").as_int();
        solid_.NUM_RANGE    = get_parameter("solid_num_range").as_int();
        solid_.NUM_HEIGHT   = get_parameter("solid_num_height").as_int();
        solid_.MIN_DISTANCE = get_parameter("solid_min_distance").as_int();
        solid_.MAX_DISTANCE = get_parameter("solid_max_distance").as_int();
        solid_.VOXEL_SIZE   = (float)get_parameter("solid_voxel_size").as_double();

        vf_keyframe_.setLeafSize(keyframe_voxel_size_, keyframe_voxel_size_, keyframe_voxel_size_);
        vf_icp_.setLeafSize(0.3f, 0.3f, 0.3f);

        // Build OdomNoiseProvider
        solid_pgo::OdomNoiseProvider::Config nc;
        nc.mode              = get_parameter("odom_noise_mode").as_string();
        nc.sigma_rot         = get_parameter("odom_sigma_rot").as_double();
        nc.sigma_trans       = get_parameter("odom_sigma_trans").as_double();
        nc.dist_scale_rot    = get_parameter("odom_dist_scale_rot").as_double();
        nc.dist_scale_trans  = get_parameter("odom_dist_scale_trans").as_double();
        nc.min_sigma_rot     = get_parameter("odom_min_sigma_rot").as_double();
        nc.min_sigma_trans   = get_parameter("odom_min_sigma_trans").as_double();
        odom_noise_provider_ = solid_pgo::OdomNoiseProvider(nc);

        RCLCPP_INFO(get_logger(), "[solid_pgo] odom_noise_mode: %s", nc.mode.c_str());
    }

    void initGTSAM()
    {
        ISAM2Params params;
        params.relinearizeThreshold = 0.01;
        params.relinearizeSkip      = 1;
        isam_ = new ISAM2(params);

        Vector6 pv; pv << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12;
        prior_noise_ = noiseModel::Diagonal::Variances(pv);

        Vector6 lv; lv << 0.5, 0.5, 0.5, 0.5, 0.5, 0.5;
        loop_noise_ = noiseModel::Robust::Create(
            noiseModel::mEstimator::Cauchy::Create(1),
            noiseModel::Diagonal::Variances(lv));
    }

    // ============================================================
    //  Thread 1: keyframe management + sequential odom factors
    // ============================================================
    void process_pg()
    {
        while (running_)
        {
            while (true)
            {
                nav_msgs::msg::Odometry::ConstSharedPtr      odom_msg;
                sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_msg;

                {
                    std::lock_guard<std::mutex> lk(buf_mutex_);
                    if (odom_buf_.empty() || cloud_buf_.empty()) break;

                    // time-align: drop stale odometry
                    double t_cloud = rclcpp::Time(cloud_buf_.front()->header.stamp).seconds();
                    while (!odom_buf_.empty() &&
                           rclcpp::Time(odom_buf_.front()->header.stamp).seconds() < t_cloud)
                        odom_buf_.pop();
                    if (odom_buf_.empty()) break;

                    odom_msg  = odom_buf_.front();  odom_buf_.pop();
                    cloud_msg = cloud_buf_.front();  cloud_buf_.pop();
                }

                // Convert
                Pose6D pose_curr = odomTopose6d(odom_msg);
                double time_curr = rclcpp::Time(odom_msg->header.stamp).seconds();

                pcl::PointCloud<PointType>::Ptr raw(new pcl::PointCloud<PointType>());
                pcl::fromROSMsg(*cloud_msg, *raw);

                // Keyframe decision
                odom_prev_ = odom_curr_;
                odom_curr_ = pose_curr;
                Pose6D dt = diffTransformation(odom_prev_, odom_curr_);
                double step_dist = sqrt(dt.x*dt.x + dt.y*dt.y + dt.z*dt.z);
                trans_accum_ += step_dist;
                rot_accum_   += (dt.roll + dt.pitch + dt.yaw);

                bool is_kf = (trans_accum_ > keyframe_meter_gap_ ||
                               rot_accum_  > keyframe_rad_gap_);
                if (!is_kf) continue;

                double kf_distance = trans_accum_;  // distance since last keyframe
                trans_accum_ = 0.0;
                rot_accum_   = 0.0;

                // Downsample for descriptor + storage
                pcl::PointCloud<PointType>::Ptr ds(new pcl::PointCloud<PointType>());
                vf_keyframe_.setInputCloud(raw);
                vf_keyframe_.filter(*ds);

                // Preprocess for SOLID (distance filter + voxel already done)
                pcl::PointCloud<PointType>::Ptr solid_cloud(new pcl::PointCloud<PointType>());
                solid_.remove_closest_points(*ds, solid_cloud);
                solid_.remove_far_points(*solid_cloud, solid_cloud);

                Eigen::VectorXd descriptor = solid_.makeSolid(*solid_cloud);

                {
                    std::lock_guard<std::mutex> lk(kf_mutex_);
                    kf_poses_.push_back(pose_curr);
                    kf_poses_updated_.push_back(pose_curr);
                    kf_times_.push_back(time_curr);
                    kf_distances_.push_back(kf_distance);
                    kf_odoms_.push_back(*odom_msg);
                    kf_clouds_.push_back(ds);
                    solid_keys_.push_back(descriptor);
                    map_redraw_ = true;
                }

                const int N = (int)kf_poses_.size();
                const int prev_idx = N - 2;
                const int curr_idx = N - 1;

                {
                    std::lock_guard<std::mutex> lk(pg_mutex_);
                    if (!graph_ready_)
                    {
                        // prior on first node
                        gtsam::Pose3 origin = pose6dToGtsam(pose_curr);
                        graph_.add(PriorFactor<gtsam::Pose3>(0, origin, prior_noise_));
                        initial_estimate_.insert(0, origin);
                        runISAM();
                        graph_ready_ = true;
                        RCLCPP_INFO(get_logger(), "[solid_pgo] Graph initialized at node 0");
                    }
                    else
                    {
                        gtsam::Pose3 pFrom = pose6dToGtsam(kf_poses_[prev_idx]);
                        gtsam::Pose3 pTo   = pose6dToGtsam(kf_poses_[curr_idx]);

                        // Per-keyframe noise: mode is selected in OdomNoiseProvider
                        auto odom_noise = odom_noise_provider_.noise_for_odom(
                            kf_odoms_[curr_idx], kf_distances_[curr_idx]);

                        graph_.add(BetweenFactor<gtsam::Pose3>(
                            prev_idx, curr_idx, pFrom.between(pTo), odom_noise));
                        initial_estimate_.insert(curr_idx, pTo);
                        runISAM();
                    }
                }

                if (curr_idx % 50 == 0)
                    RCLCPP_INFO(get_logger(), "[solid_pgo] Keyframe %d added", curr_idx);

                pubPath();

                // Publish map every 10 keyframes
                if (curr_idx % 10 == 0)
                    pubMap();
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    // ============================================================
    //  Thread 2: loop detection (SOLID)
    // ============================================================
    void process_lcd()
    {
        rclcpp::Rate rate(loop_closure_frequency_);
        while (running_ && rclcpp::ok())
        {
            rate.sleep();
            performSOLiDLoopClosure();
        }
    }

    void performSOLiDLoopClosure()
    {
        // Snapshot descriptors under a single lock to avoid repeated lock/unlock
        std::vector<Eigen::VectorXd> keys_snapshot;
        int n_kf;
        {
            std::lock_guard<std::mutex> lk(kf_mutex_);
            n_kf = (int)solid_keys_.size();
            if (n_kf < solid_num_exclude_recent_ + 1) return;
            keys_snapshot = solid_keys_;  // cheap copy (VectorXd is value type)
        }

        // Query = latest keyframe
        int query_idx = n_kf - 1;
        const Eigen::VectorXd& query_desc = keys_snapshot[query_idx];

        // Brute-force search over all non-recent keyframes
        int    best_idx   = -1;
        double best_score = solid_loop_threshold_;  // cosine similarity: higher = more similar

        int search_end = n_kf - solid_num_exclude_recent_;
        for (int i = 0; i < search_end; ++i)
        {
            double score = solid_.loop_detection(query_desc, keys_snapshot[i]);
            if (score > best_score)
            {
                best_score = score;
                best_idx   = i;
            }
        }

        if (best_idx < 0) return;

        RCLCPP_INFO(get_logger(),
                    "[solid_pgo] Loop candidate: query=%d  match=%d  similarity=%.3f",
                    query_idx, best_idx, best_score);

        {
            std::lock_guard<std::mutex> lk(icp_mutex_);
            icp_job_buf_.push({best_idx, query_idx});
        }
    }

    // ============================================================
    //  Thread 3: ICP verification + add loop factor
    // ============================================================
    void process_icp()
    {
        while (running_)
        {
            while (true)
            {
                std::pair<int,int> job;
                {
                    std::lock_guard<std::mutex> lk(icp_mutex_);
                    if (icp_job_buf_.empty()) break;
                    job = icp_job_buf_.front();
                    icp_job_buf_.pop();
                }

                auto result = doICP(job.first, job.second);
                if (!result.has_value()) continue;

                gtsam::Pose3 relative = result.value();
                int loop_idx  = job.first;
                int curr_idx  = job.second;

                {
                    std::lock_guard<std::mutex> lk(pg_mutex_);
                    graph_.add(BetweenFactor<gtsam::Pose3>(
                        loop_idx, curr_idx, relative, loop_noise_));
                    runISAM();
                }

                {
                    std::lock_guard<std::mutex> lk(kf_mutex_);
                    accepted_loops_.emplace_back(loop_idx, curr_idx);
                }

                RCLCPP_INFO(get_logger(),
                            "[solid_pgo] Loop factor added: %d <-> %d", loop_idx, curr_idx);

                pubLoopMarkers();
                pubPath();
                pubMap();
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    // ============================================================
    //  ICP: build submap around loop_idx, align curr cloud
    // ============================================================
    std::optional<gtsam::Pose3> doICP(int loop_idx, int curr_idx)
    {
        pcl::PointCloud<PointType>::Ptr src(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr tgt(new pcl::PointCloud<PointType>());

        {
            std::lock_guard<std::mutex> lk(kf_mutex_);

            // source: current keyframe transformed to global frame by its own pose
            *src += *local2global(kf_clouds_[curr_idx], kf_poses_updated_[curr_idx]);

            // target: submap around loop keyframe
            for (int k = -loop_submap_size_; k <= loop_submap_size_; ++k)
            {
                int idx = loop_idx + k;
                if (idx < 0 || idx >= (int)kf_clouds_.size()) continue;
                *tgt += *local2global(kf_clouds_[idx], kf_poses_updated_[idx]);
            }
        }

        if (src->empty() || tgt->empty()) return std::nullopt;

        // Downsample for ICP
        pcl::PointCloud<PointType>::Ptr src_ds(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr tgt_ds(new pcl::PointCloud<PointType>());
        vf_icp_.setInputCloud(src); vf_icp_.filter(*src_ds);
        vf_icp_.setInputCloud(tgt); vf_icp_.filter(*tgt_ds);

        // Publish for debug
        pubLoopClouds(src_ds, tgt_ds);

        // Build ICP initial guess from SOLID yaw estimate.
        // Both clouds are in global frame; apply yaw correction around the
        // source centroid so ICP starts closer to the true alignment.
        Eigen::Matrix4f icp_init = Eigen::Matrix4f::Identity();
        {
            std::lock_guard<std::mutex> lk(kf_mutex_);
            double yaw_deg = solid_.pose_estimation(solid_keys_[curr_idx], solid_keys_[loop_idx]);
            // Centroid of source
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*src_ds, centroid);
            float cx = centroid[0], cy = centroid[1], cz = centroid[2];
            float yaw_rad = deg2rad((float)yaw_deg);
            float c = cosf(yaw_rad), s = sinf(yaw_rad);
            // R * (p - center) + center  →  T_init = translate(center) * R * translate(-center)
            icp_init(0,0) =  c; icp_init(0,1) = -s;
            icp_init(1,0) =  s; icp_init(1,1) =  c;
            icp_init(0,3) = cx - c*cx + s*cy;
            icp_init(1,3) = cy - s*cx - c*cy;
            icp_init(2,3) = 0.0f;
        }

        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(150.0);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);
        icp.setInputSource(src_ds);
        icp.setInputTarget(tgt_ds);

        pcl::PointCloud<PointType>::Ptr aligned(new pcl::PointCloud<PointType>());
        icp.align(*aligned, icp_init);

        if (!icp.hasConverged() || icp.getFitnessScore() > icp_fitness_threshold_)
        {
            RCLCPP_WARN(get_logger(),
                        "[solid_pgo] ICP rejected: converged=%d  score=%.4f  threshold=%.4f",
                        icp.hasConverged(), icp.getFitnessScore(), icp_fitness_threshold_);
            return std::nullopt;
        }

        RCLCPP_INFO(get_logger(), "[solid_pgo] ICP accepted: score=%.4f", icp.getFitnessScore());

        // Compute relative pose in global frame:
        // corrected_curr = T_icp * curr_global
        // relative = T_loop^{-1} * corrected_curr
        float ix, iy, iz, ir, ip, iyw;
        Eigen::Affine3f T_icp(icp.getFinalTransformation());
        pcl::getTranslationAndEulerAngles(T_icp, ix, iy, iz, ir, ip, iyw);
        gtsam::Pose3 poseICP = gtsam::Pose3(gtsam::Rot3::RzRyRx(ir, ip, iyw),
                                             gtsam::Point3(ix, iy, iz));

        Pose6D lp, cp;
        {
            std::lock_guard<std::mutex> lk(kf_mutex_);
            lp = kf_poses_updated_[loop_idx];
            cp = kf_poses_updated_[curr_idx];
        }
        gtsam::Pose3 pLoop = pose6dToGtsam(lp);
        gtsam::Pose3 pCurr = pose6dToGtsam(cp);
        gtsam::Pose3 pCorrected = poseICP * pCurr;
        return pLoop.between(pCorrected);
    }

    // ============================================================
    //  iSAM2 update
    // ============================================================
    void runISAM()
    {
        // Called with pg_mutex_ held.
        // Safe to acquire kf_mutex_ here because no thread holds kf_mutex_
        // while waiting for pg_mutex_ (process_pg and process_icp always release
        // kf_mutex_ before acquiring pg_mutex_).
        isam_->update(graph_, initial_estimate_);
        isam_->update();

        graph_.resize(0);
        initial_estimate_.clear();

        isam_estimate_ = isam_->calculateEstimate();

        std::lock_guard<std::mutex> lk(kf_mutex_);
        for (int i = 0; i < (int)isam_estimate_.size() && i < (int)kf_poses_updated_.size(); ++i)
        {
            gtsam::Pose3 p = isam_estimate_.at<gtsam::Pose3>(i);
            kf_poses_updated_[i] = gtsamToPose6d(p);
        }
        recent_idx_updated_ = (int)kf_poses_updated_.size() - 1;
    }

    // ============================================================
    //  Publishers
    // ============================================================
    void pubPath()
    {
        nav_msgs::msg::Path path_msg;
        nav_msgs::msg::Odometry odom_msg;
        path_msg.header.frame_id = map_frame_;

        {
            std::lock_guard<std::mutex> lk(kf_mutex_);
            for (int i = 0; i <= recent_idx_updated_; ++i)
            {
                const Pose6D& p = kf_poses_updated_[i];
                geometry_msgs::msg::PoseStamped ps;
                ps.header.frame_id = map_frame_;
                ps.header.stamp    = rclcpp::Time((uint64_t)(kf_times_[i] * 1e9));
                ps.pose.position.x = p.x;
                ps.pose.position.y = p.y;
                ps.pose.position.z = p.z;
                tf2::Quaternion q;
                q.setRPY(p.roll, p.pitch, p.yaw);
                ps.pose.orientation.x = q.x();
                ps.pose.orientation.y = q.y();
                ps.pose.orientation.z = q.z();
                ps.pose.orientation.w = q.w();
                path_msg.poses.push_back(ps);

                if (i == recent_idx_updated_)
                {
                    odom_msg.header = ps.header;
                    odom_msg.child_frame_id = "base_link";
                    odom_msg.pose.pose = ps.pose;
                }
            }
        }

        path_msg.header.stamp = now();
        pub_path_->publish(path_msg);
        pub_odom_->publish(odom_msg);

        // TF: map -> base_link
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header = odom_msg.header;
        tf_msg.child_frame_id = "base_link";
        tf_msg.transform.translation.x = odom_msg.pose.pose.position.x;
        tf_msg.transform.translation.y = odom_msg.pose.pose.position.y;
        tf_msg.transform.translation.z = odom_msg.pose.pose.position.z;
        tf_msg.transform.rotation      = odom_msg.pose.pose.orientation;
        tf_broadcaster_->sendTransform(tf_msg);
    }

    void pubMap()
    {
        if (!map_redraw_) return;
        map_redraw_ = false;

        pcl::PointCloud<PointType>::Ptr map(new pcl::PointCloud<PointType>());
        {
            std::lock_guard<std::mutex> lk(kf_mutex_);
            for (int i = 0; i < (int)kf_clouds_.size(); ++i)
                *map += *local2global(kf_clouds_[i], kf_poses_updated_[i]);
        }

        pcl::PointCloud<PointType>::Ptr map_ds(new pcl::PointCloud<PointType>());
        pcl::VoxelGrid<PointType> vf;
        vf.setLeafSize(map_voxel_size_, map_voxel_size_, map_voxel_size_);
        vf.setInputCloud(map);
        vf.filter(*map_ds);

        sensor_msgs::msg::PointCloud2 msg;
        pcl::toROSMsg(*map_ds, msg);
        msg.header.frame_id = map_frame_;
        msg.header.stamp    = now();
        pub_map_->publish(msg);

        if (save_map_)
        {
            std::filesystem::create_directories(save_dir_);
            std::string path = save_dir_ + "/map.pcd";
            pcl::io::savePCDFileBinary(path, *map_ds);
            RCLCPP_INFO_ONCE(get_logger(), "[solid_pgo] Map will be saved to %s", path.c_str());
        }

        map_redraw_ = false;
    }

    void pubLoopClouds(const pcl::PointCloud<PointType>::Ptr& src,
                        const pcl::PointCloud<PointType>::Ptr& tgt)
    {
        auto toMsg = [&](const pcl::PointCloud<PointType>::Ptr& cloud) {
            sensor_msgs::msg::PointCloud2 msg;
            pcl::toROSMsg(*cloud, msg);
            msg.header.frame_id = map_frame_;
            msg.header.stamp    = now();
            return msg;
        };
        pub_loop_src_->publish(toMsg(src));
        pub_loop_tgt_->publish(toMsg(tgt));
    }

    void pubLoopMarkers()
    {
        visualization_msgs::msg::MarkerArray arr;
        visualization_msgs::msg::Marker line_list;
        line_list.header.frame_id = map_frame_;
        line_list.header.stamp    = now();
        line_list.ns              = "loop_closures";
        line_list.id              = 0;
        line_list.type            = visualization_msgs::msg::Marker::LINE_LIST;
        line_list.action          = visualization_msgs::msg::Marker::ADD;
        line_list.scale.x         = 0.1;
        line_list.color.r         = 0.0f;
        line_list.color.g         = 1.0f;
        line_list.color.b         = 0.0f;
        line_list.color.a         = 1.0f;

        std::lock_guard<std::mutex> lk(kf_mutex_);
        for (auto& [a, b] : accepted_loops_)
        {
            if (a >= (int)kf_poses_updated_.size() ||
                b >= (int)kf_poses_updated_.size()) continue;

            geometry_msgs::msg::Point pa, pb;
            pa.x = kf_poses_updated_[a].x;
            pa.y = kf_poses_updated_[a].y;
            pa.z = kf_poses_updated_[a].z;
            pb.x = kf_poses_updated_[b].x;
            pb.y = kf_poses_updated_[b].y;
            pb.z = kf_poses_updated_[b].z;
            line_list.points.push_back(pa);
            line_list.points.push_back(pb);
        }

        arr.markers.push_back(line_list);
        pub_loop_vis_->publish(arr);
    }
};

// ============================================================
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SolidPgoNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
