// =============================================================================
// solid_pgo_node.cpp  (ROS2)
// =============================================================================
// Ported from ros1/src/solid_pgo_node.cpp.
// All ROS1 API replaced with rclcpp equivalents.
// core/ headers (SOLiDManager, PoseGraphManager) are unchanged.
// ICP and GTSAM logic are identical to ROS1 version.
// =============================================================================

#include <fstream>
#include <math.h>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include <optional>
#include <iomanip>
#include <sstream>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_ros/transform_broadcaster.h>

#include <eigen3/Eigen/Dense>

// core/ headers (ROS-free)
#include "solid_pgo/common.h"
#include "solid_pgo/SOLiDManager.h"
#include "solid_pgo/PoseGraphManager.h"

using namespace gtsam;
using std::cout;
using std::endl;

// =============================================================================
// SolidPgoNode: rclcpp::Node subclass holding all state
// =============================================================================

class SolidPgoNode : public rclcpp::Node
{
public:
    SolidPgoNode() : rclcpp::Node("solid_pgo_node")
    {
        // ----- Parameters -----
        this->declare_parameter<std::string>("save_directory",          "/tmp/solid_pgo_data/");
        this->declare_parameter<double>     ("keyframe_meter_gap",      2.0);
        this->declare_parameter<double>     ("keyframe_deg_gap",        10.0);
        this->declare_parameter<double>     ("solid_loop_threshold",    0.7);
        this->declare_parameter<int>        ("solid_num_exclude_recent",50);
        this->declare_parameter<double>     ("mapviz_filter_size",      0.4);

        save_directory_       = this->get_parameter("save_directory").as_string();
        keyframeMeterGap_     = this->get_parameter("keyframe_meter_gap").as_double();
        double keyframeDegGap = this->get_parameter("keyframe_deg_gap").as_double();
        keyframeRadGap_       = deg2rad(keyframeDegGap);

        double solidLoopThreshold    = this->get_parameter("solid_loop_threshold").as_double();
        int    solidNumExcludeRecent = this->get_parameter("solid_num_exclude_recent").as_int();
        solidManager_.setLoopThreshold(solidLoopThreshold);
        solidManager_.setNumExcludeRecent(solidNumExcludeRecent);

        double mapVizFilterSize = this->get_parameter("mapviz_filter_size").as_double();

        // ----- File setup -----
        pgKITTIformat_    = save_directory_ + "optimized_poses.txt";
        odomKITTIformat_  = save_directory_ + "odom_poses.txt";
        pgTimeSaveStream_ = std::fstream(save_directory_ + "times.txt", std::fstream::out);
        pgTimeSaveStream_.precision(std::numeric_limits<double>::max_digits10);
        pgScansDirectory_ = save_directory_ + "Scans/";
        auto unused = system((std::string("exec rm -r ") + pgScansDirectory_).c_str());
        unused      = system((std::string("mkdir -p ")   + pgScansDirectory_).c_str());

        // ----- PGO init -----
        pgManager_.init();

        // ----- Voxel filters -----
        float filter_size = 0.4f;
        downSizeFilterScancontext_.setLeafSize(filter_size, filter_size, filter_size);
        downSizeFilterICP_.setLeafSize(filter_size, filter_size, filter_size);
        downSizeFilterMapPGO_.setLeafSize(
            static_cast<float>(mapVizFilterSize),
            static_cast<float>(mapVizFilterSize),
            static_cast<float>(mapVizFilterSize));

        // ----- Subscribers -----
        auto qos_sensor = rclcpp::SensorDataQoS();
        auto qos_default = rclcpp::QoS(100);

        subLaserCloudFullRes_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/velodyne_cloud_registered_local", qos_sensor,
            [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                mBuf_.lock();
                fullResBuf_.push(msg);
                mBuf_.unlock();
            });

        subLaserOdometry_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/aft_mapped_to_init", qos_sensor,
            [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
                mBuf_.lock();
                odometryBuf_.push(msg);
                mBuf_.unlock();
            });

        subGPS_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
            "/gps/fix", qos_default,
            [this](const sensor_msgs::msg::NavSatFix::SharedPtr msg) {
                if (useGPS_) {
                    mBuf_.lock();
                    gpsBuf_.push(msg);
                    mBuf_.unlock();
                }
            });

        // ----- Publishers -----
        pubOdomAftPGO_        = this->create_publisher<nav_msgs::msg::Odometry>("/aft_pgo_odom", 100);
        pubOdomRepubVerifier_ = this->create_publisher<nav_msgs::msg::Odometry>("/repub_odom", 100);
        pubPathAftPGO_        = this->create_publisher<nav_msgs::msg::Path>("/aft_pgo_path", 100);
        pubMapAftPGO_         = this->create_publisher<sensor_msgs::msg::PointCloud2>("/aft_pgo_map", 100);
        pubLoopScanLocal_     = this->create_publisher<sensor_msgs::msg::PointCloud2>("/loop_scan_local", 100);
        pubLoopSubmapLocal_   = this->create_publisher<sensor_msgs::msg::PointCloud2>("/loop_submap_local", 100);

        tfBroadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        // ----- Threads -----
        threadPG_      = std::thread(&SolidPgoNode::process_pg,        this);
        threadLCD_     = std::thread(&SolidPgoNode::process_lcd,       this);
        threadICP_     = std::thread(&SolidPgoNode::process_icp,       this);
        threadISAM_    = std::thread(&SolidPgoNode::process_isam,      this);
        threadVizMap_  = std::thread(&SolidPgoNode::process_viz_map,   this);
        threadVizPath_ = std::thread(&SolidPgoNode::process_viz_path,  this);

        RCLCPP_INFO(this->get_logger(), "solid_pgo_node started.");
    }

    ~SolidPgoNode()
    {
        threadPG_.join();
        threadLCD_.join();
        threadICP_.join();
        threadISAM_.join();
        threadVizMap_.join();
        threadVizPath_.join();
    }

private:
    // =========================================================================
    // State
    // =========================================================================

    double keyframeMeterGap_  = 2.0;
    double keyframeRadGap_    = 0.1745;  // ~10 deg
    double translationAccumulated_ = 1e6;
    double rotaionAccumulated_     = 1e6;
    bool   isNowKeyFrame_     = false;

    Pose6D odom_pose_prev_ {0,0,0,0,0,0};
    Pose6D odom_pose_curr_ {0,0,0,0,0,0};

    std::queue<nav_msgs::msg::Odometry::SharedPtr>    odometryBuf_;
    std::queue<sensor_msgs::msg::PointCloud2::SharedPtr> fullResBuf_;
    std::queue<sensor_msgs::msg::NavSatFix::SharedPtr>   gpsBuf_;
    std::queue<std::pair<int,int>>                    scLoopICPBuf_;

    std::mutex mBuf_;
    std::mutex mKF_;

    double timeLaserOdometry_ = 0.0;
    double timeLaser_         = 0.0;

    std::vector<pcl::PointCloud<PointType>::Ptr> keyframeLaserClouds_;
    std::vector<Pose6D> keyframePoses_;
    std::vector<Pose6D> keyframePosesUpdated_;
    std::vector<double> keyframeTimes_;
    int recentIdxUpdated_ = 0;

    SOLiDManager     solidManager_;
    PoseGraphManager pgManager_;

    pcl::VoxelGrid<PointType> downSizeFilterScancontext_;
    pcl::VoxelGrid<PointType> downSizeFilterICP_;
    std::mutex mtxICP_;
    std::mutex mtxRecentPose_;

    pcl::PointCloud<PointType>::Ptr laserCloudMapPGO_ {new pcl::PointCloud<PointType>()};
    pcl::VoxelGrid<PointType>       downSizeFilterMapPGO_;
    bool laserCloudMapPGORedraw_ = true;

    bool   useGPS_                = true;
    sensor_msgs::msg::NavSatFix::SharedPtr currGPS_;
    bool   hasGPSforThisKF_       = false;
    bool   gpsOffsetInitialized_  = false;
    double gpsAltitudeInitOffset_ = 0.0;
    double recentOptimizedX_      = 0.0;
    double recentOptimizedY_      = 0.0;

    std::string save_directory_;
    std::string pgKITTIformat_, pgScansDirectory_, odomKITTIformat_;
    std::fstream pgTimeSaveStream_;

    // Subscribers / Publishers
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subLaserCloudFullRes_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr       subLaserOdometry_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr   subGPS_;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr     pubOdomAftPGO_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr     pubOdomRepubVerifier_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr         pubPathAftPGO_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubMapAftPGO_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLoopScanLocal_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLoopSubmapLocal_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster_;

    // Worker threads
    std::thread threadPG_, threadLCD_, threadICP_, threadISAM_;
    std::thread threadVizMap_, threadVizPath_;

    // =========================================================================
    // Utilities
    // =========================================================================

    static std::string padZeros(int val, int num_digits = 6)
    {
        std::ostringstream out;
        out << std::internal << std::setfill('0') << std::setw(num_digits) << val;
        return out.str();
    }

    void saveOdometryVerticesKITTIformat(const std::string& _filename)
    {
        std::fstream stream(_filename.c_str(), std::fstream::out);
        for (const auto& _pose6d : keyframePoses_) {
            gtsam::Pose3 pose = PoseGraphManager::pose6DtoGTSAM(_pose6d);
            gtsam::Point3 t = pose.translation();
            gtsam::Rot3   R = pose.rotation();
            auto col1 = R.column(1); auto col2 = R.column(2); auto col3 = R.column(3);
            stream << col1.x() <<" "<< col2.x() <<" "<< col3.x() <<" "<< t.x() <<" "
                   << col1.y() <<" "<< col2.y() <<" "<< col3.y() <<" "<< t.y() <<" "
                   << col1.z() <<" "<< col2.z() <<" "<< col3.z() <<" "<< t.z() << endl;
        }
    }

    void saveOptimizedVerticesKITTIformat(const gtsam::Values& _estimates,
                                          const std::string& _filename)
    {
        std::fstream stream(_filename.c_str(), std::fstream::out);
        for (const auto& key_value : _estimates) {
            auto p = dynamic_cast<const gtsam::GenericValue<gtsam::Pose3>*>(&key_value.value);
            if (!p) continue;
            const gtsam::Pose3& pose = p->value();
            gtsam::Point3 t = pose.translation();
            gtsam::Rot3   R = pose.rotation();
            auto col1 = R.column(1); auto col2 = R.column(2); auto col3 = R.column(3);
            stream << col1.x() <<" "<< col2.x() <<" "<< col3.x() <<" "<< t.x() <<" "
                   << col1.y() <<" "<< col2.y() <<" "<< col3.y() <<" "<< t.y() <<" "
                   << col1.z() <<" "<< col2.z() <<" "<< col3.z() <<" "<< t.z() << endl;
        }
    }

    // =========================================================================
    // Pose helpers
    // =========================================================================

    static Pose6D getOdom(const nav_msgs::msg::Odometry::SharedPtr& _odom)
    {
        double tx = _odom->pose.pose.position.x;
        double ty = _odom->pose.pose.position.y;
        double tz = _odom->pose.pose.position.z;

        double roll, pitch, yaw;
        tf2::Quaternion q(
            _odom->pose.pose.orientation.x, _odom->pose.pose.orientation.y,
            _odom->pose.pose.orientation.z, _odom->pose.pose.orientation.w);
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

        return Pose6D{tx, ty, tz, roll, pitch, yaw};
    }

    static Pose6D diffTransformation(const Pose6D& _p1, const Pose6D& _p2)
    {
        Eigen::Affine3f SE3_p1 = pcl::getTransformation(_p1.x,_p1.y,_p1.z,_p1.roll,_p1.pitch,_p1.yaw);
        Eigen::Affine3f SE3_p2 = pcl::getTransformation(_p2.x,_p2.y,_p2.z,_p2.roll,_p2.pitch,_p2.yaw);
        Eigen::Matrix4f SE3_delta0 = SE3_p1.matrix().inverse() * SE3_p2.matrix();
        Eigen::Affine3f SE3_delta; SE3_delta.matrix() = SE3_delta0;
        float dx, dy, dz, droll, dpitch, dyaw;
        pcl::getTranslationAndEulerAngles(SE3_delta, dx, dy, dz, droll, dpitch, dyaw);
        return Pose6D{double(abs(dx)),double(abs(dy)),double(abs(dz)),
                      double(abs(droll)),double(abs(dpitch)),double(abs(dyaw))};
    }

    pcl::PointCloud<PointType>::Ptr local2global(const pcl::PointCloud<PointType>::Ptr& cloudIn,
                                                  const Pose6D& tf)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);
        Eigen::Affine3f transCur = pcl::getTransformation(tf.x,tf.y,tf.z,tf.roll,tf.pitch,tf.yaw);
        int numberOfCores = 16;
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i) {
            const auto& pt = cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0)*pt.x+transCur(0,1)*pt.y+transCur(0,2)*pt.z+transCur(0,3);
            cloudOut->points[i].y = transCur(1,0)*pt.x+transCur(1,1)*pt.y+transCur(1,2)*pt.z+transCur(1,3);
            cloudOut->points[i].z = transCur(2,0)*pt.x+transCur(2,1)*pt.y+transCur(2,2)*pt.z+transCur(2,3);
        }
        return cloudOut;
    }

    // =========================================================================
    // ISAM2 + pose sync
    // =========================================================================

    void updatePoses()
    {
        const gtsam::Values& estimate = pgManager_.getEstimate();
        mKF_.lock();
        for (int i = 0; i < int(estimate.size()); i++)
            keyframePosesUpdated_[i] = pgManager_.getOptimizedPose(i);
        mKF_.unlock();

        mtxRecentPose_.lock();
        const gtsam::Pose3& lastPose = estimate.at<gtsam::Pose3>(int(estimate.size()) - 1);
        recentOptimizedX_ = lastPose.translation().x();
        recentOptimizedY_ = lastPose.translation().y();
        recentIdxUpdated_ = int(keyframePosesUpdated_.size()) - 1;
        mtxRecentPose_.unlock();
    }

    void runISAM2opt()
    {
        pgManager_.runISAM2opt();
        updatePoses();
    }

    // =========================================================================
    // ICP
    // =========================================================================

    void loopFindNearKeyframesCloud(pcl::PointCloud<PointType>::Ptr& nearKeyframes,
                                     const int& key, const int& submap_size,
                                     const int& root_idx)
    {
        nearKeyframes->clear();
        for (int i = -submap_size; i <= submap_size; ++i) {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= int(keyframeLaserClouds_.size())) continue;
            mKF_.lock();
            *nearKeyframes += *local2global(keyframeLaserClouds_[keyNear],
                                            keyframePosesUpdated_[root_idx]);
            mKF_.unlock();
        }
        if (nearKeyframes->empty()) return;
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP_.setInputCloud(nearKeyframes);
        downSizeFilterICP_.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    std::optional<gtsam::Pose3> doICPVirtualRelative(int _loop_kf_idx, int _curr_kf_idx)
    {
        int historyKeyframeSearchNum = 25;
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr targetKeyframeCloud(new pcl::PointCloud<PointType>());
        loopFindNearKeyframesCloud(cureKeyframeCloud,   _curr_kf_idx, 0,                        _loop_kf_idx);
        loopFindNearKeyframesCloud(targetKeyframeCloud, _loop_kf_idx, historyKeyframeSearchNum, _loop_kf_idx);

        // Publish for visualization
        sensor_msgs::msg::PointCloud2 cureMsg, targetMsg;
        pcl::toROSMsg(*cureKeyframeCloud, cureMsg);
        cureMsg.header.frame_id = "camera_init";
        cureMsg.header.stamp    = this->now();
        pubLoopScanLocal_->publish(cureMsg);

        pcl::toROSMsg(*targetKeyframeCloud, targetMsg);
        targetMsg.header.frame_id = "camera_init";
        targetMsg.header.stamp    = this->now();
        pubLoopSubmapLocal_->publish(targetMsg);

        // ICP (identical settings to SC-PGO)
        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(150);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(targetKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);

        float loopFitnessScoreThreshold = 0.3f;
        if (icp.hasConverged() == false || icp.getFitnessScore() > loopFitnessScoreThreshold) {
            RCLCPP_INFO(this->get_logger(),
                "[SOLiD loop] ICP fitness test FAILED (%.4f > %.4f). Reject.",
                icp.getFitnessScore(), loopFitnessScoreThreshold);
            return std::nullopt;
        }
        RCLCPP_INFO(this->get_logger(),
            "[SOLiD loop] ICP fitness test PASSED (%.4f < %.4f). Add loop.",
            icp.getFitnessScore(), loopFitnessScoreThreshold);

        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame = icp.getFinalTransformation();
        pcl::getTranslationAndEulerAngles(correctionLidarFrame, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll,pitch,yaw),
                                              gtsam::Point3(x,y,z));
        gtsam::Pose3 poseTo   = gtsam::Pose3(gtsam::Rot3::RzRyRx(0,0,0),
                                              gtsam::Point3(0,0,0));
        return poseFrom.between(poseTo);
    }

    // =========================================================================
    // Visualization
    // =========================================================================

    void pubPath()
    {
        nav_msgs::msg::Odometry odomAftPGO;
        nav_msgs::msg::Path     pathAftPGO;
        pathAftPGO.header.frame_id = "camera_init";

        mKF_.lock();
        for (int node_idx = 0; node_idx < recentIdxUpdated_; node_idx++) {
            const Pose6D& pose_est = keyframePosesUpdated_.at(node_idx);

            nav_msgs::msg::Odometry odomAftPGOthis;
            odomAftPGOthis.header.frame_id  = "camera_init";
            odomAftPGOthis.child_frame_id   = "aft_pgo";

            rclcpp::Time stamp(
                static_cast<int64_t>(keyframeTimes_.at(node_idx) * 1e9));
            odomAftPGOthis.header.stamp = stamp;

            odomAftPGOthis.pose.pose.position.x = pose_est.x;
            odomAftPGOthis.pose.pose.position.y = pose_est.y;
            odomAftPGOthis.pose.pose.position.z = pose_est.z;

            tf2::Quaternion q;
            q.setRPY(pose_est.roll, pose_est.pitch, pose_est.yaw);
            odomAftPGOthis.pose.pose.orientation.x = q.x();
            odomAftPGOthis.pose.pose.orientation.y = q.y();
            odomAftPGOthis.pose.pose.orientation.z = q.z();
            odomAftPGOthis.pose.pose.orientation.w = q.w();

            odomAftPGO = odomAftPGOthis;

            geometry_msgs::msg::PoseStamped poseStampAftPGO;
            poseStampAftPGO.header = odomAftPGOthis.header;
            poseStampAftPGO.pose   = odomAftPGOthis.pose.pose;

            pathAftPGO.header.stamp    = odomAftPGOthis.header.stamp;
            pathAftPGO.header.frame_id = "camera_init";
            pathAftPGO.poses.push_back(poseStampAftPGO);
        }
        mKF_.unlock();

        pubOdomAftPGO_->publish(odomAftPGO);
        pubPathAftPGO_->publish(pathAftPGO);

        // TF broadcast
        geometry_msgs::msg::TransformStamped transformStamped;
        transformStamped.header.stamp    = odomAftPGO.header.stamp;
        transformStamped.header.frame_id = "camera_init";
        transformStamped.child_frame_id  = "aft_pgo";
        transformStamped.transform.translation.x = odomAftPGO.pose.pose.position.x;
        transformStamped.transform.translation.y = odomAftPGO.pose.pose.position.y;
        transformStamped.transform.translation.z = odomAftPGO.pose.pose.position.z;
        transformStamped.transform.rotation      = odomAftPGO.pose.pose.orientation;
        tfBroadcaster_->sendTransform(transformStamped);
    }

    void pubMap()
    {
        int SKIP_FRAMES = 2, counter = 0;
        laserCloudMapPGO_->clear();

        mKF_.lock();
        for (int node_idx = 0; node_idx < recentIdxUpdated_; node_idx++) {
            if (counter % SKIP_FRAMES == 0)
                *laserCloudMapPGO_ += *local2global(keyframeLaserClouds_[node_idx],
                                                    keyframePosesUpdated_[node_idx]);
            counter++;
        }
        mKF_.unlock();

        downSizeFilterMapPGO_.setInputCloud(laserCloudMapPGO_);
        downSizeFilterMapPGO_.filter(*laserCloudMapPGO_);

        sensor_msgs::msg::PointCloud2 laserCloudMapPGOMsg;
        pcl::toROSMsg(*laserCloudMapPGO_, laserCloudMapPGOMsg);
        laserCloudMapPGOMsg.header.frame_id = "camera_init";
        laserCloudMapPGOMsg.header.stamp    = this->now();
        pubMapAftPGO_->publish(laserCloudMapPGOMsg);
    }

    // =========================================================================
    // Threads  (identical logic to ROS1, only ROS API calls differ)
    // =========================================================================

    void process_pg()
    {
        while (rclcpp::ok()) {
            while (!odometryBuf_.empty() && !fullResBuf_.empty()) {
                mBuf_.lock();
                while (!odometryBuf_.empty() &&
                       rclcpp::Time(odometryBuf_.front()->header.stamp).seconds() <
                       rclcpp::Time(fullResBuf_.front()->header.stamp).seconds())
                    odometryBuf_.pop();
                if (odometryBuf_.empty()) { mBuf_.unlock(); break; }

                timeLaserOdometry_ = rclcpp::Time(odometryBuf_.front()->header.stamp).seconds();
                timeLaser_         = rclcpp::Time(fullResBuf_.front()->header.stamp).seconds();

                pcl::PointCloud<PointType>::Ptr thisKeyFrame(new pcl::PointCloud<PointType>());
                pcl::fromROSMsg(*fullResBuf_.front(), *thisKeyFrame);
                fullResBuf_.pop();

                Pose6D pose_curr = getOdom(odometryBuf_.front());
                odometryBuf_.pop();

                double eps = 0.1;
                while (!gpsBuf_.empty()) {
                    auto thisGPS     = gpsBuf_.front();
                    double thisGPSTime = rclcpp::Time(thisGPS->header.stamp).seconds();
                    if (abs(thisGPSTime - timeLaserOdometry_) < eps) {
                        currGPS_ = thisGPS;
                        hasGPSforThisKF_ = true;
                        break;
                    } else {
                        hasGPSforThisKF_ = false;
                    }
                    gpsBuf_.pop();
                }
                mBuf_.unlock();

                odom_pose_prev_ = odom_pose_curr_;
                odom_pose_curr_ = pose_curr;
                Pose6D dtf = diffTransformation(odom_pose_prev_, odom_pose_curr_);

                double delta_translation = sqrt(dtf.x*dtf.x + dtf.y*dtf.y + dtf.z*dtf.z);
                translationAccumulated_ += delta_translation;
                rotaionAccumulated_     += (dtf.roll + dtf.pitch + dtf.yaw);

                if (translationAccumulated_ > keyframeMeterGap_ ||
                    rotaionAccumulated_     > keyframeRadGap_) {
                    isNowKeyFrame_          = true;
                    translationAccumulated_ = 0.0;
                    rotaionAccumulated_     = 0.0;
                } else {
                    isNowKeyFrame_ = false;
                }

                if (!isNowKeyFrame_) continue;

                if (!gpsOffsetInitialized_ && hasGPSforThisKF_) {
                    gpsAltitudeInitOffset_ = currGPS_->altitude;
                    gpsOffsetInitialized_  = true;
                }

                pcl::PointCloud<PointType>::Ptr thisKeyFrameDS(new pcl::PointCloud<PointType>());
                downSizeFilterScancontext_.setInputCloud(thisKeyFrame);
                downSizeFilterScancontext_.filter(*thisKeyFrameDS);

                mKF_.lock();
                keyframeLaserClouds_.push_back(thisKeyFrameDS);
                keyframePoses_.push_back(pose_curr);
                keyframePosesUpdated_.push_back(pose_curr);
                keyframeTimes_.push_back(timeLaserOdometry_);
                solidManager_.makeAndSaveDescriptor(*thisKeyFrameDS);
                laserCloudMapPGORedraw_ = true;
                mKF_.unlock();

                const int prev_node_idx = keyframePoses_.size() - 2;
                const int curr_node_idx = keyframePoses_.size() - 1;

                if (!pgManager_.isGraphMade()) {
                    pgManager_.mtxPosegraph.lock();
                    pgManager_.addPriorFactor(curr_node_idx, keyframePoses_.at(curr_node_idx));
                    pgManager_.mtxPosegraph.unlock();
                    RCLCPP_INFO(this->get_logger(), "posegraph prior node %d added", curr_node_idx);
                } else {
                    pgManager_.mtxPosegraph.lock();
                    pgManager_.addOdomFactor(prev_node_idx, curr_node_idx,
                                             keyframePoses_.at(prev_node_idx),
                                             keyframePoses_.at(curr_node_idx));
                    if (hasGPSforThisKF_) {
                        double curr_altitude_offseted = currGPS_->altitude - gpsAltitudeInitOffset_;
                        mtxRecentPose_.lock();
                        pgManager_.addGPSFactor(curr_node_idx,
                                                recentOptimizedX_, recentOptimizedY_,
                                                curr_altitude_offseted);
                        mtxRecentPose_.unlock();
                        RCLCPP_INFO(this->get_logger(), "GPS factor added at node %d", curr_node_idx);
                    }
                    pgManager_.mtxPosegraph.unlock();

                    if (curr_node_idx % 100 == 0)
                        RCLCPP_INFO(this->get_logger(),
                            "posegraph odom node %d added.", curr_node_idx);
                }

                std::string curr_node_idx_str = padZeros(curr_node_idx);
                pcl::io::savePCDFileBinary(pgScansDirectory_ + curr_node_idx_str + ".pcd", *thisKeyFrame);
                pgTimeSaveStream_ << timeLaser_ << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    void process_lcd()
    {
        rclcpp::Rate rate(1.0);
        while (rclcpp::ok()) {
            rate.sleep();
            performSOLiDLoopClosure();
        }
    }

    void performSOLiDLoopClosure()
    {
        if (int(keyframePoses_.size()) < solidManager_.NUM_EXCLUDE_RECENT)
            return;

        auto detectResult = solidManager_.detectLoopClosureID();
        int closestHistoryFrameID = detectResult.first;

        if (closestHistoryFrameID != -1) {
            const int prev_node_idx = closestHistoryFrameID;
            const int curr_node_idx = keyframePoses_.size() - 1;
            RCLCPP_INFO(this->get_logger(),
                "Loop detected! between %d and %d", prev_node_idx, curr_node_idx);
            mBuf_.lock();
            scLoopICPBuf_.push({prev_node_idx, curr_node_idx});
            mBuf_.unlock();
        }
    }

    void process_icp()
    {
        while (rclcpp::ok()) {
            while (!scLoopICPBuf_.empty()) {
                if (scLoopICPBuf_.size() > 30)
                    RCLCPP_WARN(this->get_logger(),
                        "Too many loop closure candidates waiting for ICP.");

                mBuf_.lock();
                auto loop_idx_pair = scLoopICPBuf_.front();
                scLoopICPBuf_.pop();
                mBuf_.unlock();

                auto relative_pose_optional = doICPVirtualRelative(
                    loop_idx_pair.first, loop_idx_pair.second);
                if (relative_pose_optional) {
                    pgManager_.mtxPosegraph.lock();
                    pgManager_.addLoopFactor(loop_idx_pair.first, loop_idx_pair.second,
                                             relative_pose_optional.value());
                    pgManager_.mtxPosegraph.unlock();
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    void process_isam()
    {
        rclcpp::Rate rate(1.0);
        while (rclcpp::ok()) {
            rate.sleep();
            if (pgManager_.isGraphMade()) {
                pgManager_.mtxPosegraph.lock();
                runISAM2opt();
                pgManager_.mtxPosegraph.unlock();
                RCLCPP_INFO(this->get_logger(), "running isam2 optimization ...");

                saveOptimizedVerticesKITTIformat(pgManager_.getEstimate(), pgKITTIformat_);
                saveOdometryVerticesKITTIformat(odomKITTIformat_);
            }
        }
    }

    void process_viz_path()
    {
        rclcpp::Rate rate(10.0);
        while (rclcpp::ok()) {
            rate.sleep();
            if (recentIdxUpdated_ > 1) pubPath();
        }
    }

    void process_viz_map()
    {
        rclcpp::Rate rate(0.1);
        while (rclcpp::ok()) {
            rate.sleep();
            if (recentIdxUpdated_ > 1) pubMap();
        }
    }
};

// =============================================================================
// main
// =============================================================================

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SolidPgoNode>());
    rclcpp::shutdown();
    return 0;
}
