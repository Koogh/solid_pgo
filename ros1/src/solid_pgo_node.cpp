// =============================================================================
// solid_pgo_node.cpp  (ROS1)
// =============================================================================
// Based on SC-PGO's laserPosegraphOptimization.cpp (gisbi-kim/FAST_LIO_SLAM)
// Changes:
//   - SCManager → SOLiDManager (core/include/solid_pgo/SOLiDManager.h)
//   - GTSAM globals → PoseGraphManager (core/include/solid_pgo/PoseGraphManager.h)
//   - PointType = pcl::PointXYZ (no intensity)
//   - ScanContext params replaced by SOLiD params
// Everything else (ICP, topics, thread structure, mutex) is identical to SC-PGO.
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

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>

#include <eigen3/Eigen/Dense>

// core/ headers (ROS-free)
#include "solid_pgo/common.h"
#include "solid_pgo/SOLiDManager.h"
#include "solid_pgo/PoseGraphManager.h"

using namespace gtsam;
using std::cout;
using std::endl;

// =============================================================================
// Global state
// =============================================================================

double keyframeMeterGap;
double keyframeDegGap, keyframeRadGap;
double translationAccumulated = 1000000.0;
double rotaionAccumulated     = 1000000.0;

bool isNowKeyFrame = false;

Pose6D odom_pose_prev {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
Pose6D odom_pose_curr {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

std::queue<nav_msgs::Odometry::ConstPtr>    odometryBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<sensor_msgs::NavSatFix::ConstPtr> gpsBuf;
std::queue<std::pair<int, int>>             scLoopICPBuf;

std::mutex mBuf;
std::mutex mKF;

double timeLaserOdometry = 0.0;
double timeLaser         = 0.0;

pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudMapAfterPGO(new pcl::PointCloud<PointType>());

std::vector<pcl::PointCloud<PointType>::Ptr> keyframeLaserClouds;
std::vector<Pose6D> keyframePoses;
std::vector<Pose6D> keyframePosesUpdated;
std::vector<double> keyframeTimes;
int recentIdxUpdated = 0;

// Core managers (ROS-free)
SOLiDManager    solidManager;
PoseGraphManager pgManager;

pcl::VoxelGrid<PointType> downSizeFilterScancontext;
pcl::VoxelGrid<PointType> downSizeFilterICP;
std::mutex mtxICP;
std::mutex mtxRecentPose;

pcl::PointCloud<PointType>::Ptr laserCloudMapPGO(new pcl::PointCloud<PointType>());
pcl::VoxelGrid<PointType> downSizeFilterMapPGO;
bool laserCloudMapPGORedraw = true;

bool useGPS = true;
sensor_msgs::NavSatFix::ConstPtr currGPS;
bool hasGPSforThisKF       = false;
bool gpsOffsetInitialized  = false;
double gpsAltitudeInitOffset = 0.0;
double recentOptimizedX    = 0.0;
double recentOptimizedY    = 0.0;

ros::Publisher pubMapAftPGO, pubOdomAftPGO, pubPathAftPGO;
ros::Publisher pubLoopScanLocal, pubLoopSubmapLocal;
ros::Publisher pubOdomRepubVerifier;

std::string save_directory;
std::string pgKITTIformat, pgScansDirectory;
std::string odomKITTIformat;
std::fstream pgTimeSaveStream;

// =============================================================================
// Utility
// =============================================================================

std::string padZeros(int val, int num_digits = 6)
{
    std::ostringstream out;
    out << std::internal << std::setfill('0') << std::setw(num_digits) << val;
    return out.str();
}

void saveOdometryVerticesKITTIformat(std::string _filename)
{
    std::fstream stream(_filename.c_str(), std::fstream::out);
    for (const auto& _pose6d : keyframePoses) {
        gtsam::Pose3 pose = PoseGraphManager::pose6DtoGTSAM(_pose6d);
        gtsam::Point3 t = pose.translation();
        gtsam::Rot3   R = pose.rotation();
        auto col1 = R.column(1);
        auto col2 = R.column(2);
        auto col3 = R.column(3);
        stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x() << " "
               << col1.y() << " " << col2.y() << " " << col3.y() << " " << t.y() << " "
               << col1.z() << " " << col2.z() << " " << col3.z() << " " << t.z() << endl;
    }
}

void saveOptimizedVerticesKITTIformat(const gtsam::Values& _estimates, std::string _filename)
{
    std::fstream stream(_filename.c_str(), std::fstream::out);
    for (const auto& key_value : _estimates) {
        auto p = dynamic_cast<const gtsam::GenericValue<gtsam::Pose3>*>(&key_value.value);
        if (!p) continue;
        const gtsam::Pose3& pose = p->value();
        gtsam::Point3 t = pose.translation();
        gtsam::Rot3   R = pose.rotation();
        auto col1 = R.column(1);
        auto col2 = R.column(2);
        auto col3 = R.column(3);
        stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x() << " "
               << col1.y() << " " << col2.y() << " " << col3.y() << " " << t.y() << " "
               << col1.z() << " " << col2.z() << " " << col3.z() << " " << t.z() << endl;
    }
}

// =============================================================================
// ROS callbacks
// =============================================================================

void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr& _laserOdometry)
{
    mBuf.lock();
    odometryBuf.push(_laserOdometry);
    mBuf.unlock();
}

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr& _laserCloudFullRes)
{
    mBuf.lock();
    fullResBuf.push(_laserCloudFullRes);
    mBuf.unlock();
}

void gpsHandler(const sensor_msgs::NavSatFix::ConstPtr& _gps)
{
    if (useGPS) {
        mBuf.lock();
        gpsBuf.push(_gps);
        mBuf.unlock();
    }
}

// =============================================================================
// Pose helpers
// =============================================================================

Pose6D getOdom(nav_msgs::Odometry::ConstPtr _odom)
{
    auto tx = _odom->pose.pose.position.x;
    auto ty = _odom->pose.pose.position.y;
    auto tz = _odom->pose.pose.position.z;

    double roll, pitch, yaw;
    geometry_msgs::Quaternion quat = _odom->pose.pose.orientation;
    tf::Matrix3x3(tf::Quaternion(quat.x, quat.y, quat.z, quat.w)).getRPY(roll, pitch, yaw);

    return Pose6D{tx, ty, tz, roll, pitch, yaw};
}

Pose6D diffTransformation(const Pose6D& _p1, const Pose6D& _p2)
{
    Eigen::Affine3f SE3_p1 = pcl::getTransformation(_p1.x, _p1.y, _p1.z, _p1.roll, _p1.pitch, _p1.yaw);
    Eigen::Affine3f SE3_p2 = pcl::getTransformation(_p2.x, _p2.y, _p2.z, _p2.roll, _p2.pitch, _p2.yaw);
    Eigen::Matrix4f SE3_delta0 = SE3_p1.matrix().inverse() * SE3_p2.matrix();
    Eigen::Affine3f SE3_delta; SE3_delta.matrix() = SE3_delta0;
    float dx, dy, dz, droll, dpitch, dyaw;
    pcl::getTranslationAndEulerAngles(SE3_delta, dx, dy, dz, droll, dpitch, dyaw);
    return Pose6D{double(abs(dx)), double(abs(dy)), double(abs(dz)),
                  double(abs(droll)), double(abs(dpitch)), double(abs(dyaw))};
}

// Transform cloud to the coordinate frame defined by tf (PointXYZ — no intensity)
pcl::PointCloud<PointType>::Ptr local2global(const pcl::PointCloud<PointType>::Ptr& cloudIn,
                                              const Pose6D& tf)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(tf.x, tf.y, tf.z,
                                                       tf.roll, tf.pitch, tf.yaw);
    int numberOfCores = 16;
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i) {
        const auto& pt = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0)*pt.x + transCur(0,1)*pt.y + transCur(0,2)*pt.z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0)*pt.x + transCur(1,1)*pt.y + transCur(1,2)*pt.z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0)*pt.x + transCur(2,1)*pt.y + transCur(2,2)*pt.z + transCur(2,3);
    }
    return cloudOut;
}

pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn,
                                                     gtsam::Pose3 transformIn)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(
        transformIn.translation().x(), transformIn.translation().y(), transformIn.translation().z(),
        transformIn.rotation().roll(), transformIn.rotation().pitch(), transformIn.rotation().yaw());

    int numberOfCores = 8;
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i) {
        const auto& pt = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0)*pt.x + transCur(0,1)*pt.y + transCur(0,2)*pt.z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0)*pt.x + transCur(1,1)*pt.y + transCur(1,2)*pt.z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0)*pt.x + transCur(2,1)*pt.y + transCur(2,2)*pt.z + transCur(2,3);
    }
    return cloudOut;
}

// =============================================================================
// ISAM2 update + pose sync
// =============================================================================

void updatePoses(void)
{
    const gtsam::Values& estimate = pgManager.getEstimate();
    mKF.lock();
    for (int i = 0; i < int(estimate.size()); i++) {
        Pose6D& p = keyframePosesUpdated[i];
        p = pgManager.getOptimizedPose(i);
    }
    mKF.unlock();

    mtxRecentPose.lock();
    const gtsam::Pose3& lastPose = estimate.at<gtsam::Pose3>(int(estimate.size()) - 1);
    recentOptimizedX = lastPose.translation().x();
    recentOptimizedY = lastPose.translation().y();
    recentIdxUpdated = int(keyframePosesUpdated.size()) - 1;
    mtxRecentPose.unlock();
}

void runISAM2opt(void)
{
    pgManager.runISAM2opt();
    updatePoses();
}

// =============================================================================
// ICP helper
// =============================================================================

void loopFindNearKeyframesCloud(pcl::PointCloud<PointType>::Ptr& nearKeyframes,
                                 const int& key, const int& submap_size,
                                 const int& root_idx)
{
    nearKeyframes->clear();
    for (int i = -submap_size; i <= submap_size; ++i) {
        int keyNear = key + i;
        if (keyNear < 0 || keyNear >= int(keyframeLaserClouds.size()))
            continue;
        mKF.lock();
        *nearKeyframes += *local2global(keyframeLaserClouds[keyNear], keyframePosesUpdated[root_idx]);
        mKF.unlock();
    }
    if (nearKeyframes->empty()) return;

    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    downSizeFilterICP.setInputCloud(nearKeyframes);
    downSizeFilterICP.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
}

std::optional<gtsam::Pose3> doICPVirtualRelative(int _loop_kf_idx, int _curr_kf_idx)
{
    int historyKeyframeSearchNum = 25;
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr targetKeyframeCloud(new pcl::PointCloud<PointType>());
    loopFindNearKeyframesCloud(cureKeyframeCloud,   _curr_kf_idx, 0,                        _loop_kf_idx);
    loopFindNearKeyframesCloud(targetKeyframeCloud, _loop_kf_idx, historyKeyframeSearchNum, _loop_kf_idx);

    // Publish for visualization / debugging
    sensor_msgs::PointCloud2 cureKeyframeCloudMsg;
    pcl::toROSMsg(*cureKeyframeCloud, cureKeyframeCloudMsg);
    cureKeyframeCloudMsg.header.frame_id = "camera_init";
    pubLoopScanLocal.publish(cureKeyframeCloudMsg);

    sensor_msgs::PointCloud2 targetKeyframeCloudMsg;
    pcl::toROSMsg(*targetKeyframeCloud, targetKeyframeCloudMsg);
    targetKeyframeCloudMsg.header.frame_id = "camera_init";
    pubLoopSubmapLocal.publish(targetKeyframeCloudMsg);

    // ICP settings (identical to SC-PGO original)
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

    float loopFitnessScoreThreshold = 0.3;
    if (icp.hasConverged() == false || icp.getFitnessScore() > loopFitnessScoreThreshold) {
        std::cout << "[SOLiD loop] ICP fitness test failed (" << icp.getFitnessScore()
                  << " > " << loopFitnessScoreThreshold << "). Reject." << std::endl;
        return std::nullopt;
    } else {
        std::cout << "[SOLiD loop] ICP fitness test passed (" << icp.getFitnessScore()
                  << " < " << loopFitnessScoreThreshold << "). Add loop." << std::endl;
    }

    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame = icp.getFinalTransformation();
    pcl::getTranslationAndEulerAngles(correctionLidarFrame, x, y, z, roll, pitch, yaw);
    gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw),
                                          gtsam::Point3(x, y, z));
    gtsam::Pose3 poseTo   = gtsam::Pose3(gtsam::Rot3::RzRyRx(0.0, 0.0, 0.0),
                                          gtsam::Point3(0.0, 0.0, 0.0));
    return poseFrom.between(poseTo);
}

// =============================================================================
// Visualization helpers
// =============================================================================

void pubPath(void)
{
    nav_msgs::Odometry odomAftPGO;
    nav_msgs::Path     pathAftPGO;
    pathAftPGO.header.frame_id = "camera_init";

    mKF.lock();
    for (int node_idx = 0; node_idx < recentIdxUpdated; node_idx++) {
        const Pose6D& pose_est = keyframePosesUpdated.at(node_idx);

        nav_msgs::Odometry odomAftPGOthis;
        odomAftPGOthis.header.frame_id  = "camera_init";
        odomAftPGOthis.child_frame_id   = "/aft_pgo";
        odomAftPGOthis.header.stamp     = ros::Time().fromSec(keyframeTimes.at(node_idx));
        odomAftPGOthis.pose.pose.position.x    = pose_est.x;
        odomAftPGOthis.pose.pose.position.y    = pose_est.y;
        odomAftPGOthis.pose.pose.position.z    = pose_est.z;
        odomAftPGOthis.pose.pose.orientation   =
            tf::createQuaternionMsgFromRollPitchYaw(pose_est.roll, pose_est.pitch, pose_est.yaw);
        odomAftPGO = odomAftPGOthis;

        geometry_msgs::PoseStamped poseStampAftPGO;
        poseStampAftPGO.header = odomAftPGOthis.header;
        poseStampAftPGO.pose   = odomAftPGOthis.pose.pose;

        pathAftPGO.header.stamp    = odomAftPGOthis.header.stamp;
        pathAftPGO.header.frame_id = "camera_init";
        pathAftPGO.poses.push_back(poseStampAftPGO);
    }
    mKF.unlock();

    pubOdomAftPGO.publish(odomAftPGO);
    pubPathAftPGO.publish(pathAftPGO);

    static tf::TransformBroadcaster br;
    tf::Transform  transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odomAftPGO.pose.pose.position.x,
                                    odomAftPGO.pose.pose.position.y,
                                    odomAftPGO.pose.pose.position.z));
    q.setW(odomAftPGO.pose.pose.orientation.w);
    q.setX(odomAftPGO.pose.pose.orientation.x);
    q.setY(odomAftPGO.pose.pose.orientation.y);
    q.setZ(odomAftPGO.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odomAftPGO.header.stamp,
                                          "camera_init", "/aft_pgo"));
}

void pubMap(void)
{
    int SKIP_FRAMES = 2;
    int counter = 0;
    laserCloudMapPGO->clear();

    mKF.lock();
    for (int node_idx = 0; node_idx < recentIdxUpdated; node_idx++) {
        if (counter % SKIP_FRAMES == 0)
            *laserCloudMapPGO += *local2global(keyframeLaserClouds[node_idx],
                                               keyframePosesUpdated[node_idx]);
        counter++;
    }
    mKF.unlock();

    downSizeFilterMapPGO.setInputCloud(laserCloudMapPGO);
    downSizeFilterMapPGO.filter(*laserCloudMapPGO);

    sensor_msgs::PointCloud2 laserCloudMapPGOMsg;
    pcl::toROSMsg(*laserCloudMapPGO, laserCloudMapPGOMsg);
    laserCloudMapPGOMsg.header.frame_id = "camera_init";
    pubMapAftPGO.publish(laserCloudMapPGOMsg);
}

// =============================================================================
// Threads
// =============================================================================

void process_pg()
{
    while (1)
    {
        while (!odometryBuf.empty() && !fullResBuf.empty())
        {
            mBuf.lock();
            while (!odometryBuf.empty() &&
                   odometryBuf.front()->header.stamp.toSec() < fullResBuf.front()->header.stamp.toSec())
                odometryBuf.pop();
            if (odometryBuf.empty()) { mBuf.unlock(); break; }

            timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();
            timeLaser         = fullResBuf.front()->header.stamp.toSec();

            laserCloudFullRes->clear();
            pcl::PointCloud<PointType>::Ptr thisKeyFrame(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(*fullResBuf.front(), *thisKeyFrame);
            fullResBuf.pop();

            Pose6D pose_curr = getOdom(odometryBuf.front());
            odometryBuf.pop();

            // Match nearest GPS
            double eps = 0.1;
            while (!gpsBuf.empty()) {
                auto thisGPS     = gpsBuf.front();
                auto thisGPSTime = thisGPS->header.stamp.toSec();
                if (abs(thisGPSTime - timeLaserOdometry) < eps) {
                    currGPS = thisGPS;
                    hasGPSforThisKF = true;
                    break;
                } else {
                    hasGPSforThisKF = false;
                }
                gpsBuf.pop();
            }
            mBuf.unlock();

            // Keyframe selection (distance + rotation threshold)
            odom_pose_prev = odom_pose_curr;
            odom_pose_curr = pose_curr;
            Pose6D dtf = diffTransformation(odom_pose_prev, odom_pose_curr);

            double delta_translation = sqrt(dtf.x*dtf.x + dtf.y*dtf.y + dtf.z*dtf.z);
            translationAccumulated += delta_translation;
            rotaionAccumulated     += (dtf.roll + dtf.pitch + dtf.yaw);

            if (translationAccumulated > keyframeMeterGap || rotaionAccumulated > keyframeRadGap) {
                isNowKeyFrame          = true;
                translationAccumulated = 0.0;
                rotaionAccumulated     = 0.0;
            } else {
                isNowKeyFrame = false;
            }

            if (!isNowKeyFrame) continue;

            if (!gpsOffsetInitialized && hasGPSforThisKF) {
                gpsAltitudeInitOffset = currGPS->altitude;
                gpsOffsetInitialized  = true;
            }

            // Downsample and save keyframe
            pcl::PointCloud<PointType>::Ptr thisKeyFrameDS(new pcl::PointCloud<PointType>());
            downSizeFilterScancontext.setInputCloud(thisKeyFrame);
            downSizeFilterScancontext.filter(*thisKeyFrameDS);

            mKF.lock();
            keyframeLaserClouds.push_back(thisKeyFrameDS);
            keyframePoses.push_back(pose_curr);
            keyframePosesUpdated.push_back(pose_curr);
            keyframeTimes.push_back(timeLaserOdometry);

            // Generate and store SOLiD descriptor
            solidManager.makeAndSaveDescriptor(*thisKeyFrameDS);

            laserCloudMapPGORedraw = true;
            mKF.unlock();

            const int prev_node_idx = keyframePoses.size() - 2;
            const int curr_node_idx = keyframePoses.size() - 1;

            if (!pgManager.isGraphMade()) {
                // Prior factor for first node
                pgManager.mtxPosegraph.lock();
                pgManager.addPriorFactor(curr_node_idx, keyframePoses.at(curr_node_idx));
                pgManager.mtxPosegraph.unlock();
                cout << "posegraph prior node " << curr_node_idx << " added" << endl;
            } else {
                // Odometry factor
                pgManager.mtxPosegraph.lock();
                pgManager.addOdomFactor(prev_node_idx, curr_node_idx,
                                        keyframePoses.at(prev_node_idx),
                                        keyframePoses.at(curr_node_idx));

                if (hasGPSforThisKF) {
                    double curr_altitude_offseted = currGPS->altitude - gpsAltitudeInitOffset;
                    mtxRecentPose.lock();
                    pgManager.addGPSFactor(curr_node_idx,
                                           recentOptimizedX, recentOptimizedY,
                                           curr_altitude_offseted);
                    mtxRecentPose.unlock();
                    cout << "GPS factor added at node " << curr_node_idx << endl;
                }
                pgManager.mtxPosegraph.unlock();

                if (curr_node_idx % 100 == 0)
                    cout << "posegraph odom node " << curr_node_idx << " added." << endl;
            }

            // Save to disk
            std::string curr_node_idx_str = padZeros(curr_node_idx);
            pcl::io::savePCDFileBinary(pgScansDirectory + curr_node_idx_str + ".pcd", *thisKeyFrame);
            pgTimeSaveStream << timeLaser << std::endl;
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

void performSOLiDLoopClosure(void)
{
    if (int(keyframePoses.size()) < solidManager.NUM_EXCLUDE_RECENT)
        return;

    auto detectResult = solidManager.detectLoopClosureID();
    int closestHistoryFrameID = detectResult.first;

    if (closestHistoryFrameID != -1) {
        const int prev_node_idx = closestHistoryFrameID;
        const int curr_node_idx = keyframePoses.size() - 1;
        cout << "Loop detected! - between " << prev_node_idx
             << " and " << curr_node_idx << endl;

        mBuf.lock();
        scLoopICPBuf.push(std::pair<int, int>(prev_node_idx, curr_node_idx));
        mBuf.unlock();
    }
}

void process_lcd(void)
{
    float loopClosureFrequency = 1.0;
    ros::Rate rate(loopClosureFrequency);
    while (ros::ok()) {
        rate.sleep();
        performSOLiDLoopClosure();
    }
}

void process_icp(void)
{
    while (1) {
        while (!scLoopICPBuf.empty()) {
            if (scLoopICPBuf.size() > 30)
                ROS_WARN("Too many loop closure candidates waiting for ICP. "
                         "Reduce loopClosureFrequency.");

            mBuf.lock();
            std::pair<int, int> loop_idx_pair = scLoopICPBuf.front();
            scLoopICPBuf.pop();
            mBuf.unlock();

            const int prev_node_idx = loop_idx_pair.first;
            const int curr_node_idx = loop_idx_pair.second;
            auto relative_pose_optional = doICPVirtualRelative(prev_node_idx, curr_node_idx);
            if (relative_pose_optional) {
                gtsam::Pose3 relative_pose = relative_pose_optional.value();
                pgManager.mtxPosegraph.lock();
                pgManager.addLoopFactor(prev_node_idx, curr_node_idx, relative_pose);
                pgManager.mtxPosegraph.unlock();
            }
        }
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

void process_viz_path(void)
{
    float hz = 10.0;
    ros::Rate rate(hz);
    while (ros::ok()) {
        rate.sleep();
        if (recentIdxUpdated > 1)
            pubPath();
    }
}

void process_isam(void)
{
    float hz = 1.0;
    ros::Rate rate(hz);
    while (ros::ok()) {
        rate.sleep();
        if (pgManager.isGraphMade()) {
            pgManager.mtxPosegraph.lock();
            runISAM2opt();
            cout << "running isam2 optimization ..." << endl;
            pgManager.mtxPosegraph.unlock();

            saveOptimizedVerticesKITTIformat(pgManager.getEstimate(), pgKITTIformat);
            saveOdometryVerticesKITTIformat(odomKITTIformat);
        }
    }
}

void process_viz_map(void)
{
    float vizmapFrequency = 0.1;
    ros::Rate rate(vizmapFrequency);
    while (ros::ok()) {
        rate.sleep();
        if (recentIdxUpdated > 1)
            pubMap();
    }
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char** argv)
{
    ros::init(argc, argv, "solid_pgo");
    ros::NodeHandle nh;

    // Save directory
    nh.param<std::string>("save_directory", save_directory, "/");
    pgKITTIformat    = save_directory + "optimized_poses.txt";
    odomKITTIformat  = save_directory + "odom_poses.txt";
    pgTimeSaveStream = std::fstream(save_directory + "times.txt", std::fstream::out);
    pgTimeSaveStream.precision(std::numeric_limits<double>::max_digits10);
    pgScansDirectory = save_directory + "Scans/";
    auto unused = system((std::string("exec rm -r ") + pgScansDirectory).c_str());
    unused      = system((std::string("mkdir -p ")   + pgScansDirectory).c_str());

    // Keyframe params
    nh.param<double>("keyframe_meter_gap", keyframeMeterGap, 2.0);
    nh.param<double>("keyframe_deg_gap",   keyframeDegGap,   10.0);
    keyframeRadGap = deg2rad(keyframeDegGap);

    // SOLiD loop closure params
    double solidLoopThreshold;
    int    solidNumExcludeRecent;
    nh.param<double>("solid_loop_threshold",     solidLoopThreshold,     0.7);
    nh.param<int>   ("solid_num_exclude_recent", solidNumExcludeRecent,  50);
    solidManager.setLoopThreshold(solidLoopThreshold);
    solidManager.setNumExcludeRecent(solidNumExcludeRecent);

    // ISAM2 + noise init
    pgManager.init();

    // Voxel filters
    float filter_size = 0.4;
    downSizeFilterScancontext.setLeafSize(filter_size, filter_size, filter_size);
    downSizeFilterICP.setLeafSize(filter_size, filter_size, filter_size);

    double mapVizFilterSize;
    nh.param<double>("mapviz_filter_size", mapVizFilterSize, 0.4);
    downSizeFilterMapPGO.setLeafSize(mapVizFilterSize, mapVizFilterSize, mapVizFilterSize);

    // Subscribers
    ros::Subscriber subLaserCloudFullRes =
        nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_registered_local", 100,
                                               laserCloudFullResHandler);
    ros::Subscriber subLaserOdometry =
        nh.subscribe<nav_msgs::Odometry>("/aft_mapped_to_init", 100, laserOdometryHandler);
    ros::Subscriber subGPS =
        nh.subscribe<sensor_msgs::NavSatFix>("/gps/fix", 100, gpsHandler);

    // Publishers
    pubOdomAftPGO       = nh.advertise<nav_msgs::Odometry>("/aft_pgo_odom", 100);
    pubOdomRepubVerifier= nh.advertise<nav_msgs::Odometry>("/repub_odom", 100);
    pubPathAftPGO       = nh.advertise<nav_msgs::Path>("/aft_pgo_path", 100);
    pubMapAftPGO        = nh.advertise<sensor_msgs::PointCloud2>("/aft_pgo_map", 100);
    pubLoopScanLocal    = nh.advertise<sensor_msgs::PointCloud2>("/loop_scan_local", 100);
    pubLoopSubmapLocal  = nh.advertise<sensor_msgs::PointCloud2>("/loop_submap_local", 100);

    // Launch threads (same structure as SC-PGO)
    std::thread posegraph_slam  {process_pg};
    std::thread lc_detection    {process_lcd};
    std::thread icp_calculation {process_icp};
    std::thread isam_update     {process_isam};
    std::thread viz_map         {process_viz_map};
    std::thread viz_path        {process_viz_path};

    ros::spin();

    return 0;
}
