// This file is part of LINS.
//
// Copyright (C) 2020 Chao Qin <cscharlesqin@gmail.com>,
// Robotics and Multiperception Lab (RAM-LAB <https://ram-lab.com>),
// The Hong Kong University of Science and Technology
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.

#include <Estimator.h>

int g_frame_count = 0;

namespace fusion
{

  int Scan::scan_counter_ = 0;

  LinsFusion::LinsFusion(ros::NodeHandle &nh, ros::NodeHandle &pnh)
      : nh_(nh), pnh_(pnh) {}

  LinsFusion::~LinsFusion() { delete estimator; }

  void LinsFusion::run() { initialization(); }

  void LinsFusion::initialization()
  {
    // Implement an iterative-ESKF Kalman filter class
    estimator = new StateEstimator();

    // Subscribe to IMU, segmented point clouds, and map-refined odometry feedback
    subImu = pnh_.subscribe<sensor_msgs::Imu>(IMU_TOPIC, 100, &LinsFusion::imuCallback, this);
    subLaserCloud = pnh_.subscribe<sensor_msgs::PointCloud2>("/segmented_cloud", 2, &LinsFusion::laserCloudCallback, this);
    subLaserCloudInfo = pnh_.subscribe<cloud_msgs::cloud_info>("/segmented_cloud_info", 2, &LinsFusion::laserCloudInfoCallback, this);
    subOutlierCloud = pnh_.subscribe<sensor_msgs::PointCloud2>("/outlier_cloud", 2, &LinsFusion::outlierCloudCallback, this);

    // Set publishers
    pubUndistortedPointCloud = pnh_.advertise<sensor_msgs::PointCloud2>("/undistorted_point_cloud", 1);

    pubCornerPointsSharp = pnh_.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 1);
    pubCornerPointsLessSharp = pnh_.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 1);
    pubSurfPointsFlat = pnh_.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 1);
    pubSurfPointsLessFlat = pnh_.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 1);

    pubLaserCloudCornerLast = pnh_.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 2);
    pubLaserCloudSurfLast = pnh_.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2);
    pubOutlierCloudLast = pnh_.advertise<sensor_msgs::PointCloud2>("/outlier_cloud_last", 2);
    pubLaserOdometry = pnh_.advertise<nav_msgs::Odometry>(LIDAR_ODOMETRY_TOPIC, 5);
    pubLaserPath = pnh_.advertise<nav_msgs::Path>("laser_local_path", 1);

    // Set types of the point cloud
    distortedPointCloud.reset(new pcl::PointCloud<PointType>());
    outlierPointCloud.reset(new pcl::PointCloud<PointType>());

    // Allocate measurement buffers for sensors
    imuBuf_.allocate(500);
    pclBuf_.allocate(3);
    outlierBuf_.allocate(3);
    cloudInfoBuf_.allocate(3);

    // Initialize IMU propagation parameters
    isImuCalibrated = CALIBARTE_IMU;
    ba_init_ = INIT_BA;
    bw_init_ = INIT_BW;
    ba_tmp_.setZero();
    bw_tmp_.setZero();
    sample_counter_ = 0;

    duration_ = 0.0;
    scan_counter_ = 0;

    ROS_INFO_STREAM("Subscribe to \033[1;32m---->\033[0m " << IMU_TOPIC);
    ROS_INFO_STREAM("Subscribe to \033[1;32m---->\033[0m " << LIDAR_TOPIC);
  }

  void LinsFusion::laserCloudCallback(
      const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
  {
    // Add a new segmented point cloud
    pclBuf_.addMeas(laserCloudMsg, laserCloudMsg->header.stamp.toSec());
  }
  void LinsFusion::laserCloudInfoCallback(
      const cloud_msgs::cloud_infoConstPtr &cloudInfoMsg)
  {
    // Add segmentation information of the point cloud
    cloudInfoBuf_.addMeas(*cloudInfoMsg, cloudInfoMsg->header.stamp.toSec());
  }

  void LinsFusion::outlierCloudCallback(
      const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
  {
    outlierBuf_.addMeas(laserCloudMsg, laserCloudMsg->header.stamp.toSec());
  }

  void LinsFusion::imuCallback(const sensor_msgs::Imu::ConstPtr &imuMsg)
  {
    // Align IMU measurements from IMU frame to vehicle frame
    // two frames share same roll and pitch angles, but with a small
    // misalign-angle in the yaw direction
    acc_raw_ << imuMsg->linear_acceleration.x, imuMsg->linear_acceleration.y, imuMsg->linear_acceleration.z;
    gyr_raw_ << imuMsg->angular_velocity.x, imuMsg->angular_velocity.y, imuMsg->angular_velocity.z;
    misalign_euler_angles_ << deg2rad(0.0), deg2rad(0.0), deg2rad(IMU_MISALIGN_ANGLE);
    alignIMUtoVehicle(misalign_euler_angles_, acc_raw_, gyr_raw_, acc_aligned_, gyr_aligned_);

    // Add a new IMU measurement
    Imu imu(imuMsg->header.stamp.toSec(), acc_aligned_, gyr_aligned_);
    imuBuf_.addMeas(imu, imuMsg->header.stamp.toSec());

    // Trigger the Kalman filter
    performStateEstimation();
  }

  void LinsFusion::processFirstPointCloud()
  {
    // Use the most recent poing cloud to initialize the estimator
    pclBuf_.getLastTime(scan_time_);

    sensor_msgs::PointCloud2::ConstPtr pclMsg;
    pclBuf_.getLastMeas(pclMsg);
    distortedPointCloud->clear();
    pcl::fromROSMsg(*pclMsg, *distortedPointCloud);

    sensor_msgs::PointCloud2::ConstPtr outlierMsg;
    outlierBuf_.getLastMeas(outlierMsg);
    outlierPointCloud->clear();
    pcl::fromROSMsg(*outlierMsg, *outlierPointCloud);

    cloud_msgs::cloud_info cloudInfoMsg;
    cloudInfoBuf_.getLastMeas(cloudInfoMsg);

    // The latest IMU measurement records the inertial information when the new
    // point cloud is recorded
    Imu imu;
    imuBuf_.getLastMeas(imu);

    // Initialize the iterative-ESKF by the first PCL
    estimator->processPCL(scan_time_, imu, distortedPointCloud, cloudInfoMsg, outlierPointCloud);

    // Clear all the PCLs before the initialization PCL
    pclBuf_.clean(estimator->getTime());
    cloudInfoBuf_.clean(estimator->getTime());
    outlierBuf_.clean(estimator->getTime());
  }

  void LinsFusion::publishTopics()
  {
    if (g_frame_count < 10 || g_frame_count % 5 != 0)
      return;

    if (pubLaserCloudCornerLast.getNumSubscribers() != 0)
    {
      publishCloudMsg(pubLaserCloudCornerLast,
                      estimator->scan_last_->cornerPointsLessSharp_,
                      ros::Time().fromSec(scan_time_), "/laser_local");
    }
    if (pubLaserCloudSurfLast.getNumSubscribers() != 0)
    {
      publishCloudMsg(pubLaserCloudSurfLast,
                      estimator->scan_last_->surfPointsLessFlat_,
                      ros::Time().fromSec(scan_time_), "/laser_local");
    }
    if (pubOutlierCloudLast.getNumSubscribers() != 0)
    {
      publishCloudMsg(pubOutlierCloudLast,
                      estimator->scan_last_->outlierPointCloud_,
                      ros::Time().fromSec(scan_time_), "/laser_local");
    }

    // Publish the estimated 6-DOF odometry by a YZX-frame convention (e.g. camera
    // frame convention), where Z points forward, X poins leftward, and Y poitns
    // upwards.
    // Notive that the estimator is performed in a XYZ-frame convention,
    // where X points forward, Y...leftward, Z...upward. Therefore, we have to
    // transforme the odometry from XYZ-convention to YZX-convention to meet the
    // mapping module's requirement.
    publishOdometry(scan_time_);
  }

  bool LinsFusion::processPointClouds()
  {
    // Obtain the next PCL
    pclBuf_.itMeas_ = pclBuf_.measMap_.upper_bound(estimator->getTime());
    sensor_msgs::PointCloud2::ConstPtr pclMsg = pclBuf_.itMeas_->second;
    scan_time_ = pclBuf_.itMeas_->first;
    distortedPointCloud->clear();
    pcl::fromROSMsg(*pclMsg, *distortedPointCloud);

    outlierBuf_.itMeas_ = outlierBuf_.measMap_.upper_bound(estimator->getTime());
    sensor_msgs::PointCloud2::ConstPtr outlierMsg = outlierBuf_.itMeas_->second;
    outlierPointCloud->clear();
    pcl::fromROSMsg(*outlierMsg, *outlierPointCloud);

    cloudInfoBuf_.itMeas_ =
        cloudInfoBuf_.measMap_.upper_bound(estimator->getTime());
    cloud_msgs::cloud_info cloudInfoMsg = cloudInfoBuf_.itMeas_->second;

    imuBuf_.getLastTime(last_imu_time_);
    if (last_imu_time_ < scan_time_)
    {
      // ROS_WARN("Wait for more IMU measurement!");
      return false;
    }

    // Propagate IMU measurements between two consecutive scans
    int imu_couter = 0;
    while (estimator->getTime() < scan_time_ &&
           (imuBuf_.itMeas_ = imuBuf_.measMap_.upper_bound(estimator->getTime())) != imuBuf_.measMap_.end())
    {
      double dt = std::min(imuBuf_.itMeas_->first, scan_time_) - estimator->getTime();
      Imu imu = imuBuf_.itMeas_->second;
      estimator->processImu(dt, imu.acc, imu.gyr);
    }

    Imu imu;
    imuBuf_.getLastMeas(imu);

    // Update the iterative-ESKF using a new PCL
    estimator->processPCL(scan_time_, imu, distortedPointCloud, cloudInfoMsg, outlierPointCloud);
    g_frame_count++;

    // Clear all measurements before the current time stamp
    imuBuf_.clean(estimator->getTime());
    pclBuf_.clean(estimator->getTime());
    cloudInfoBuf_.clean(estimator->getTime());
    outlierBuf_.clean(estimator->getTime());

    return true;
  }

  void LinsFusion::performStateEstimation()
  {
    if (imuBuf_.empty() || pclBuf_.empty() || cloudInfoBuf_.empty() ||
        outlierBuf_.empty())
      return;

    if (!estimator->isInitialized())
    {
      processFirstPointCloud();
      return;
    }

    // Iterate all PCL measurements in the buffer
    pclBuf_.getLastTime(last_scan_time_);
    while (!pclBuf_.empty() && estimator->getTime() < last_scan_time_)
    {
      TicToc ts_total;
      if (!processPointClouds())
        break;
      double time_total = ts_total.toc();
      duration_ = (duration_ * scan_counter_ + time_total) / (scan_counter_ + 1);
      scan_counter_++;
      // ROS_INFO_STREAM("Pure-odometry processing time: " << duration_);
      ROS_INFO("process points cost: %lf ms", time_total);
      publishTopics();

      // if (VERBOSE) {
      //   cout << "ba: " << estimator->globalState_.ba_.transpose() << endl;
      //   cout << "bw: " << estimator->globalState_.bw_.transpose() << endl;
      //   cout << "gw: " << estimator->globalState_.gn_.transpose() << endl;
      //   duration_ = (duration_ * scan_counter_ + time_total) / (scan_counter_ +
      //   1); scan_counter_++; ROS_INFO_STREAM(); cout << "Odometry: time: " <<
      //   duration_ << endl;
      // }
    }
  }

  void LinsFusion::alignIMUtoVehicle(const V3D &rpy, const V3D &acc_in,
                                     const V3D &gyr_in, V3D &acc_out,
                                     V3D &gyr_out)
  {
    // i2l in l
    M3D R = Eigen::Matrix3d::Identity();
    // R << 0.999976, -0.000000, -0.006981,
    //     0.000000, 1.000000, -0.000000,
    //     0.006981, 0.000000, 0.999976;
    // Eigen::Vector3d T(-0.3, 0, 0.02);

    acc_out = R.transpose() * acc_in;
    gyr_out = R.transpose() * gyr_in;
  }

  void LinsFusion::publishOdometry(double timeStamp)
  {
    // cout << "pub q: " << math_utils::Q2rpy(estimator->globalState_.ql_).transpose() * 57.3 << endl;
    // cout << "pub p: " << estimator->globalState_.pl_.transpose() << endl;

    // 前面几帧不稳定，跳过
    if (g_frame_count < 10 || g_frame_count % 5 != 0)
      return;

    laserOdometry.header.frame_id = "/world";
    laserOdometry.child_frame_id = "/laser_odom";
    laserOdometry.header.stamp = ros::Time().fromSec(timeStamp);
    laserOdometry.pose.pose.orientation.x = estimator->globalState_.ql_.x();
    laserOdometry.pose.pose.orientation.y = estimator->globalState_.ql_.y();
    laserOdometry.pose.pose.orientation.z = estimator->globalState_.ql_.z();
    laserOdometry.pose.pose.orientation.w = estimator->globalState_.ql_.w();
    laserOdometry.pose.pose.position.x = estimator->globalState_.pl_[0];
    laserOdometry.pose.pose.position.y = estimator->globalState_.pl_[1];
    laserOdometry.pose.pose.position.z = estimator->globalState_.pl_[2];
    pubLaserOdometry.publish(laserOdometry);

    // publish path
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = laserOdometry.header;
    pose_stamped.pose = laserOdometry.pose.pose;
    laserPath.header = laserOdometry.header;
    laserPath.poses.push_back(pose_stamped);
    pubLaserPath.publish(laserPath);

    tf::StampedTransform laserOdometryTrans;
    laserOdometryTrans.frame_id_ = "/world";
    laserOdometryTrans.child_frame_id_ = "/laser_odom";
    laserOdometryTrans.stamp_ = ros::Time().fromSec(timeStamp);
    laserOdometryTrans.setRotation(tf::Quaternion(
        estimator->globalState_.ql_.x(),
        estimator->globalState_.ql_.y(),
        estimator->globalState_.ql_.z(),
        estimator->globalState_.ql_.w()));
    laserOdometryTrans.setOrigin(tf::Vector3(estimator->globalState_.pl_[0],
                                             estimator->globalState_.pl_[1],
                                             estimator->globalState_.pl_[2]));
    tfBroadcaster.sendTransform(laserOdometryTrans);
  }

  // void LinsFusion::performImuBiasEstimation() {
  //   Imu imu;
  //   while (imuBuf_.getSize() != 0) {
  //     imuBuf_.getFirstMeas(imu);
  //     ba_tmp_ += imu.acc - V3D(0, 0, G0);
  //     bw_tmp_ += imu.gyr;
  //     sample_counter_++;
  //     if (sample_counter_ == AVERAGE_NUMS) {
  //       ba_init_ = ba_tmp_ * (1. / sample_counter_);
  //       bw_init_ = bw_tmp_ * (1. / sample_counter_);
  //       isImuCalibrated = true;
  //       ROS_INFO_STREAM("Estimated IMU acceleration bias: \n "
  //                       << ba_init_.transpose() << " and gyroscope bias: \n"
  //                       << bw_init_.transpose());
  //       ba_tmp_.setZero();
  //       bw_tmp_.setZero();
  //       sample_counter_ = 0;
  //       break;
  //     }
  //     imuBuf_.clean(imu.time);
  //     pclBuf_.clean(imu.time);
  //     outlierBuf_.clean(imu.time);
  //     cloudInfoBuf_.clean(imu.time);
  //   }
  // }

} // namespace fusion
