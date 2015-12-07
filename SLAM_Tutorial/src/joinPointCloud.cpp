// =====================================================================================
//
//       Filename:  joinPointCloud.cpp
//
//    Description:
//
//        Version:  1.0
//        Created:  03/12/15 10:55:02
//       Revision:  none
//       Compiler:  g++
//
//         Author:  Cheng Zhao(Henry) (Univeristy of Birmingham), IRobotCheng@gmail.com
//        Company:
//
// =====================================================================================
#include<iostream>
using namespace std;

#include "slamBase.h"

#include <opencv2/core/eigen.hpp>

#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

int main(int argc, char** argv)
{
    ParameterReader PD;
    FRAME frame1, frame2;
    frame1.rgb = cv::imread("../data/rgb1.png");
    frame1.depth = cv::imread("../data/depth1.png", -1);
    frame2.rgb = cv::imread("../data/rgb2.png");
    frame2.depth = cv::imread("../data/depth2.png", -1);
    CAMERA_INTRINSIC_PARAMETERS camera;
    camera.fx = atof(PD.getData("camera.fx").c_str());
    camera.fy = atof(PD.getData("camera.fy").c_str());
    camera.cx = atof(PD.getData("camera.cx").c_str());
    camera.cy = atof(PD.getData("camera.cy").c_str());
    camera.scale = atof(PD.getData("camera.scale").c_str());
/*=============================================================================*/
    cout << "extracting features" << endl;
    string detector = PD.getData("detector");
    string descriptor = PD.getData("descriptor");
    computeKeyPointsAndDesp(frame1, detector, descriptor);
    computeKeyPointsAndDesp(frame2, detector, descriptor);

    cout << "solving PNP" << endl;
    RESULT_OF_PNP result = estimateMotion(frame1, frame2, camera);
    cout << "rvec:" << endl <<result.rvec << endl;
    cout << "tvec:" << endl <<result.tvec << endl;
/*=============================================================================*/
    cout << "translation" << endl;
    cv::Mat R;
    cv::Rodrigues(result.rvec, R);
    Eigen::Matrix3d r;
    cv::cv2eigen(R, r);
    cout << "R:" << endl << R << endl;
    cout << "r:" << endl << r << endl;
    Eigen::AngleAxisd angle(r);
    cout << "angle:" << endl << angle.matrix() << endl;
    /*transform rotation vctor to rotation matrix*/
/*=============================================================================*/
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    cout << "T1:" << endl << T.matrix() << endl;
    T = angle;
    cout << "T2:" << endl << T.matrix() << endl;

    Eigen::Translation<double, 3> trans(result.tvec.at<double>(0,0),
            result.tvec.at<double>(0,1),
            result.tvec.at<double>(0,2));
    cout << "T3:" << endl << T.matrix() << endl;

    T(0,3) = result.tvec.at<double>(0,0);
    T(1,3) = result.tvec.at<double>(0,1);
    T(2,3) = result.tvec.at<double>(0,2);
    cout << "T4:" << endl << T.matrix() << endl;
    /*rotation matrix + translation vector = transforation matrix*/
/*=============================================================================*/
    cout << "converting image to clouds" << endl;
    PointCloud::Ptr cloud1 = image2PointCloud(frame1.rgb, frame1.depth, camera);
    PointCloud::Ptr cloud2 = image2PointCloud(frame2.rgb, frame2.depth, camera);
    cout << "combining clouds" << endl;
    PointCloud::Ptr output (new PointCloud());
    pcl::transformPointCloud(*cloud1, *output, T.matrix());
    *output += *cloud2;
/*=============================================================================*/
    pcl::io::savePCDFile("../data/combinedPlouds.pcd", *output);
    cout << "Final result saved" << endl;
/*=============================================================================*/
    pcl::visualization::CloudViewer viewer("viewer");
    viewer.showCloud(output);
    while(!viewer.wasStopped())
    {

    }

    return 0;
}
