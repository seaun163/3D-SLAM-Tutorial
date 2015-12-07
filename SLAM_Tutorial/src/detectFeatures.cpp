// =====================================================================================
//
//       Filename:  detectFeature.cpp
//
//    Description:
//
//        Version:  1.0
//        Created:  01/12/15 21:10:09
//       Revision:  none
//       Compiler:  g++
//
//         Author:  Cheng Zhao(Henry) (Univeristy of Birmingham), IRobotCheng@gmail.com
//        Company:
//
// =====================================================================================
#include<iostream>
#include "slamBase.h"
using namespace std;
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>

int main( int argc, char** argv )
{
    CAMERA_INTRINSIC_PARAMETERS C;
    C.cx = 325.5;
    C.cy = 253.5;
    C.fx = 518.0;
    C.fy = 519.0;
    C.scale = 1000.0;
    double camera_matrix_data[3][3] = {
        {C.fx, 0, C.cx},
        {0, C.fy, C.cy},
        {0, 0, 1}};

    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
  

    cv::Mat rgb1 = cv::imread( "../data/rgb1.png");
    cv::Mat rgb2 = cv::imread( "../data/rgb2.png");
    cv::Mat depth1 = cv::imread( "../data/depth1.png", -1);
    cv::Mat depth2 = cv::imread( "../data/depth2.png", -1);

    cv::initModule_nonfree();
    cv::Ptr<cv::FeatureDetector> _detector = cv::FeatureDetector::create( "SIFT" );
    cv::Ptr<cv::DescriptorExtractor> _descriptor = cv::DescriptorExtractor::create( "SIFT" );
// =====================================================================================
    vector< cv::KeyPoint > kp1, kp2;
    _detector->detect( rgb1, kp1 );
    _detector->detect( rgb2, kp2 );
    cout<<"Key points of two images: "<<kp1.size()<<", "<<kp2.size()<<endl;
    cv::Mat imgKeypoints;
    cv::drawKeypoints( rgb1, kp1, imgKeypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    cv::imshow( "keypoints", imgKeypoints );
    cv::imwrite( "../data/keypoints.png", imgKeypoints );
    cv::waitKey(0);
// =====================================================================================
    cv::Mat desp1, desp2;
    _descriptor->compute( rgb1, kp1, desp1 );
    _descriptor->compute( rgb2, kp2, desp2 );
    vector< cv::DMatch > matches;
    cv::FlannBasedMatcher matcher;
    matcher.match( desp1, desp2, matches );
    cout<<"Find total "<<matches.size()<<" matches."<<endl;
    cv::Mat imgMatches;
    cv::drawMatches( rgb1, kp1, rgb2, kp2, matches, imgMatches );
    cv::imshow( "matches", imgMatches );
    cv::imwrite( "../data/matches.png", imgMatches );
    cv::waitKey( 0 );
// =====================================================================================
    vector< cv::DMatch > goodMatches;
    double minDis = 9999;
    for ( size_t i=0; i<matches.size(); i++ )
    {
        if ( matches[i].distance < minDis )
            minDis = matches[i].distance;
    }

    for ( size_t i=0; i<matches.size(); i++ )
    {
        if (matches[i].distance < 4*minDis)
            goodMatches.push_back( matches[i] );
    }
    cout<<"good matches= "<<goodMatches.size()<<endl;
    cv::Mat imgGoodMatches;
    cv::drawMatches( rgb1, kp1, rgb2, kp2, goodMatches, imgGoodMatches );
    cv::imshow( "good matches", imgGoodMatches );
    cv::imwrite( "../data/goodMatches.png", imgGoodMatches );
    cv::waitKey(0);
// =====================================================================================
    vector<cv::Point3f> pts_obj;
    vector<cv::Point2f> pts_img;

    for (size_t i=0; i<goodMatches.size(); i++)
    {
        cv::Point2f p = kp1[goodMatches[i].queryIdx].pt;
        ushort d = depth1.ptr<ushort>( int(p.y) )[ int(p.x) ];
        if (d == 0)
            continue;
        cv::Point3f pt ( p.x, p.y, d );
        cv::Point3f pd = point2dTo3d( pt, C );
        pts_obj.push_back( pd );
        pts_img.push_back( cv::Point2f( kp2[goodMatches[i].trainIdx].pt ) );
    }

    cv::Mat rvec, tvec, inliers;
    cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, 100, inliers );

    cout<<"inliers: "<<inliers.rows<<endl;
    cout<<"R="<<rvec<<endl;
    cout<<"t="<<tvec<<endl;

    vector<cv::DMatch> inliersMatches;
    cv::Mat imgInliers;
    for (size_t i=0; i<inliers.rows; i++)
    {
        inliersMatches.push_back( goodMatches[inliers.ptr<int>(i)[0]] );
    }
    cv::drawMatches( rgb1, kp1, rgb2, kp2, inliersMatches, imgInliers );
    cv::imshow( "inlier matches", imgInliers );
    cv::imwrite( "../data/inliers.png", imgInliers);
    cv::waitKey( 0 );

    return 0;
}
