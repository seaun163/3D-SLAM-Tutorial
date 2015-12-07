#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef g2o::BlockSolver_6_3 SlamBlockSolver; 
typedef g2o::LinearSolverCSparse< SlamBlockSolver::PoseMatrixType > SlamLinearSolver; 

struct CAMERA_INTRINSIC_PARAMETERS
{
    double cx, cy, fx, fy, scale;
};

struct FRAME
{
    int frameID;
    cv::Mat rgb, depth;
    cv::Mat desp;
    vector<cv::KeyPoint> kp;
};

struct RESULT_OF_PNP
{
    cv::Mat rvec, tvec;
    int inliers;
};

enum CHECK_RESULT {NOT_MATCHED = 0, TOO_FAR_AWAY, TOO_CLOSE, KEYFRAME};

PointCloud::Ptr image2PointCloud(cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera);

cv::Point3f point2dTo3d(cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera);

void computeKeyPointsAndDesp(FRAME& frame, string detector, string descriptor);

RESULT_OF_PNP estimateMotion(FRAME& frame1, FRAME& frame2, CAMERA_INTRINSIC_PARAMETERS& camera);

Eigen::Isometry3d cvMat2Eigen(cv::Mat& rvec, cv::Mat& tvec);

PointCloud::Ptr joinPointCloud(PointCloud::Ptr original, FRAME& newFrame, Eigen::Isometry3d T, CAMERA_INTRINSIC_PARAMETERS& camera);

class ParameterReader
{
    public:
        ParameterReader();
        string getData(string key);
    public:
        map<string, string> data;
        string filename;
};

inline static CAMERA_INTRINSIC_PARAMETERS getDefaultCamera()
{
    ParameterReader pd;
    CAMERA_INTRINSIC_PARAMETERS camera;
    camera.fx = atof( pd.getData( "camera.fx" ).c_str());
    camera.fy = atof( pd.getData( "camera.fy" ).c_str());
    camera.cx = atof( pd.getData( "camera.cx" ).c_str());
    camera.cy = atof( pd.getData( "camera.cy" ).c_str());
    camera.scale = atof( pd.getData( "camera.scale" ).c_str() );
    return camera;
}

FRAME readFrame(int index, ParameterReader& PD);

double normofTransform(cv::Mat rvec, cv::Mat tvec);

CHECK_RESULT checkKeyframes(FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opti, bool is_loops = false);

void checkNearbyLoops(vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti);

void checkRandomLoops(vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti);










//the following are UBUNTU/LINUX ONLY terminal color
#define RESET "\033[0m"
#define BLACK "\033[30m" /* Black */
#define RED "\033[31m" /* Red */
#define GREEN "\033[32m" /* Green */
#define YELLOW "\033[33m" /* Yellow */
#define BLUE "\033[34m" /* Blue */
#define MAGENTA "\033[35m" /* Magenta */
#define CYAN "\033[36m" /* Cyan */
#define WHITE "\033[37m" /* White */
#define BOLDBLACK "\033[1m\033[30m" /* Bold Black */
#define BOLDRED "\033[1m\033[31m" /* Bold Red */
#define BOLDGREEN "\033[1m\033[32m" /* Bold Green */
#define BOLDYELLOW "\033[1m\033[33m" /* Bold Yellow */
#define BOLDBLUE "\033[1m\033[34m" /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m" /* Bold Magenta */
#define BOLDCYAN "\033[1m\033[36m" /* Bold Cyan */
#define BOLDWHITE "\033[1m\033[37m" /* Bold White */

