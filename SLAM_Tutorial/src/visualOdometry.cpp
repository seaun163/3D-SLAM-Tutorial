// =====================================================================================
//
//       Filename:  visualOdometry.cpp
//
//    Description:
//
//        Version:  1.0
//        Created:  04/12/15 14:13:00
//       Revision:  none
//       Compiler:  g++
//
//         Author:  Cheng Zhao(Henry) (Univeristy of Birmingham), IRobotCheng@gmail.com
//        Company:
//
// =====================================================================================
#include <slamBase.h>

int main(int argc, char** argv)
{    
    cout << "Initializing ..." << endl;
    ParameterReader PD;
    int startIndex = atoi(PD.getData( "start_index" ).c_str());
    int endIndex = atoi(PD.getData( "end_index" ).c_str());
    string detector = PD.getData("detector");
    string descriptor = PD.getData("descriptor");
    CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
    pcl::visualization::CloudViewer viewer("viewer");
    bool visualize = PD.getData("visualize_pointcloud") == string("yes");
    int min_inliers = atoi(PD.getData("min_inliers").c_str());
    double max_norm = atof(PD.getData("max_norm").c_str());

    int currIndex = startIndex;
    FRAME lastFrame = readFrame(currIndex, PD);

    computeKeyPointsAndDesp(lastFrame, detector, descriptor);
    PointCloud::Ptr cloud = image2PointCloud(lastFrame.rgb, lastFrame.depth, camera);


    for(currIndex = startIndex + 1; currIndex < endIndex; currIndex++)
    {
        cout << "Reading files" << currIndex << endl;
        FRAME currFrame = readFrame(currIndex, PD);
        computeKeyPointsAndDesp(currFrame, detector, descriptor);

        RESULT_OF_PNP result = estimateMotion(lastFrame, currFrame, camera);
        if(result.inliers < min_inliers)
            continue;
        double norm = normofTransform(result.rvec, result.tvec);
        cout << "norm = " << norm << endl;
        if(norm >= max_norm)
            continue;

        Eigen::Isometry3d T = cvMat2Eigen(result.rvec, result.tvec);
        cout << "T = " << T.matrix() << endl;
        cloud = joinPointCloud(cloud, currFrame, T, camera);
        if(visualize == true)
            viewer.showCloud(cloud);

        lastFrame = currFrame;
    }

    pcl::io::savePCDFile( "../data/result.pcd", *cloud );
    return 0;
}
