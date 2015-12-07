// =====================================================================================
//
//       Filename:  slamBase.cpp
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
#include "slamBase.h"

int main( int argc, char** argv )
{
    cout<<"Initializing ..."<<endl;
    ParameterReader pd;
    int startIndex = atoi( pd.getData( "start_index" ).c_str() );
    int endIndex = atoi( pd.getData( "end_index"   ).c_str() );
    string detector = pd.getData( "detector" );
    string descriptor = pd.getData( "descriptor" );
    CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
    double keyframe_threshold = atof( pd.getData("keyframe_threshold").c_str() );
    bool check_loop_closure = pd.getData("check_loop_closure")==string("yes");
/*===========================================================================================*/
    vector< FRAME > keyframes;    
    int currIndex = startIndex; 
    FRAME currFrame = readFrame( currIndex, pd );     
    computeKeyPointsAndDesp( currFrame, detector, descriptor );
    PointCloud::Ptr cloud = image2PointCloud( currFrame.rgb, currFrame.depth, camera );
    keyframes.push_back( currFrame );
/*===========================================================================================*/    
    SlamLinearSolver* linearSolver = new SlamLinearSolver();
    linearSolver->setBlockOrdering( false );
    SlamBlockSolver* blockSolver = new SlamBlockSolver( linearSolver );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( blockSolver );
    g2o::SparseOptimizer globalOptimizer; 
    globalOptimizer.setAlgorithm( solver ); 
    globalOptimizer.setVerbose( false );
/*===========================================================================================*/    
    g2o::VertexSE3* v = new g2o::VertexSE3();
    v->setId( currIndex );
    v->setEstimate( Eigen::Isometry3d::Identity() ); 
    v->setFixed( true ); 
    globalOptimizer.addVertex( v );
/*===========================================================================================*/     
    for ( currIndex=startIndex+1; currIndex<endIndex; currIndex++ )
    {
        cout<<"Reading files "<<currIndex<<endl;
        FRAME currFrame = readFrame( currIndex,pd );
        computeKeyPointsAndDesp( currFrame, detector, descriptor );
        CHECK_RESULT result = checkKeyframes( keyframes.back(), currFrame, globalOptimizer );
        switch (result)
        {
        case NOT_MATCHED:
            cout<<RED"Not enough inliers."<<endl;
            break;
        case TOO_FAR_AWAY:
            cout<<RED"Too far away, may be an error."<<endl;
            break;
        case TOO_CLOSE:         
            cout<<RESET"Too close, not a keyframe"<<endl;
            break;
        case KEYFRAME:
            cout<<GREEN"This is a new keyframe"<<endl;
            if (check_loop_closure)
            {
                checkNearbyLoops( keyframes, currFrame, globalOptimizer );
                checkRandomLoops( keyframes, currFrame, globalOptimizer );
            }
            keyframes.push_back( currFrame );
            break;
        default:
            break;
        }      
      }
/*===========================================================================================*/
    cout<<RESET"optimizing pose graph, vertices: "<<globalOptimizer.vertices().size()<<endl;
    globalOptimizer.save("../data/result_before.g2o");
    globalOptimizer.initializeOptimization();
    globalOptimizer.optimize( 100 ); 
    globalOptimizer.save( "../data/result_after.g2o" );
    cout<<"Optimization done."<<endl;
/*===========================================================================================*/
    cout<<"saving the point cloud map..."<<endl;
    PointCloud::Ptr output ( new PointCloud() ); 
    PointCloud::Ptr tmp ( new PointCloud() );

    pcl::VoxelGrid<PointT> voxel; 
    double gridsize = atof( pd.getData( "voxel_grid" ).c_str() );
    voxel.setLeafSize( gridsize, gridsize, gridsize );
    pcl::PassThrough<PointT> pass;
    pass.setFilterFieldName("z");
    pass.setFilterLimits( 0.0, 4.0 ); 

    for (size_t i=0; i<keyframes.size(); i++)
    {
        g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex( keyframes[i].frameID ));
        Eigen::Isometry3d pose = vertex->estimate(); 
        PointCloud::Ptr newCloud = image2PointCloud( keyframes[i].rgb, keyframes[i].depth, camera );
        voxel.setInputCloud( newCloud );
        voxel.filter( *tmp );
        pass.setInputCloud( tmp );
        pass.filter( *newCloud );

        pcl::transformPointCloud( *newCloud, *tmp, pose.matrix() );
        *output += *tmp;
        tmp->clear();
        newCloud->clear();
    }

    voxel.setInputCloud( output );
    voxel.filter( *tmp );
    pcl::io::savePCDFile( "../data/result.pcd", *tmp ); 
    cout<<"Final map is saved."<<endl;
    globalOptimizer.clear();
/*===========================================================================================*/
    return 0;
}
