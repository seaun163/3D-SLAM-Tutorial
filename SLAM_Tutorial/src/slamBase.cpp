// =====================================================================================
//
//       Filename:  slamBase.cpp
//
//    Description:
//
//        Version:  1.0
//        Created:  01/12/15 15:45:52
//       Revision:  none
//       Compiler:  g++
//
//         Author:  Cheng Zhao(Henry) (Univeristy of Birmingham), IRobotCheng@gmail.com
//        Company:
//
// =====================================================================================
#include <slamBase.h>



PointCloud::Ptr image2PointCloud( cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    PointCloud::Ptr cloud ( new PointCloud );

    for (int m = 0; m < depth.rows; m++)
        for (int n=0; n < depth.cols; n++)
        {
            ushort d = depth.ptr<ushort>(m)[n];
            if (d == 0)
                continue;

	    PointT p;
            p.z = double(d) / camera.scale;
            p.x = (n - camera.cx) * p.z / camera.fx;
            p.y = (m - camera.cy) * p.z / camera.fy;
            p.b = rgb.ptr<uchar>(m)[n*3];
            p.g = rgb.ptr<uchar>(m)[n*3+1];
            p.r = rgb.ptr<uchar>(m)[n*3+2];

            cloud->points.push_back( p );
        }

    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense = false;

    return cloud;
}

cv::Point3f point2dTo3d( cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    cv::Point3f p;
    p.z = double( point.z ) / camera.scale;
    p.x = ( point.x - camera.cx) * p.z / camera.fx;
    p.y = ( point.y - camera.cy) * p.z / camera.fy;
    return p;
}

void computeKeyPointsAndDesp(FRAME& frame, string detector, string descriptor)
{
    cv::initModule_nonfree();
    cv::Ptr<cv::FeatureDetector> _detector = cv::FeatureDetector::create( detector.c_str() );
    cv::Ptr<cv::DescriptorExtractor> _descriptor = cv::DescriptorExtractor::create( descriptor.c_str() );

    if(!_detector || !_descriptor)
    {
        cerr << "Unknow detector or descriptor type!" << detector << ", " << descriptor << endl;
        return;
    }

    _detector->detect(frame.rgb, frame.kp);
    _descriptor->compute(frame.rgb, frame.kp, frame.desp);
}

RESULT_OF_PNP estimateMotion(FRAME& frame1, FRAME& frame2, CAMERA_INTRINSIC_PARAMETERS& camera)
{
    static ParameterReader PReader;
    vector< cv::DMatch > matches;
    cv::FlannBasedMatcher matcher;
    RESULT_OF_PNP result;

    matcher.match( frame1.desp, frame2.desp, matches);
    cout<<"Find total "<<matches.size()<<" matches."<<endl;

    vector< cv::DMatch > goodMatches;
    double minDis = 9999;
    double good_match_threshold = atof( PReader.getData("good_match_threshold").c_str() );
    for ( size_t i=0; i<matches.size(); i++ )
    {
        if ( matches[i].distance < minDis )
            minDis = matches[i].distance;
    }

    for ( size_t i=0; i<matches.size(); i++ )
    {
        if (matches[i].distance < good_match_threshold*minDis)
            goodMatches.push_back( matches[i] );
    }
    cout<<"good matches= "<<goodMatches.size()<<endl;
    if(goodMatches.size() <= atof(PReader.getData("min_good_match").c_str()))
    {
        result.inliers = -1;
        return result;
    }

    vector<cv::Point3f> pts_obj;
    vector<cv::Point2f> pts_img;

    for (size_t i=0; i<goodMatches.size(); i++)
    {
        cv::Point2f p = frame1.kp[goodMatches[i].queryIdx].pt;
        ushort d = frame1.depth.ptr<ushort>( int(p.y) )[ int(p.x) ];
        if (d == 0)
            continue;
        cv::Point3f pt ( p.x, p.y, d );
        cv::Point3f pd = point2dTo3d( pt, camera );
        pts_obj.push_back( pd );
        pts_img.push_back( cv::Point2f( frame2.kp[goodMatches[i].trainIdx].pt ) );
    }

    if(pts_obj.size() == 0 || pts_img.size() == 0)
    {
        result.inliers = -1;
        return result;
    }

    double camera_matrix_data[3][3] = {
        {camera.fx, 0, camera.cx},
        {0, camera.fy, camera.cy},
        {0, 0, 1}};

    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
    cv::Mat rvec, tvec, inliers;
    cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, 100, inliers );

    result.rvec = rvec;
    result.tvec = tvec;
    result.inliers = inliers.rows;

    return result;
}

ParameterReader::ParameterReader():filename("../data/parameters.txt")
{
    ifstream fin(filename.c_str());
    if(!fin)
    {
        cerr << "parameters file does not exit." << endl;
        return;
    }

    while(!fin.eof())
    {
        string str;
        getline(fin, str);
        if(str[0] == '#')
            continue;
        int pos = str.find("=");
        if(pos == -1)
            continue;
        string key = str.substr(0, pos);
        string value = str.substr(pos+1, str.length());
        data[key] = value;
        if(!fin.good())
            break;
    }
}

string ParameterReader::getData(string key)
{
    map<string, string>::iterator iter = data.find(key);

    if(iter == data.end())
    {
        cerr << "parameters name  " << key << "not found" << endl;
        return string("NOT_FOUND");
    }

    return iter->second;
}

Eigen::Isometry3d cvMat2Eigen( cv::Mat& rvec, cv::Mat& tvec )
{
    cv::Mat R;
    cv::Rodrigues( rvec, R );
    Eigen::Matrix3d r;
    cv::cv2eigen(R, r);
    Eigen::AngleAxisd angle(r);

    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    Eigen::Translation<double,3> trans(tvec.at<double>(0,0), tvec.at<double>(0,1), tvec.at<double>(0,2));
    T = angle;
    T(0,3) = tvec.at<double>(0,0);
    T(1,3) = tvec.at<double>(0,1);
    T(2,3) = tvec.at<double>(0,2);
    return T;
}

PointCloud::Ptr joinPointCloud( PointCloud::Ptr original, FRAME& newFrame, Eigen::Isometry3d T, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    PointCloud::Ptr newCloud = image2PointCloud( newFrame.rgb, newFrame.depth, camera );
    PointCloud::Ptr output (new PointCloud());
    pcl::transformPointCloud( *original, *output, T.matrix() );
    *newCloud += *output;

    static pcl::VoxelGrid<PointT> voxel;
    static ParameterReader PD;
    double gridSize = atof(PD.getData("voxel_grid").c_str());
    voxel.setLeafSize(gridSize, gridSize, gridSize);
    voxel.setInputCloud(newCloud);
    PointCloud::Ptr tmp(new PointCloud());
    voxel.filter(*tmp);
    return tmp;
}

FRAME readFrame(int index, ParameterReader& PD)
{
    FRAME f;
    f.frameID = index;
    string rgbDir = PD.getData("rgb_dir");
    string depthDir = PD.getData("depth_dir");
    string rgbExt = PD.getData("rgb_extension");
    string depthExt = PD.getData("depth_extension");
    string filename;
    stringstream stringIO;

    stringIO << rgbDir << index << rgbExt;
    stringIO >> filename;
    f.rgb = cv::imread(filename);
    stringIO.clear();
    filename.clear();

    stringIO << depthDir << index << depthExt;
    stringIO >> filename;
    f.depth = cv::imread(filename, -1);
    stringIO.clear();
    filename.clear();

    return f;
}

double normofTransform(cv::Mat rvec, cv::Mat tvec)
{
    return fabs(cv::norm(tvec)) + fabs(min(cv::norm(rvec), 2*M_PI - cv::norm(rvec)));
}


CHECK_RESULT checkKeyframes( FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opti, bool is_loops)
{
    static ParameterReader pd;
    static int min_inliers = atoi( pd.getData("min_inliers").c_str() );
    static double max_norm = atof( pd.getData("max_norm").c_str() );
    static double keyframe_threshold = atof( pd.getData("keyframe_threshold").c_str() );
    static double max_norm_lp = atof( pd.getData("max_norm_lp").c_str() );
    static CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
    static g2o::RobustKernel* robustKernel = g2o::RobustKernelFactory::instance()->construct( "Cauchy" );
   
    RESULT_OF_PNP result = estimateMotion( f1, f2, camera );   
    if ( result.inliers < min_inliers ) 
        return NOT_MATCHED;     
    double norm = normofTransform(result.rvec, result.tvec);
    Eigen::Isometry3d T = cvMat2Eigen( result.rvec, result.tvec ); 

    if ( is_loops == false )
    {
        if ( norm >= max_norm )
            return TOO_FAR_AWAY;
    }
    else
    {
        if ( norm >= max_norm_lp)
            return TOO_FAR_AWAY;
    }
    if ( norm <= keyframe_threshold )
        return TOO_CLOSE; 
    if (is_loops == false)
    {
        g2o::VertexSE3 *v = new g2o::VertexSE3();
        v->setId( f2.frameID );
        v->setEstimate( Eigen::Isometry3d::Identity() );
        opti.addVertex(v);
    }

    g2o::EdgeSE3* edge = new g2o::EdgeSE3();
    edge->vertices() [0] = opti.vertex( f1.frameID );
    edge->vertices() [1] = opti.vertex( f2.frameID );
    edge->setRobustKernel( robustKernel );
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix< double, 6,6 >::Identity();
    information(0,0) = information(1,1) = information(2,2) = information(3,3) = information(4,4) = information(5,5) = 100;
    edge->setInformation( information );   
    edge->setMeasurement( T.inverse() );
    opti.addEdge(edge);
    return KEYFRAME;
}

void checkNearbyLoops( vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti )
{
    static ParameterReader pd;
    static int nearby_loops = atoi( pd.getData("nearby_loops").c_str() );
    
    if ( frames.size() <= nearby_loops )
    {
        for (size_t i=0; i<frames.size(); i++)
        {
            checkKeyframes( frames[i], currFrame, opti, true );
        }
    }
    else
    {
        for (size_t i = frames.size()-nearby_loops; i<frames.size(); i++)
        {
            checkKeyframes( frames[i], currFrame, opti, true );
        }
    }
}

void checkRandomLoops( vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti )
{
    static ParameterReader pd;
    static int random_loops = atoi( pd.getData("random_loops").c_str() );
    srand( (unsigned int) time(NULL) );
    
    if ( frames.size() <= random_loops )
    {
        for (size_t i=0; i<frames.size(); i++)
        {
            checkKeyframes( frames[i], currFrame, opti, true );
        }
    }
    else
    {
        for (int i=0; i<random_loops; i++)
        {
            int index = rand()%frames.size();
            checkKeyframes( frames[index], currFrame, opti, true );
        }
    }
}

