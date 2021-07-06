#include "association.hpp"
#include "pose3D.hpp"

/*********************** Marker *************************/
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/dictionary.hpp>
#include <opencv2/imgproc/imgproc.hpp>
/*********************** ROS ****************************/
#include <string>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseArray.h>
#include <turtlesim/Spawn.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <visualization_msgs/Marker.h>
/*********************** LIBRARIES **********************/
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>

using namespace std;
using namespace cv;
using namespace chrono;

/*
    时间：2019年12月1日
    检测marker姿态并且作为世界坐标系的原点，然后计算得出相机在世界坐标系下的姿态;
    将在openpose下的2D姿态记录下来;
*/
class ImageProcessor
{
public:
	ImageProcessor(int camera_index, string camera_type, string calib_path):it(nh)
	{
		//cout << "回调函数开始" << endl;
		camera_index_ = camera_index;

        this->calib_path_ = calib_path;

		char color_topic[50];
		string image_topic = "/qhd/image_color_rect";
		camera_type += image_topic;
		//sprintf(color_topic,"%s%s",camera_topic,image_topic);
		cout << camera_type << endl;
		image_sub =it.subscribe(camera_type,1,&ImageProcessor::imageCallback,this);

		char key_points_topic[50];
		sprintf(key_points_topic,"/openpose_ros/human_list_%d",camera_index);
		human_keypoints_sub = nh.subscribe(key_points_topic,1,&ImageProcessor::human_keypoints_callback,this);

        listener = new tf::TransformListener;

		colors.resize(5);
		colors[0] = cv::Scalar(0, 0, 255);
		colors[1] = cv::Scalar(0, 255, 0);
		colors[2] = cv::Scalar(255, 0, 0);
		colors[3] = cv::Scalar(0, 255, 255);
		colors[4] = cv::Scalar(255, 0, 255);
	}

	void getMarkerCenter(vector<Point2f>& marker_center)
	{
		marker_center = this->marker_center;
	}

	void getKinectPose(Eigen::Matrix3d &kinect_rot, Eigen::Matrix<double,3,1> &kinect_trans)
	{
		kinect_rot = this->kinect_rot;
		kinect_trans = this->kinect_trans;
	}

	void getHumanJoint(vector<KeyPoint_prob>& key_points_basic)
	{
		key_points_basic = this->key_points_basic;
	}

	void sendMarkerTf(vector<Vec3d>& marker_trans,vector<Vec3d>& marker_rot,vector<int>& ids,string camera_id,Eigen::Matrix3d& temp_rot,Eigen::Matrix<double,3,1>& trans);
	//void sendWorldTf(const Eigen::Matrix<double, 3, 1>& point,const int camera_id, const string& camera_name);
	void sendWorldTf(const Point3d& point,const int axes_id, const string& camera_name);

/************************************* Camera **********************************************/

	void imageCallback(const sensor_msgs::ImageConstPtr& msg);
	void loadCalibrationFiles(std::string& calib_path, cv::Mat& cameraMatrix, cv::Mat& distCoeffs, double scale);
	void initCalibration(double scale);
	void ComputeKinectPose(Eigen::Matrix3d &Cam_rot, Eigen::Matrix<double,3,1> &Cam_trans);
    void waitforKinectPose();

/************************************* Marker function *************************************/
	void getMarkerCoordinate(vector<vector<Point2f>>& corners,vector<int>& ids,vector<Point2f>& marker_center_);

/************************************* human_track *****************************************/

	void human_keypoints_callback(openpose_ros_msgs::OpenPoseHumanList keypoints);
	void human_bounding_boxs(const cv::Mat& img, const vector<Pose>& pose);
	void image_crop(const cv::Mat& src, Bounding_boxs& bbox);
	void draw_human_pose(Mat& image,const vector<Pose> &pose);

	void run(){
		initCalibration(0.5);
		waitforKinectPose();
	}

    vector<double> getCamParam();

	vector<Pose> pose;
	vector<cv::Mat> pose_img;

    Eigen::Matrix3d kinect_rot;
    Eigen::Matrix<double,3,1> kinect_trans;

    cv::Mat image;

    cv::Mat cameraMatrix;


private:
	int camera_index_;

	ros::NodeHandle nh;
	image_transport::ImageTransport it;
    image_transport::Subscriber image_sub;
	ros::Subscriber human_keypoints_sub;
	tf::TransformListener* listener;

    string calib_path_;

	string camera_type;
	cv::Mat distCoeffs;

	cv::Mat image_origin;
	//cv::Size size_color = Size(640, 640 / 1920 * 1080);

	cv::Ptr<aruco::Dictionary> dictionary;

	

	vector<Point2f> marker_center;
	Eigen::Matrix<double,3,1> worldPoint;

	vector<KeyPoint_prob> key_points_basic;
	vector<int> camera_pose;

	int person_num;
	vector<cv::Scalar> colors;
};

vector<double> ImageProcessor::getCamParam() {
    vector<double> param(4);
    param[0] = cameraMatrix.at<double>(0,0);
    param[1] = cameraMatrix.at<double>(1,1);
    param[2] = cameraMatrix.at<double>(0,2);
    param[3] = cameraMatrix.at<double>(1,2);

    return param;
}

void ImageProcessor::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
	try
	{
		cv::Mat color_mat = cv_bridge::toCvShare(msg,"bgr8")->image;
		image = color_mat.clone();
		//cv::resize(image, image, Size(640, 360));
		//ComputeKinectPose(kinect_rot,kinect_trans);
		// if(!pose.empty()){
		// 	draw_human_pose(image, pose);
		// }
		//getMarker(image,marker_center,0);
		//cout << "marker_center:" << marker_center << endl;
		
		//human_bounding_boxs(image, pose);
		// char camera_ID[20];
		// sprintf(camera_ID,"camera_%d",camera_index_);
		// cv::imshow(camera_ID,image);
		// cv::waitKey(3);

		// while(ros::ok()){
		// 	char c = cv::waitKey(10);
        // 	if(c == ' '){
        //     	break;
        // 	}
		// }
	}
	catch(cv_bridge::Exception& e)
	{
		ROS_ERROR("Could not convert from '%s' to 'bgr8'.",msg->encoding.c_str());
		return;
	}
}

void ImageProcessor::loadCalibrationFiles(std::string& calib_path_, cv::Mat& cameraMatrix, cv::Mat& distCoeffs, double scale)
{
	cv::FileStorage fs;

	cv::Mat cameraMatrix_origin;

	string calib_path = calib_path_ + "/calib_color.yaml";
	//std::cout << calib_path << std::endl;
	if(fs.open(calib_path, cv::FileStorage::READ))
	{
	std::cout << "open!" << std::endl;
	fs["cameraMatrix"] >> cameraMatrix_origin;
	std::cout << "matrix load success" << std::endl;
	cameraMatrix = cameraMatrix_origin.clone();
	cameraMatrix.at<double>(0,0) *=scale;
	cameraMatrix.at<double>(1,1) *=scale;
	cameraMatrix.at<double>(0,2) *=scale;
	cameraMatrix.at<double>(1,2) *=scale;

	distCoeffs = cv::Mat::zeros(1,5,CV_64F);

	//fs["distortionCoefficients"] >> distCoeffs;
	//std::cout << "matrix load success" << std::endl;
	fs.release();
	}
}

void ImageProcessor::initCalibration(double scale)
{
	loadCalibrationFiles(calib_path_,cameraMatrix,distCoeffs,scale);
}

/***************************功能：读取marker中心点的坐标，主要是在早期验证相机位姿和TF发布坐标的准确性***********************/
void ImageProcessor::getMarkerCoordinate(vector<vector<Point2f>>& corners,vector<int>& ids,vector<Point2f>& marker_center_)
{
	for(int i=0;i<ids.size();i++)
	{
		Point2f center(0.f,0.f);
		for(int j=0;j<corners[i].size();j++)
		{
			if(ids[i] == 0)
			{
				center += corners[i][j];
			}
		}
		center /= 4.0;
		marker_center_.push_back(center);
	}
}


/*************************功能：将相机的位姿通过tf发送到rviz中**********************************/
void ImageProcessor::sendMarkerTf(vector<Vec3d>& marker_trans,vector<Vec3d>& marker_rot,vector<int>& ids,string camera_id,Eigen::Matrix3d& temp_rot,Eigen::Matrix<double,3,1>& trans)
{
	Mat rot(3, 3, CV_64FC1);
	Mat rot_to_ros(3, 3, CV_64FC1);
	rot_to_ros.at<double>(0,0) = -1.0;
	rot_to_ros.at<double>(0,1) = 0.0;
	rot_to_ros.at<double>(0,2) = 0.0;
	rot_to_ros.at<double>(1,0) = 0.0;
	rot_to_ros.at<double>(1,1) = 0.0;
	rot_to_ros.at<double>(1,2) = 1.0;
	rot_to_ros.at<double>(2,0) = 0.0;
	rot_to_ros.at<double>(2,1) = 1.0;
	rot_to_ros.at<double>(2,2) = 0.0;

	static tf::TransformBroadcaster marker_position_broadcaster;
    for(int i = 0; i < ids.size(); i++)
    {

		if(ids[i] == 0)
		{
			cv::Rodrigues(marker_rot[i], rot);
        	rot.convertTo(rot, CV_64FC1);


        	tf::Matrix3x3 tf_rot(rot.at<double>(0,0), rot.at<double>(0,1), rot.at<double>(0,2),
                             rot.at<double>(1,0), rot.at<double>(1,1), rot.at<double>(1,2),
                             rot.at<double>(2,0), rot.at<double>(2,1), rot.at<double>(2,2));


        	tf::Vector3 tf_trans(marker_trans[i][0], marker_trans[i][1], marker_trans[i][2]);
        	tf::Transform transform(tf_rot, tf_trans);
			transform = transform.inverse();
			Eigen::Quaterniond q_eigen;
			tf::quaternionTFToEigen(transform.getRotation(), q_eigen);
			temp_rot = q_eigen;
			tf::vectorTFToEigen(transform.getOrigin(), trans);
        	ostringstream oss;
        	oss << "marker_" << ids[i];
        	marker_position_broadcaster.sendTransform(tf::StampedTransform(transform, ros::Time::now(), oss.str(), camera_id.c_str()));
		}

    }
}

/********************************功能：将计算后的3D点通过tf发出，验证计算的准确性******************************/
void ImageProcessor::sendWorldTf(const Point3d& point,const int axes_id, const string& camera_name)
{
    static tf::TransformBroadcaster world_position;

    tf::Vector3 tf_trans(point.x, point.y, point.z);
	tf::Quaternion q(0,0,0,1);
    tf::Transform transform(q, tf_trans);
    char marker_name[20];
	sprintf(marker_name, "%d", axes_id);
    world_position.sendTransform(tf::StampedTransform(transform, ros::Time::now(),camera_name.c_str(), marker_name));
}

void ImageProcessor::waitforKinectPose()
{
	tf::StampedTransform transform;
	char kinect_id[10];
	sprintf(kinect_id,"camera_base_%d",camera_index_);
    while(ros::ok()) {
        try
        {
            listener->waitForTransform("/marker_0",kinect_id,ros::Time(0),ros::Duration(3.0));
            listener->lookupTransform("/marker_0",kinect_id,ros::Time(0),transform);
        }
        catch (tf::TransformException &ex)
        {
            ROS_ERROR("%s",ex.what());
            ros::Duration(1.0).sleep();
            continue;
        }
        Eigen::Quaterniond q(transform.getRotation().getW(),transform.getRotation().getX(),transform.getRotation().getY(),transform.getRotation().getZ());
        Eigen::Vector3d trans(transform.getOrigin().getX(),transform.getOrigin().getY(),transform.getOrigin().getZ());
        kinect_rot=q.toRotationMatrix();
        kinect_trans = trans;
        break;
    }
	ROS_INFO("Camera_%d pose has listener sunccessfully!", this->camera_index_);
    delete listener;

}

/*******************************功能：通过tf监听得到kinect相机的位姿，用于在录制的视频播放;
 *                              在实际中我们会通过marker得到相机的位姿，这段程序将不被使用******************************/
void ImageProcessor::ComputeKinectPose(Eigen::Matrix3d &Cam_rot, Eigen::Matrix<double,3,1> &Cam_trans)
{
	tf::StampedTransform transform;
	char kinect_id[10];
	sprintf(kinect_id,"camera_base_%d",camera_index_);
 	try
    {
        listener->waitForTransform("/marker_0",kinect_id,ros::Time(0),ros::Duration(3.0));
        listener->lookupTransform("/marker_0",kinect_id,ros::Time(0),transform);
    }
    catch (tf::TransformException &ex)
    {
        ROS_ERROR("%s",ex.what());
        ros::Duration(1.0).sleep();
    }
	Eigen::Quaterniond q(transform.getRotation().getW(),transform.getRotation().getX(),transform.getRotation().getY(),transform.getRotation().getZ());
    Eigen::Vector3d trans(transform.getOrigin().getX(),transform.getOrigin().getY(),transform.getOrigin().getZ());
	Cam_rot=q.toRotationMatrix();
	Cam_trans = trans;
}

/******************************功能：读取openpose下的2d关节信息************************************/
void ImageProcessor::human_keypoints_callback(openpose_ros_msgs::OpenPoseHumanList keypoints)
{
	key_points_basic.clear();
	person_num = keypoints.num_humans;
	pose.clear();
	//vector<vector<double> > distance_pool;
	//distance_pool.resize(person_num);
	if(person_num > 0)
	{
		for(int person=0;person < person_num;++person)
		{
			auto body_keypoints = keypoints.human_list[person].body_key_points_with_prob;

			int count = 0;
			double prob_sum = 0.0;
			for(int i=0;i < body_keypoints.size();i++)
			{
				if(body_keypoints[i].prob > 0.0)
				{
					prob_sum += body_keypoints[i].prob;
					count++;
				}
			}
			double prob_eval = prob_sum/count;
			if(prob_eval < 0.4)
			{
				continue;
			}
            Pose pose_new;
			pose_new.setLabel(-1);
            pose_new.setCameraId(camera_index_);
			// pose_new.setColor(this->colors[person]);
            pose_new.setPose(keypoints, person);
            pose.push_back(pose_new);
		}
	}
}

void ImageProcessor::human_bounding_boxs(const cv::Mat& img, const vector<Pose>& pose){
	if(img.empty() || pose.empty())
		return;
	
	pose_img.clear();
	for(int i=0;i<pose.size(); ++i){
		Pose person = pose[i];
		Bounding_boxs bbox = person.bounding_box;
		image_crop(img, bbox);
	}
	cout << pose_img.size() << endl;
}

/************************************在图像中画出人体骨骼框架**********************************/
void ImageProcessor::draw_human_pose(Mat& image, const vector<Pose> &pose)
{
	if(image.empty())
	{
		ROS_ERROR("Can't open image!");
	}
	if(pose.empty())
	{
		ROS_ERROR("No Human!");
	}

	pose_img.clear();
	for(int i=0;i<pose.size(); ++i){
		Pose person = pose[i];
		Bounding_boxs bbox = person.bounding_box;
		cv::Scalar col = person.color;

		// cout << param.size() << endl;
		// cout << param[0] << " " << param[1] << " " << param[2] << " " << param[3] << endl;
		cv::rectangle(image,cvPoint(bbox.x,bbox.y), cvPoint(bbox.width,bbox.height), col, 4);
		cv::putText(image, to_string(person.label), cv::Point(bbox.x+20, bbox.y-10),cv::FONT_HERSHEY_SCRIPT_COMPLEX, 1 , col);
		//cv::rectangle(image,cvPoint(param[0],param[0]), cvPoint(param[0],param[0]), cv::Scalar(0, 0, 255));

		for(int j=0; j<person.pose_joints.size(); ++j){
			circle(image, cv::Point(person.pose_joints[j].x, person.pose_joints[j].y), 3, col, -1, 8);
			//circle(image, cv::Point(person.center_point.x, person.center_point.y), 5,col, -1 ,8);
		}

		//image_crop(image, bbox);

	}
	
	// char winName[50];
	// for(int i=0; i<pose_img.size(); ++i){
	// 	sprintf(winName, "No%d", i);
	// 	cv::imshow(winName, pose_img[i]);
	// }
	// cv::waitKey(3);

	
	//ROS_INFO("Human pose drawed.");
}


void ImageProcessor::image_crop(const cv::Mat& src, Bounding_boxs& bbox){
	int xmin = bbox.x;
	int xmax = bbox.width;
	int ymin = bbox.y;
	int ymax = bbox.height;
	
	if(xmin <= 0)
        xmin = 0;
    if(xmin >= src.cols)
        xmin = src.cols;

	if(xmax <= 0)
        xmax = 0;
    if(xmax >= src.cols)
        xmax = src.cols;

	if(ymin <= 0)
        ymin = 0;
    if(ymin >= src.rows)
        ymin = src.rows;

	if(ymax <= 0)
        ymax = 0;
    if(ymax >= src.rows)
        ymax = src.rows;

	cv::Mat cropImage = src(cv::Range(ymin, ymax), cv::Range(xmin, xmax));
	//cv::imwrite("/home/agent/1.jpg",cropImage);
	pose_img.push_back(cropImage);
}


int main(int argc,char **argv)
{
	bool check = true;


	ros::init(argc,argv,"marker_node");
	ros::NodeHandle n;
	//tf::TransformListener listener;
	ros::Publisher pose_pub = n.advertise<visualization_msgs::Marker>("visualization_marker",1);
	ros::Rate loop_rate(20);

	if(argc > 1) {
		check = false;
		ROS_INFO("[INFO]: Disable iteration check\n");
	}

   // string calib_path_1 = "/home/luk/agent_ws/calibration_data/1";
   // string calib_path_2 = "/home/luk/agent_ws/calibration_data/2";
   // string calib_path_3 = "/home/luk/agent_ws/calibration_data/3";

    string calib_path_1 = "/home/agent/catkin_ws/src/iai_kinect2/kinect2_bridge/data/003415165047";
    string calib_path_2 = "/home/agent/catkin_ws/src/iai_kinect2/kinect2_bridge/data/092465240847";
    string calib_path_3 = "/home/agent/catkin_ws/src/cv_camera/calibration_data";
	string calib_path_4 = "/home/agent/catkin_ws/src/iai_kinect2/kinect2_bridge/data/007538564147";

    ImageProcessor cam_a(1,"kinect2_1", calib_path_1);
	ImageProcessor cam_b(2,"kinect2_2", calib_path_2);
    ImageProcessor cam_c(3,"camera_3", calib_path_3);
	//ImageProcessor cam_c(3,"kinect2_3", calib_path_4);

	cam_a.run();
	cam_b.run();
	cam_c.run();



    PerFrameAssociation ac(check);

    ac.addCamParam(cam_a.getCamParam());
    ac.addCamParam(cam_b.getCamParam());
    ac.addCamParam(cam_c.getCamParam());

    ac.addCamPose(cam_a.kinect_rot, cam_a.kinect_trans);
    ac.addCamPose(cam_b.kinect_rot, cam_b.kinect_trans);
    ac.addCamPose(cam_c.kinect_rot, cam_c.kinect_trans);

    Track3DPose tracker(pose_pub);
    tracker.CameraMatrix.push_back(cam_a.cameraMatrix);
    tracker.CameraMatrix.push_back(cam_b.cameraMatrix);
    tracker.CameraMatrix.push_back(cam_c.cameraMatrix);
    tracker.addCamPose(cam_a.kinect_rot, cam_a.kinect_trans);
    tracker.addCamPose(cam_b.kinect_rot, cam_b.kinect_trans);
    tracker.addCamPose(cam_c.kinect_rot, cam_c.kinect_trans);

	while(ros::ok()){
        ros::spinOnce();
        ac.clearPoses();
        ac.addAvailablePose(cam_a.pose);
        ac.addAvailablePose(cam_b.pose);
        ac.addAvailablePose(cam_c.pose);
        ac.generatePair();
        ac.associate();


        tracker.image[0] = cam_a.image;
        tracker.image[1] = cam_b.image;
        tracker.image[2] = cam_c.image;
        tracker.poses = ac.AvailablePose;
        tracker.pose_ass = ac.extract2DAssociation();
        tracker.getAvailableSkeleton(ac.extract3DPoses());


        //auto t1 = chrono::steady_clock::now();

        //auto t2 = chrono::steady_clock::now();
        //ROS_INFO("Time is used: %f",chrono::duration<double>(t2-t1).count());
        //cout << "Time is used: " << chrono::duration<double>(t2-t1).count() << endl;


        // tracker.verify2DPoseAss();
		//hm.getPose(ac.extract3DPoses());
		//hm.setJointAndName();
        //tracker.tracking(5);


        //tracker.get3Dskeletons(ac.extract3DPoses());
        //tracker.draw_human_pose();

		ros::spinOnce();
		//loop_rate.sleep();
	}

	return 0;
}
