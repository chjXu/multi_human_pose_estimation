/*********************** opencv *************************/
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
/*********************** Marker *************************/
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/dictionary.hpp>
#include <vector>
/*********************** ROS ****************************/
#include <string>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseArray.h>
#include <turtlesim/Spawn.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <tf_conversions/tf_eigen.h>
#include <visualization_msgs/Marker.h>
/*********************** LIBRARIES **********************/
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <cmath>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
/*********************** ROSBAG *************************/
#include <openpose_ros_msgs/OpenPoseHumanList.h>
#include <openpose_ros_msgs/PointWithProb.h>
#include <openpose_ros_msgs/OpenPoseHuman.h>

#include <iostream>
#include <sstream>
#include <fstream>

using namespace std;
using namespace cv;
using namespace ceres;

struct KeyPoint_prob{
	double x;
	double y;
	double p;
};

class ImageProcessor
{
public:
	ImageProcessor(int camera_index, string camera_type):it(nh),size_color(640,360)
	{
		cout << "回调函数开始" << endl;
		camera_index_ = camera_index;

		char color_topic[50];
		string image_topic = "/qhd/image_color_rect";
		camera_type += image_topic;
		//sprintf(color_topic,"%s%s",camera_topic,image_topic);
		image_sub =it.subscribe(camera_type,1,&ImageProcessor::imageCallback,this);

		char key_points_topic[50];
		sprintf(key_points_topic,"openpose_ros/human_list_%d",camera_index);
		human_keypoints_sub = nh.subscribe(key_points_topic,1,&ImageProcessor::human_keypoints_callback,this);

		initCalibration(0.333,camera_index_);
		image = Mat(size_color,CV_8UC3);

		cout << "回调函数结束" << endl;
	}

	void getMarkerCenter(vector<Point2f>& marker_center)
	{
		marker_center = this->marker_center;
	}

	void getCameraPose(Eigen::Matrix3d & camera_rot,Eigen::Matrix<double,3,1>& camera_trans)
	{
		camera_rot = this->camera_rot;
		camera_trans = this->camera_trans;
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
	void initCalibration(double scale,int camera_index);
	void ComputeKinectPose(Eigen::Matrix3d &Cam_rot, Eigen::Matrix<double,3,1> &Cam_trans);

/************************************* Marker function *************************************/

	void getMarker(cv::Mat &marker_image,vector<Point2f>& marker_center, bool Key);
	void getMarkerCoordinate(vector<vector<Point2f>>& corners,vector<int>& ids,vector<Point2f>& marker_center_);

/************************************* human_track *****************************************/

	void human_keypoints_callback(openpose_ros_msgs::OpenPoseHumanList keypoints);
	void draw_human_pose(Mat& image,const vector<KeyPoint_prob>& human_joints, Scalar color);


private:
	int camera_index_;

	ros::NodeHandle nh;
	image_transport::ImageTransport it;
    image_transport::Subscriber image_sub;
	ros::Subscriber human_keypoints_sub;

	tf::TransformListener listener;

	string camera_type;
	cv::Mat distCoeffs;
	cv::Mat cameraMatrix;

	cv::Mat image;
	cv::Mat image_origin;
	cv::Size size_color;

	cv::Ptr<aruco::Dictionary> dictionary;

	Eigen::Matrix3d camera_rot;
	Eigen::Matrix<double,3,1> camera_trans;
	Eigen::Matrix3d kinect_rot;
	Eigen::Matrix<double,3,1> kinect_trans;

	vector<Point2f> marker_center;
	Eigen::Matrix<double,3,1> worldPoint;

	vector<std::string> human_joint_names;
	vector<KeyPoint_prob> key_points_basic;

};


void ImageProcessor::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
	try
	{
		cv::Mat color_mat = cv_bridge::toCvShare(msg,"bgr8")->image;
		image = color_mat.clone();

		ComputeKinectPose(kinect_rot,kinect_trans);
		if(!key_points_basic.empty()){
		draw_human_pose(image,key_points_basic,cv::Scalar(255,0,0));
		}

		getMarker(image,marker_center,0);
		//cout << "marker_center:" << marker_center << endl;
		char camera_ID[20];
		sprintf(camera_ID,"camera_%d",camera_index_);
		cv::imshow(camera_ID,image);
		cv::waitKey(3);

	}
	catch(cv_bridge::Exception& e)
	{
		ROS_ERROR("Could not convert from '%s' to 'bgr8'.",msg->encoding.c_str());
		return;
	}
}


void ImageProcessor::loadCalibrationFiles(std::string& calib_path, cv::Mat& cameraMatrix, cv::Mat& distCoeffs, double scale)
{
	cv::FileStorage fs;

	cv::Mat cameraMatrix_origin;

	calib_path = calib_path + "/calib_color.yaml";
	std::cout << calib_path << std::endl;
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


void ImageProcessor::initCalibration(double scale,int camera_index)
{
	string calib_path = "/home/xuchengjun/catkin_ws/src/cv_camera/kinect_calibration_data";
	loadCalibrationFiles(calib_path,cameraMatrix,distCoeffs,scale);
}


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


void ImageProcessor::getMarker(cv::Mat &marker_image,vector<Point2f>& marker_center, bool Key)
{
	dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
	std::vector<int> ids;
	std::vector<std::vector<cv::Point2f> > corners;
	std::vector<cv::Vec3d> rvecs,tvecs;

	if(!marker_image.empty())
	{
		cv::aruco::detectMarkers(marker_image,dictionary,corners,ids);
		if(ids.size() > 0)
		{
			cv::aruco::drawDetectedMarkers(marker_image,corners,ids);
			for(int i=0;i<ids.size();i++)
			{
				cv::aruco::estimatePoseSingleMarkers(corners,0.155,cameraMatrix,distCoeffs,rvecs,tvecs);
				if(!rvecs.empty() && !tvecs.empty() && ids[i] == 0)
				{
					cv::aruco::drawAxis(marker_image,cameraMatrix,distCoeffs,rvecs[i],tvecs[i],0.1);
					getMarkerCoordinate(corners,ids,marker_center);

				}
				char num[10];
				sprintf(num,"camera_%d",camera_index_);
				if(Key == 1)
				{
				sendMarkerTf(tvecs,rvecs,ids,num,camera_rot,camera_trans);
				}
			}
		}
	}

}


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

/*
void ImageProcessor::sendWorldTf(const Eigen::Matrix<double, 3, 1>& point,const int camera_id, const string& camera_name)
{
    static tf::TransformBroadcaster world_position;

    tf::Vector3 tf_trans(point(0,0), point(1,0), point(2,0));
	tf::Quaternion q(0,0,0,1);
    tf::Transform transform(q, tf_trans);
    char marker_name[20];
	sprintf(marker_name, "point_under_cam_%d", camera_id);
    world_position.sendTransform(tf::StampedTransform(transform, ros::Time::now(),camera_name.c_str(), marker_name));
}
*/


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


void ImageProcessor::ComputeKinectPose(Eigen::Matrix3d &Cam_rot, Eigen::Matrix<double,3,1> &Cam_trans)
{
	tf::StampedTransform transform;
	char kinect_id[10];
	sprintf(kinect_id,"camera_base_%d",camera_index_);
 	try
    {
        listener.waitForTransform("/marker_0",kinect_id,ros::Time(0),ros::Duration(3.0));
        listener.lookupTransform("/marker_0",kinect_id,ros::Time(0),transform);
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



void ImageProcessor::human_keypoints_callback(openpose_ros_msgs::OpenPoseHumanList keypoints)
{
	key_points_basic.clear();
	int person_num = keypoints.num_humans;
	vector<vector<double> > distance_pool;
	distance_pool.resize(person_num);
	if(person_num > 0)
	{
		int person_index = 0;
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
			KeyPoint_prob key_point_ite;
			//head
			key_point_ite.x = body_keypoints[0].x;
			key_point_ite.y = body_keypoints[0].y;
			key_point_ite.p = body_keypoints[0].prob;
			key_points_basic.push_back(key_point_ite);
			//
			key_point_ite.x = body_keypoints[1].x;
			key_point_ite.y = body_keypoints[1].y;
			key_point_ite.p = body_keypoints[1].prob;
			key_points_basic.push_back(key_point_ite);
			//rWrist
			key_point_ite.x = body_keypoints[2].x;
			key_point_ite.y = body_keypoints[2].y;
			key_point_ite.p = body_keypoints[2].prob;
			key_points_basic.push_back(key_point_ite);
			//rElbow
			key_point_ite.x = body_keypoints[3].x;
			key_point_ite.y = body_keypoints[3].y;
			key_point_ite.p = body_keypoints[3].prob;
			key_points_basic.push_back(key_point_ite);
			//rArm
			key_point_ite.x = body_keypoints[4].x;
			key_point_ite.y = body_keypoints[4].y;
			key_point_ite.p = body_keypoints[4].prob;
			key_points_basic.push_back(key_point_ite);
			//hip
			key_point_ite.x = body_keypoints[5].x;
			key_point_ite.y = body_keypoints[5].y;
			key_point_ite.p = body_keypoints[5].prob;
			key_points_basic.push_back(key_point_ite);
			//lArm
			key_point_ite.x = body_keypoints[6].x;
			key_point_ite.y = body_keypoints[6].y;
			key_point_ite.p = body_keypoints[6].prob;
			key_points_basic.push_back(key_point_ite);
			//lElbow
			key_point_ite.x = body_keypoints[7].x;
			key_point_ite.y = body_keypoints[7].y;
			key_point_ite.p = body_keypoints[7].prob;
			key_points_basic.push_back(key_point_ite);
			//lWrist
			key_point_ite.x = body_keypoints[8].x;
			key_point_ite.y = body_keypoints[8].y;
			key_point_ite.p = body_keypoints[8].prob;
			key_points_basic.push_back(key_point_ite);
			//
			key_point_ite.x = body_keypoints[9].x;
			key_point_ite.y = body_keypoints[9].y;
			key_point_ite.p = body_keypoints[9].prob;
			key_points_basic.push_back(key_point_ite);
			//
			key_point_ite.x = body_keypoints[10].x;
			key_point_ite.y = body_keypoints[10].y;
			key_point_ite.p = body_keypoints[10].prob;
			key_points_basic.push_back(key_point_ite);
			//
			key_point_ite.x = body_keypoints[11].x;
			key_point_ite.y = body_keypoints[11].y;
			key_point_ite.p = body_keypoints[11].prob;
			key_points_basic.push_back(key_point_ite);
			//
			key_point_ite.x = body_keypoints[12].x;
			key_point_ite.y = body_keypoints[12].y;
			key_point_ite.p = body_keypoints[12].prob;
			key_points_basic.push_back(key_point_ite);
			//
			key_point_ite.x = body_keypoints[13].x;
			key_point_ite.y = body_keypoints[13].y;
			key_point_ite.p = body_keypoints[13].prob;
			key_points_basic.push_back(key_point_ite);
			//
			key_point_ite.x = body_keypoints[14].x;
			key_point_ite.y = body_keypoints[14].y;
			key_point_ite.p = body_keypoints[14].prob;
			key_points_basic.push_back(key_point_ite);
		}
	}
}

//////可能会造成Segmation fault ////////
void ImageProcessor::draw_human_pose(Mat& image,const vector<KeyPoint_prob>& human_joints, Scalar color)
{
	if(human_joints[0].p < 0.01)
	{
		circle(image, cv::Point(human_joints[1].x,human_joints[1].y), 3, color, -1, 8);
		circle(image, cv::Point(human_joints[4].x,human_joints[2].y), 3, color, -1, 8);
		for(int i=2; i < human_joints.size(); i++)
		{
			if(human_joints[i].p < 0.01)
				continue;
			circle(image, cv::Point(human_joints[i].x, human_joints[i].y), 3, color, -1, 8);
			if(i != 4 && i != 0 && i !=7 )
				line(image, cv::Point(human_joints[i-1].x, human_joints[i-1].y), cv::Point(human_joints[i].x, human_joints[i].y), color, 2);
			else
				continue;
		}
	}
	else{
		circle(image, cv::Point(human_joints[0].x, human_joints[0].y), 3, color, -1, 8);
		for(int i=1; i < human_joints.size(); i++)
		{
			if(human_joints[i].p < 0.01)
				continue;
			circle(image, cv::Point(human_joints[i].x, human_joints[i].y), 3, color, -1, 8);
			if(i != 7 && i != 0)
				line(image, cv::Point(human_joints[i-1].x, human_joints[i-1].y), cv::Point(human_joints[i].x, human_joints[i].y), color, 2);
		}
	}
	//ROS_INFO("Human pose drawed.");
}


struct CostFunction_cam {
  	CostFunction_cam(Point3d _observe_point1, Point3d _observe_point2, Point3d _observe_point3,
	  				  Eigen::Matrix3d _camera_rot_1, Eigen::Matrix3d _camera_rot_2, Eigen::Matrix3d _camera_rot_3,
					  Eigen::Matrix<double,3,1> _camera_trans_1,Eigen::Matrix<double,3,1> _camera_trans_2, Eigen::Matrix<double,3,1> _camera_trans_3)
	  :observe_point1(_observe_point1),observe_point2(_observe_point2),observe_point3(_observe_point3),
	   camera_rot_1(_camera_rot_1), camera_rot_2(_camera_rot_2),camera_rot_3(_camera_rot_3),
	   camera_trans_1(_camera_trans_1), camera_trans_2(_camera_trans_2), camera_trans_3(_camera_trans_3){}

// 残差的计算
  	template <typename T>
	  bool operator()(const T *const depths, T *residual) const {
		const T cx = T(959.19/2), cy = T(578.16/2), fx = T(1061.45/2), fy = T(1056.70/2);
		const T ccx = T(983.13/2), ccy = T(667.25/2), cfx = T(804.20/2), cfy = T(804.12/2);

		Eigen::Matrix<T,3,1> Cam_point1;
		Eigen::Matrix<T,3,1> Cam_point2;
		Eigen::Matrix<T,3,1> Cam_point3;
		Eigen::Matrix<T,3,1> World_point1;
		Eigen::Matrix<T,3,1> World_point2;
		Eigen::Matrix<T,3,1> World_point3;
		T confidence_1, confidence_2, confidence_3;
		confidence_1 = T(observe_point1.z) * T(observe_point2.z);
		confidence_2 = T(observe_point2.z) * T(observe_point3.z);
		confidence_3 = T(observe_point1.z) * T(observe_point3.z);

		Cam_point1(2,0) = depths[0];
		Cam_point2(2,0) = depths[1];
		Cam_point3(2,0) = depths[2];
		Cam_point1(0,0) = (T(observe_point1.x) - cx) * Cam_point1(2,0) / fx;
		Cam_point1(1,0) = (T(observe_point1.y) - cy) * Cam_point1(2,0) / fy;
		Cam_point2(0,0) = (T(observe_point2.x) - cx) * Cam_point2(2,0) / fx;
		Cam_point2(1,0) = (T(observe_point2.y) - cy) * Cam_point2(2,0) / fy;
		Cam_point3(0,0) = (T(observe_point3.x) - ccx) * Cam_point3(2,0) / cfx;
		Cam_point3(1,0) = (T(observe_point3.y) - ccy) * Cam_point3(2,0) / cfy;
		World_point1 = camera_rot_1.cast<T>() * Cam_point1 + camera_trans_1.cast<T>();
		World_point2 = camera_rot_2.cast<T>() * Cam_point2 + camera_trans_2.cast<T>();
		World_point3 = camera_rot_3.cast<T>() * Cam_point3 + camera_trans_3.cast<T>();

    	residual[0] = T(1.0)/confidence_1
						*((World_point1(0,0) - World_point2(0,0)) * (World_point1(0,0) - World_point2(0,0))
						+(World_point1(1,0) - World_point2(1,0)) * (World_point1(1,0) - World_point2(1,0))
						+(World_point1(2,0) - World_point2(2,0)) * (World_point1(2,0) - World_point2(2,0)));

    	residual[1] = T(1.0)/confidence_2
						*((World_point1(0,0) - World_point3(0,0)) * (World_point1(0,0) - World_point3(0,0))
						+(World_point1(1,0) - World_point3(1,0)) * (World_point1(1,0) - World_point3(1,0))
						+(World_point1(2,0) - World_point3(2,0)) * (World_point1(2,0) - World_point3(2,0)));

    	residual[2] = T(1.0)/confidence_3
						*((World_point2(0,0) - World_point3(0,0)) * (World_point2(0,0) - World_point3(0,0))
						+(World_point2(1,0) - World_point3(1,0)) * (World_point2(1,0) - World_point3(1,0))
						+(World_point2(2,0) - World_point3(2,0)) * (World_point2(2,0) - World_point3(2,0)));

    	return true;

  	}

	Point3d observe_point1;
	Point3d observe_point2;
	Point3d observe_point3;
	Eigen::Matrix3d camera_rot_1;
	Eigen::Matrix3d camera_rot_2;
	Eigen::Matrix3d camera_rot_3;
	Eigen::Matrix<double,3,1> camera_trans_1;
	Eigen::Matrix<double,3,1> camera_trans_2;
	Eigen::Matrix<double,3,1> camera_trans_3;

};


Point3d solve_point_av(Point3d& point_1, Point3d& point_2, Point3d& point_3)
{
	Point3d point_av;
	point_av.x = (point_1.x + point_2.x + point_3.x) / 3;
	point_av.y = (point_1.y + point_2.y + point_3.y) / 3;
	point_av.z = (point_1.z + point_2.z + point_3.z) / 3;

	return point_av;

}


void point3d_To_matrix(Point3d& point, Eigen::Matrix<double,3,1>& matrix)
{
	matrix(0,0) = point.x;
	matrix(1,0) = point.y;
	matrix(2,0) = point.z;
}

void matrix_to_point3d(Eigen::Matrix<double,3,1>& matrix, Point3d& point)
{
	point.x = matrix(0,0);
	point.y = matrix(1,0);
	point.z = matrix(2,0);
}


int main(int argc,char **argv)
{
	ros::init(argc,argv,"marker_node");
	ros::NodeHandle n;
	tf::TransformListener listener;
	ros::Publisher pose_pub = n.advertise<visualization_msgs::Marker>("visualization_marker",1);

	ImageProcessor imageprocess_1(1,"kinect2_1");
	ImageProcessor imageprocess_2(2,"kinect2_2");
	ImageProcessor imageprocess_3(3,"camera_3");

	ros::Rate loop_rate(30);

	int N = 15; //数据点
	vector<KeyPoint_prob> key_point1, key_point2, key_point3;
	Eigen::Matrix3d cam_rot_1, cam_rot_2, cam_rot_3;
	Eigen::Matrix<double,3,1> cam_trans_1, cam_trans_2, cam_trans_3;
	vector<Point3d> p1_data(N), p2_data(N), p3_data(N);

	//配置求解器
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	options.minimizer_progress_to_stdout = true;

  	double depths[N][3] = {}; //待优化值
	double cx = 959.19/2, cy = 578.16/2, fx = 1061.45/2, fy = 1056.70/2;
	double ccx = 983.13/2, ccy = 667.25/2, cfx = 804.20/2, cfy = 804.12/2;

	while(ros::ok()){
		//imageprocess_1.getMarkerCenter(key_point1);
		//imageprocess_2.getMarkerCenter(key_point2);

		imageprocess_1.getKinectPose(cam_rot_1,cam_trans_1);
		imageprocess_2.getKinectPose(cam_rot_2,cam_trans_2);
		imageprocess_3.getKinectPose(cam_rot_3,cam_trans_3);

		imageprocess_1.getHumanJoint(key_point1);
		imageprocess_2.getHumanJoint(key_point2);
		imageprocess_3.getHumanJoint(key_point3);
		/*
		imageprocess_1.getCameraPose(cam_rot_1,cam_trans_1);
		imageprocess_2.getCameraPose(cam_rot_2,cam_trans_2);
		*/

		if(!key_point1.empty() && !key_point2.empty())
		{
			for(int i=0;i<N;i++)
			{
				p1_data[i].x = key_point1[i].x;
				p1_data[i].y = key_point1[i].y;
				p1_data[i].z = key_point1[i].p;
				p2_data[i].x = key_point2[i].x;
				p2_data[i].y = key_point2[i].y;
				p2_data[i].z = key_point2[i].p;
				p3_data[i].x = key_point3[i].x;
				p3_data[i].y = key_point3[i].y;
				p3_data[i].z = key_point3[i].p;
			}
		}

		ceres::Problem problem;
		for(int i=0;i < N;i++)
		{
    		problem.AddResidualBlock(     // 向问题中添加误差项
    	  	// 使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
      		new ceres::AutoDiffCostFunction<CostFunction_cam, 3, 3>(
        		new CostFunction_cam(p1_data[i], p2_data[i],p3_data[i],cam_rot_1,cam_rot_2,cam_rot_3,cam_trans_1,cam_trans_2,cam_trans_3)
      		),
      		nullptr,            // 核函数，这里不使用，为空
      		depths[i]                // 待估计参数
    		);
		}
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		//cout << summary.BriefReport() << endl;

		visualization_msgs::Marker points,line_strip;
		points.header.frame_id = line_strip.header.frame_id = "marker_0";
		points.header.stamp = line_strip.header.stamp = ros::Time::now();
		points.ns = line_strip.ns = "marker_node";
		points.action = line_strip.action = visualization_msgs::Marker::ADD;
		points.pose.orientation.w = line_strip.pose.orientation.w = 1.0;
		points.pose.orientation.x = line_strip.pose.orientation.x = 0.0;
		points.pose.orientation.y = line_strip.pose.orientation.y = 0.0;
		points.pose.orientation.z = line_strip.pose.orientation.z = 0.0;

		points.id = 0;
		line_strip.id = 1;

		points.type = visualization_msgs::Marker::POINTS;
		line_strip.type = visualization_msgs::Marker::LINE_STRIP;

		points.scale.x = 0.03;
		points.scale.y = 0.03;
		points.scale.z = 0.03;
		line_strip.scale.x = 0.02;

		points.color.r = 1.0;
		points.color.a = 1.0;
		line_strip.color.g = 1.0;
		line_strip.color.a = 1.0;

		points.lifetime = line_strip.lifetime = ros::Duration(1);

		vector<Point3d> Cam_Point1(N), Cam_Point2(N), Cam_Point3(N);
		vector<Eigen::Matrix<double,3,1> > Wor_Point1(N),Wor_Point2(N),Wor_Point3(N);

		for(int p=0;p<N;p++)
		{
			Cam_Point1[p].z = depths[p][0];
			Cam_Point2[p].z = depths[p][1];
			Cam_Point3[p].z = depths[p][2];
			Cam_Point1[p].x = (p1_data[p].x - cx) * Cam_Point1[p].z / fx;
			Cam_Point1[p].y = (p1_data[p].y - cy) * Cam_Point1[p].z / fy;
			Cam_Point2[p].x = (p2_data[p].x - cx) * Cam_Point2[p].z / fx;
			Cam_Point2[p].y = (p2_data[p].y - cy) * Cam_Point2[p].z / fy;
			Cam_Point3[p].x = (p3_data[p].x -ccx) * Cam_Point3[p].z / cfx;
			Cam_Point3[p].y = (p3_data[p].y -ccy) * Cam_Point3[p].z / cfy;

			point3d_To_matrix(Cam_Point1[p],Wor_Point1[p]);
			point3d_To_matrix(Cam_Point2[p],Wor_Point2[p]);
			point3d_To_matrix(Cam_Point3[p],Wor_Point3[p]);
			Wor_Point1[p] = cam_rot_1.cast<double>() * Wor_Point1[p] + cam_trans_1.cast<double>();
			Wor_Point2[p] = cam_rot_2.cast<double>() * Wor_Point2[p] + cam_trans_2.cast<double>();
			Wor_Point3[p] = cam_rot_3.cast<double>() * Wor_Point3[p] + cam_trans_3.cast<double>();
			matrix_to_point3d(Wor_Point1[p],Cam_Point1[p]);
			matrix_to_point3d(Wor_Point2[p],Cam_Point2[p]);
			matrix_to_point3d(Wor_Point3[p],Cam_Point3[p]);

			Point3d point_av = solve_point_av(Cam_Point1[p],Cam_Point2[p],Cam_Point3[p]);
			geometry_msgs::Point marker_point;
			marker_point.x = point_av.x;
			marker_point.y = point_av.y;
			marker_point.z = point_av.z;
			points.points.push_back(marker_point);
			line_strip.points.push_back(marker_point);
		}
		pose_pub.publish(points);
		pose_pub.publish(line_strip);

    	ros::spinOnce();
		loop_rate.sleep();
	}

	return 0;
}
