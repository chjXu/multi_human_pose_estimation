#ifndef POSE_HPP
#define POSE_HPP

using namespace std;
#include <ros/ros.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

#include <openpose_ros_msgs/OpenPoseHumanList.h>
#include <openpose_ros_msgs/PointWithProb.h>
#include <openpose_ros_msgs/OpenPoseHuman.h>
#include <openpose_ros_msgs/BoundingBox.h>

struct KeyPoint_prob
{
    float x;
    float y;
    float p;
};

struct Bounding_boxs
{
    float x;
    float y;
    float width;
    float height;
};

struct KeyPoint_3d
{
    bool available;
    int index = 0;
    float x;
    float y;
    float z;
};

class Pose
{
private:


public:
    int label;
    int camera_index;
    int num_as;
    vector<int> as_id;
    vector<int> as;

    vector<KeyPoint_prob> pose_joints;
    Bounding_boxs bounding_box;
    vector<KeyPoint_3d> pose_joints_3d;
    cv::Scalar color;

	Pose(){
		pose_joints.resize(17);
        pose_joints_3d.resize(17);
        for(int i = 0; i < 17; ++i) {
            pose_joints_3d[i].available = false;
        }
        num_as = 1;
        as.resize(1);
        as_id.resize(1);
	}
	~Pose(){}
	void setPose(openpose_ros_msgs::OpenPoseHumanList& keypoints, int id);
    Bounding_boxs setBoundingBox(const vector<KeyPoint_prob> &pose, int firstIndex, int lastIndex, int threshold);
	void getPose(vector<KeyPoint_prob> &pose_joints);
    bool setCenterPoint();


	void setCameraId(const int camera_index){
        this->camera_index = camera_index;
    }
    int getLabel() const{
        return label;
    }
    void setLabel(int new_label){
        label = new_label;
    }
    int getCameraId() const{
        return camera_index;
    }
    void setColor(const cv::Scalar& color){
        this->color = color;
    }
};

void Pose::setPose(openpose_ros_msgs::OpenPoseHumanList& keypoints, int id)
{
    for(int i = 0; i < 17; ++i) {
        pose_joints[i].x = keypoints.human_list[id].body_key_points_with_prob[i].x;
        pose_joints[i].y = keypoints.human_list[id].body_key_points_with_prob[i].y;
        pose_joints[i].p = keypoints.human_list[id].body_key_points_with_prob[i].prob;
        //ROS_INFO("cam:%d, p%d: x:%.2f y:%.2f p:%.2f", camera_index, i, pose_joints[i].x, pose_joints[i].y, pose_joints[i].p);
    }
    
	bounding_box = setBoundingBox(pose_joints,0,-1, 0.5);
}

Bounding_boxs Pose::setBoundingBox(const vector<KeyPoint_prob> &pose, int firstIndex, int lastIndex, int threshold)
{
	if(pose.empty())
		return {};

    const auto numberKeypoints = pose.size();
    const auto lastIndexClean = (lastIndex < 0 ? numberKeypoints : lastIndex);

    if (numberKeypoints < 1)
        ROS_ERROR("Number body parts must be > 0.");
    if (lastIndexClean > numberKeypoints)
        ROS_ERROR("The value of `lastIndex` must be less or equal than `numberKeypoints`. Currently: ");
    if (firstIndex > lastIndexClean)
        ROS_ERROR("The value of `firstIndex` must be less or equal than `lastIndex`. Currently: ");

    Bounding_boxs bbox;
	double minX = std::numeric_limits<double>::max();
	double maxX = std::numeric_limits<double>::lowest();
	double minY = minX;
	double maxY = maxX;

	for(int i=firstIndex; i<numberKeypoints; ++i){
		const auto score = pose[i].p;
		if(score > threshold){
			const auto x = pose[i].x;
			const auto y = pose[i].y;

			//set x
			if(maxX < x)
				maxX = x;
			if(minX > x)
				minX = x;

			//set Y
			if(maxY < y)
				maxY = y;
			if(minY > y)
				minY = y;			
		}
	}

	if(maxX >= minX && maxY >= minY){
		// assert(minX >= 0 && minX <= image.cols);
		// assert(maxX >= 0 && maxX <= image.cols);
		// assert(minY >= 0 && minY <= image.rows);
		// assert(maxY >= 0 && maxY <= image.rows);
		bbox.x = minX - 20;
		bbox.y = minY - 20;
		bbox.width = maxX + 20;
		bbox.height = maxY + 20;
	}
	return bbox;
}


#endif
