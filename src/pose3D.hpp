#pragma once

#include "pose.hpp"
#include "association.hpp"
#include "SkeletonPair.hpp"
#include "humanModel.hpp"
#include <list>
#include <cmath>
#include <algorithm>
#include <queue>
#include <visualization_msgs/Marker.h>
#include <stdexcept>

using namespace std;

class Track3DPose
{
private:
    vector<vector<KeyPoint_3d>> framePoses;
    vector<double> correspondence;
    vector<Scalar> colors;
    vector<Eigen::Matrix3d> camera_rot;
    vector<Eigen::Matrix<double, 3, 1>> camera_trans;
    ros::Publisher *pose_pub;
    int t=0;

    pthread_mutex_t track_mutex;
public:
    vector<cv::Mat> image;
    vector<cv::Mat> CameraMatrix;
    vector<Pose> poses;
    vector<vector<int>> pose_ass;

    //Skeleton availableSkeletons;
    //Skeleton skeleton;
    queue<Skeleton> queue_availableSkeletons;
    list<SkeletonPair> skeleton_pair;

    // Track3DPose() {
    //     image.resize(3);
    //     colors.resize(5);
    //     colors[0] = Scalar(0,0,255);
    //     colors[1] = Scalar(255,255,0);
    //     colors[2] = Scalar(0,255,0);
    //     colors[3] = Scalar(255,0,0);
    //     colors[4] = Scalar(255,0,255);
    // }

    Track3DPose(ros::Publisher& pub) : hm("lhz",1), hm1("xcj",2){
        pose_pub = &pub;
        image.resize(3);
        colors.resize(5);
        colors[0] = Scalar(0,0,255);
        colors[1] = Scalar(255,255,0);
        colors[2] = Scalar(0,255,0);
        colors[3] = Scalar(255,0,0);
        colors[4] = Scalar(255,0,255);

        track_mutex = PTHREAD_MUTEX_INITIALIZER;
    }
    HumanModel hm;
    HumanModel hm1;

    void publish_3d_poses(Skeleton &sk_tracking);

    void getCameraMatrix(cv::Mat&);

    void get3Dskeletons(vector<vector<KeyPoint_3d>> unsort_poses );

    cv::Point2d getImageCoordinate(KeyPoint_3d&, int);

    void draw_human_pose(Mat& image, vector<KeyPoint_prob>& pose_2d, Scalar color);

    void verify2DPoseAss();

    void addCamPose(Eigen::Matrix3d& cam_rot, Eigen::Matrix<double, 3, 1>& cam_trans);

    bool getSkeletonsCorrespondence(SkeletonPair &skeleton_pair, double threshold = 0.4);

    double getDistance(cv::Point3d skeleton_point_1, cv::Point3d skeleton_point_2);

    void getAvailableSkeleton(vector<vector<KeyPoint_3d>> pre_skeletons);

    void generateSkeletonPair();

    void skeletonAss();

    void setframePose(vector<frame_skeleton> &sk_tracking);

    void verifyTracking(Skeleton &);
};

void Track3DPose::addCamPose(Eigen::Matrix3d& cam_rot, Eigen::Matrix<double, 3, 1>& cam_trans) {
    camera_rot.push_back(cam_rot.transpose());
    camera_trans.push_back(cam_trans);
}

void Track3DPose::getCameraMatrix(cv::Mat& cameraMatrix)
{
    CameraMatrix.push_back(cameraMatrix);
}

void addLine(visualization_msgs::Marker& line_list, vector<KeyPoint_3d>& framePose, int a, int b) {
    if(framePose[a].available && framePose[b].available) {
        geometry_msgs::Point p_a,p_b;
        p_a.x = framePose[a].x;
        p_a.y = framePose[a].y;
        p_a.z = framePose[a].z;

        p_b.x = framePose[b].x;
        p_b.y = framePose[b].y;
        p_b.z = framePose[b].z;

        line_list.points.push_back(p_a);
        line_list.points.push_back(p_b);
    }
}


void Track3DPose::publish_3d_poses(Skeleton &sk_tracking) {
    for(int i = 0; i < sk_tracking.skeleton_3d.size(); ++i) {
        visualization_msgs::Marker line_list;
        line_list.id = i;
        line_list.type = visualization_msgs::Marker::LINE_LIST;

        line_list.header.frame_id = "/marker_0";
        line_list.header.stamp = ros::Time::now();
        line_list.ns = "humans";
        line_list.action = visualization_msgs::Marker::ADD;
        line_list.pose.orientation.w = 1.0;

        line_list.scale.x = 0.01;
        line_list.scale.y = 0.01;
        line_list.color.r = 1.0;
        line_list.color.a = 1.0;

        addLine(line_list, sk_tracking.skeleton_3d[i].a_skeleton, 2, 3);
        addLine(line_list, sk_tracking.skeleton_3d[i].a_skeleton, 4, 3);
        addLine(line_list, sk_tracking.skeleton_3d[i].a_skeleton, 2, 5);
        addLine(line_list, sk_tracking.skeleton_3d[i].a_skeleton, 6, 5);
        addLine(line_list, sk_tracking.skeleton_3d[i].a_skeleton, 7, 6);
        addLine(line_list, sk_tracking.skeleton_3d[i].a_skeleton, 2, 8);
        addLine(line_list, sk_tracking.skeleton_3d[i].a_skeleton, 5, 8);

        pose_pub->publish(line_list);
    }
}

void drawLine(Mat& image, vector<cv::Point2d>& current_frame_2d, int a, int b, Scalar color) {
    if( (current_frame_2d[a].x == 0.0 && current_frame_2d[a].y == 0.0) ||
        (current_frame_2d[b].x == 0.0 && current_frame_2d[b].y == 0.0) ) {
        //ROS_INFO("no line");
        return;
    }
    else {
        //ROS_INFO("line, %.1f,%.1f : %.1f,%.1f", current_frame_2d[a].x, current_frame_2d[a].y, current_frame_2d[b].x, current_frame_2d[b].y);
        line(image, cv::Point(current_frame_2d[a].x, current_frame_2d[a].y), cv::Point(current_frame_2d[b].x, current_frame_2d[b].y), color, 2);
    }
}

void Track3DPose::verify2DPoseAss() {
    if (pose_ass.empty() || poses.empty() || framePoses.empty()) {
        return;
    }
    for(int i = 0; i < 3; ++i) {
        if(image[i].empty()){
            return;
        }
    }
    for(int i = 0; i < pose_ass.size(); ++i) {
        for(int j = 0; j < pose_ass[i].size(); ++j) {
            int index = pose_ass[i][j];
            char text[10];
            sprintf(text, "id:%d", pose_ass[i][j]);
            draw_human_pose(image[poses[index].camera_index - 1], poses[index].pose_joints, colors[i]);
            putText(image[poses[index].camera_index - 1], text, Point(poses[index].pose_joints[2].x - 40,poses[index].pose_joints[0].y + 30), FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2);
        }
    }
    if(!framePoses.empty()) {
        //publish_3d_poses();
        for(int i = 0; i < framePoses.size(); ++i) {
            for(int c = 0; c < 3; ++c) {
                vector<cv::Point2d> current_frame_2d;
                for(int j=0; j<9; j++)
                {
                    current_frame_2d.push_back(getImageCoordinate(framePoses[i][j], c));
                }
                for(int j=0; j<9; j++){
                    circle(image[c], cv::Point(current_frame_2d[j].x, current_frame_2d[j].y),2, colors[i], 2);
                }
                //line(image[0], cv::Point(current_frame_2d[0].x, current_frame_2d[0].y), cv::Point(current_frame_2d[1].x, current_frame_2d[1].y), color, 2);
                drawLine(image[c], current_frame_2d, 2, 3, colors[i]);
                drawLine(image[c], current_frame_2d, 4, 3, colors[i]);
                drawLine(image[c], current_frame_2d, 2, 5, colors[i]);
                drawLine(image[c], current_frame_2d, 6, 5, colors[i]);
                drawLine(image[c], current_frame_2d, 6, 7, colors[i]);
                drawLine(image[c], current_frame_2d, 2, 8, colors[i]);
                drawLine(image[c], current_frame_2d, 5, 8, colors[i]);
            }
        }
    }


    //num++;
    //string image_path_1 = "/home/agent/luk_ws/read_bag/camera_1/"+to_string(num)+ ".jpg";
    //string image_path_2 = "/home/agent/luk_ws/read_bag/camera_2/"+to_string(num)+ ".jpg";
    //string image_path_3 = "/home/agent/luk_ws/read_bag/camera_3/"+to_string(num)+ ".jpg";

    //cv::imwrite(image_path_1,image[0]);
    cv::imshow("1",image[0]);
    //cv::waitKey(3);
    //cv::imwrite(image_path_2,image[1]);
    cv::imshow("2",image[1]);
    //cv::waitKey(3);
    //cv::imwrite(image_path_3,image[2]);
    cv::imshow("3",image[2]);
    cv::waitKey(3);
    //t++;
    //if(t>5){
    //    t=0;
    //}

    // while(ros::ok()) {
    //     char c = cv::waitKey(30);
    //     if(c == ' '){
    //         break;
    //     }
    //
    // }
}

void Track3DPose::verifyTracking(Skeleton &sk_tracking) {
    if(sk_tracking.skeleton_3d.empty()){
        return;
    }

    for(int i = 0; i < 3; ++i) {
        if(image[i].empty()){
            return;
        }
    }

    if(!sk_tracking.skeleton_3d.empty()) {
        publish_3d_poses(sk_tracking);
        for(int i=0; i<sk_tracking.skeleton_3d.size(); i++){
            if(sk_tracking.skeleton_3d[i].label == 0 || sk_tracking.skeleton_3d[i].label == 2){
                hm.getPose(sk_tracking.skeleton_3d[i]);
            }else if(sk_tracking.skeleton_3d[i].label == 1 || sk_tracking.skeleton_3d[i].label == 3){
                hm1.getPose(sk_tracking.skeleton_3d[i]);
            }
        }
        // hm.getPose(sk_tracking.skeleton_3d[0]);
        // cout << "lhz的label: " << sk_tracking.skeleton_3d[0].label << endl;
        // if(sk_tracking.skeleton_3d.size() >=2)
        //     hm1.getPose(sk_tracking.skeleton_3d[1]);
        // hm.getPose(sk_tracking);
        // hm.setJointAndName();
        //cout << sk_tracking.skeleton_3d[0].a_skeleton.size() << endl;
        for(int i = 0; i < sk_tracking.skeleton_3d.size(); ++i) {
            int color_index = sk_tracking.skeleton_3d[i].label;
            for(int c = 0; c < 3; c++) {
                vector<cv::Point2d> current_frame_2d;
                for(int j=0; j<9; j++)
                {
                    current_frame_2d.push_back(getImageCoordinate(sk_tracking.skeleton_3d[i].a_skeleton[j], c));
                }
                for(int j=1; j<9; j++){
                    circle(image[c], cv::Point(current_frame_2d[j].x, current_frame_2d[j].y),2, colors[color_index], 2);
                }
                //line(image[0], cv::Point(current_frame_2d[0].x, current_frame_2d[0].y), cv::Point(current_frame_2d[1].x, current_frame_2d[1].y), color, 2);
                if(sk_tracking.skeleton_3d[i].label != -1){
                    drawLine(image[c], current_frame_2d, 2, 3, colors[color_index]);
                    drawLine(image[c], current_frame_2d, 4, 3, colors[color_index]);
                    drawLine(image[c], current_frame_2d, 2, 5, colors[color_index]);
                    drawLine(image[c], current_frame_2d, 6, 5, colors[color_index]);
                    drawLine(image[c], current_frame_2d, 6, 7, colors[color_index]);
                    drawLine(image[c], current_frame_2d, 2, 8, colors[color_index]);
                    drawLine(image[c], current_frame_2d, 5, 8, colors[color_index]);
                }
            }

        }

    }



    cv::imshow("1",image[0]);
    //cv::waitKey(3);
    cv::imshow("2",image[1]);
    //cv::waitKey(3);
    cv::imshow("3",image[2]);
    cv::waitKey(3);

    // while(ros::ok()) {
    //     char c = cv::waitKey(30);
    //     if(c == ' '){
    //         break;
    //     }

    // }
}


cv::Point2d Track3DPose::getImageCoordinate(KeyPoint_3d& world_cord, int camera)
{

    if(!world_cord.available) {
        return Point2d(0.0, 0.0);
    }
    //ROS_INFO("point ok");
    Eigen::Matrix<double,3,1> World_point;
    Eigen::Matrix<double,3,1> Cam_point;
    World_point(0,0) = world_cord.x;
    World_point(1,0) = world_cord.y;
    World_point(2,0) = world_cord.z;

    Cam_point = camera_rot[camera] * (World_point - camera_trans[camera]);

    cv::Point2d image_cord;
    double fx = CameraMatrix[camera].at<double>(0, 0);
    double fy = CameraMatrix[camera].at<double>(1, 1);
    double cx = CameraMatrix[camera].at<double>(0, 2);
    double cy = CameraMatrix[camera].at<double>(1, 2);

    image_cord.x = (int)(Cam_point(0,0) * fx / Cam_point(2,0) + cx);
    image_cord.y = (int)(Cam_point(1,0) * fy / Cam_point(2,0) + cy);

    return image_cord;
}

void Track3DPose::draw_human_pose(Mat& image, vector<KeyPoint_prob>& pose_2d, Scalar color)
{
    for(int j=0; j<pose_2d.size(); j++){
        if((j >= 2 && j <= 4) || (j >= 9 && j <= 11)) {
        circle(image, cv::Point(pose_2d[j].x, pose_2d[j].y),2, color, 1);
        }
        else{
        circle(image, cv::Point(pose_2d[j].x, pose_2d[j].y),1, color, 2);
        }
    }
}



void Track3DPose::getAvailableSkeleton(vector<vector<KeyPoint_3d>> pre_skeletons)
{
    if(pre_skeletons.empty()){
        return;
    }
    Skeleton availableSkeletons;
    //availableSkeletons.skeleton_3d.clear();
    //framePoses = pre_skeletons;
    //cout <<"输入姿态大小：" << pre_skeletons[0].size() << endl;
    availableSkeletons.skeleton_3d.resize(pre_skeletons.size());
    for(int i=0; i<availableSkeletons.skeleton_3d.size(); i++){
        //availableSkeletons.time_index = t;
        availableSkeletons.skeleton_3d[i].a_skeleton = pre_skeletons[i];
        //availableSkeletons.skeleton_3d[i].index = i;

        // skeleton.time_index = t;
        // skeleton.skeleton_3d = pre_skeletons[i];
        // availableSkeletons.emplace_back(skeleton);
    }
    //cout << availableSkeletons.skeleton_3d.size() << endl;
    // availableSkeletons.time_index = t;
    // availableSkeletons.skeleton_3d = pre_skeletons;
    //t++;
    queue_availableSkeletons.push(availableSkeletons);
    //cout << queue_availableSkeletons.size() << endl;
    if(queue_availableSkeletons.size() == 1){
        for(int i = 0; i < queue_availableSkeletons.front().skeleton_3d.size(); i++){
            queue_availableSkeletons.front().skeleton_3d[i].label = i;
        }

        // for(int i = 0; i < queue_availableSkeletons.front().skeleton_3d.size(); ++i){
        //     cout << "第一帧label：" << queue_availableSkeletons.front().skeleton_3d[i].label << endl;
        // }
    }
    else {
        if(queue_availableSkeletons.size()>1){

            if(queue_availableSkeletons.back().skeleton_3d.size() == 1){
                queue_availableSkeletons.back().skeleton_3d[0].label = 0;
            }

            if(queue_availableSkeletons.front().skeleton_3d.size() != queue_availableSkeletons.back().skeleton_3d.size()){
                for(int i = 0; i < queue_availableSkeletons.back().skeleton_3d.size(); i++){
                    queue_availableSkeletons.back().skeleton_3d[i].label = i;
                }
            }
            //queue_availableSkeletons.pop();

            // for(int i=0; i<queue_availableSkeletons. front().skeleton_3d.size(); i++){
            //     cout << "上一帧人的索引"  << i << "的label："<< queue_availableSkeletons.front().skeleton_3d[i].label << endl;
            // }

            generateSkeletonPair();
            skeletonAss();

            // for(int i=0; i<queue_availableSkeletons.back().skeleton_3d.size(); i++){
            //     cout << "当前帧人的索引"  << i << "的label："<< queue_availableSkeletons.back().skeleton_3d[i].label << "\n";
            // }
            // cout << endl;

            //setframePose(queue_availableSkeletons.back().skeleton_3d);
            verifyTracking(queue_availableSkeletons.back());
            queue_availableSkeletons.pop();


	    }
    // cout << "队列：" << queue_availableSkeletons.front().skeleton_3d.size() << endl;
    }

    // cout << "\n" << endl;
    // cout <<"队列大小：" << queue_availableSkeletons.size() << endl;


    //cout << "q_size: " << queue_availableSkeletons.size() << endl;
    //cout << "头" << queue_availableSkeletons.front().time_index << endl;
    //cout << "头1" << queue_availableSkeletons.front().skeleton_3d << endl;
    //cout << "尾" << queue_availableSkeletons.back().time_index << endl;
    //cout << "尾1" << queue_availableSkeletons.back().skeleton_3d[1][0].x << endl;
}


void Track3DPose::generateSkeletonPair()
{
    skeleton_pair.clear();
    if(queue_availableSkeletons.empty())
    {
        return;
    }
    //cout << "queue_front size: " << queue_availableSkeletons.front().skeleton_3d.size() << endl;
    //cout << "queue_back size: " << queue_availableSkeletons.back().skeleton_3d.size() << endl;
    // current_pair.resize(2);
    // current_pair[0].resize(queue_availableSkeletons.front().skeleton_3d.size()); //0 is previous frame indexes
    // current_pair[1].resize(queue_availableSkeletons.back().skeleton_3d.size()); //1 is current frame indexes
    //cout << "上一帧索引大小：" << current_pair[0].size() << endl;
    //cout << "当前帧索引大小：" << current_pair[1].size() << endl;
    //if index in current pair == 0, it is available for pairing, otherwise it has been used in previous pairing.
    for(int i=0; i<queue_availableSkeletons.back().skeleton_3d.size(); i++){
        for(int j=0; j<queue_availableSkeletons.front().skeleton_3d.size(); j++){
            if(queue_availableSkeletons.back().skeleton_3d[i].label == -1){
                // current_pair[0][i] = queue_availableSkeletons.front().skeleton_3d[i].label;
                // current_pair[1][j] = queue_availableSkeletons.back().skeleton_3d[j].label;
                SkeletonPair s_pair;
                s_pair.index_1 = i;
                s_pair.index_2 = j;
                skeleton_pair.push_back(s_pair);
            }

            //s_pair.time1 = queue_availableSkeletons.front().time_index;
            //s_pair.time2 = queue_availableSkeletons.back().time_index;
            //cout << queue_availableSkeletons.front().skeleton_3d[i].index << endl;

        }
    }
    //cout << "Pair size: " << skeleton_pair.size() << endl;
}
struct track_args{
    Track3DPose *obj;
    list<SkeletonPair>::iterator pair;
};

void* Skeleton_thread(void *args){
    track_args* m_args = (track_args*)args;
    m_args->obj->getSkeletonsCorrespondence(*(m_args->pair));
}

void Track3DPose::skeletonAss()
{
    // for(auto it = skeleton_pair.begin(); it!=skeleton_pair.end();it++){
    //     getSkeletonsCorrespondence(*it);
    //     //cout <<"对应的置信度：" << (*it).delta << endl;
    // }
    pthread_t track_thread[skeleton_pair.size()];
    track_args t_args[skeleton_pair.size()];
    int i=0;

    for(auto it = skeleton_pair.begin(); it != skeleton_pair.end(); ++it){
        t_args[i] = {this,it};
        int ret = pthread_create(&track_thread[i], NULL, Skeleton_thread, &(t_args[i]));
        if(ret == 0) {
            //ROS_INFO("Trackthread cannot create!");
        }
        ++i;
    }

    for(int i=0; i<skeleton_pair.size(); ++i){
        pthread_join(track_thread[i], NULL);
    }

    // for(auto it = current_pair.begin(); it!=current_pair.end();it++){
    //     for(int i=0; i<it->size();i++){
    //         cout << (*it)[i] << endl;
    //     }
    // }

    // for(auto it = skeleton_pair.begin(); it != skeleton_pair.end(); it++){
    //     if(it->delta > 0.2 ){
    //         it = skeleton_pair.erase(it);
    //         ROS_INFO("pair: %d, %d erased", it->index_1, it->index_2);
    //     }
    // }
    auto corres_min = min_element(skeleton_pair.begin(), skeleton_pair.end(), SkeletonPair::comp);
    double threshold = 0.4;
    int count = 0;
    while(!skeleton_pair.empty() && corres_min->delta <= threshold)
    {
        count++;
        //double delta_min = corres_min->delta;
        //ROS_INFO("pair: %d, %d is min.", corres_min->index_1, corres_min->index_2);
        int id_1 = corres_min->index_1;//current_frame
        int id_2 = corres_min->index_2;//pre_frame
        //cout << "赋值前一帧的lable: " << queue_availableSkeletons.front().skeleton_3d[id_1].label << endl;
        queue_availableSkeletons.back().skeleton_3d[id_1].label = queue_availableSkeletons.front().skeleton_3d[id_2].label;
        //ROS_INFO("current_frame label: %d", queue_availableSkeletons.back().skeleton_3d[id_1].label);
        skeleton_pair.erase(corres_min);
        //if one of the pose has been used, skip this one
        for(auto it=skeleton_pair.begin(); it != skeleton_pair.end();){
            if(it->index_1 == id_1 && it->index_2 != id_2){
                //ROS_INFO("pair: %d, %d erased", it->index_1, it->index_2);
                it = skeleton_pair.erase(it);
            }
            else{
                ++it;
            }
        }
        //cout << count << "轮删除后的pair: " << skeleton_pair.size() << endl;
        // if(current_pair[0][id_1] != 0 || current_pair[1][id_2] != 0) {
		//     skeleton_pair.erase(corres_min);
        //     continue;
	    // }
        //queue_availableSkeletons.back().skeleton_3d[id_2].index = queue_availableSkeletons.front().skeleton_3d[id_1].index;
        //ROS_INFO("pre_frame: %d pair with cur_frame: %d.", queue_availableSkeletons.front().skeleton_3d[id_1].index, queue_availableSkeletons.back().skeleton_3d[id_2].index);
        //verifytracking(queue_availableSkeletons.back());
        // queue_availableSkeletons.back().skeleton_3d[id_1].index = queue_availableSkeletons.front().skeleton_3d[id_2].index;
        // current_pair[0][id_1] = -1;
        // current_pair[1][id_2] = -1;
        if(!skeleton_pair.empty()){
            corres_min = min_element(skeleton_pair.begin(), skeleton_pair.end(), SkeletonPair::comp);
        }
        else{
            break;
        }
        //check if any pose is unpaired, if yes, give it a unique new index
        for(int i = 0; i < queue_availableSkeletons.back().skeleton_3d.size(); ++i) {
	       if(queue_availableSkeletons.back().skeleton_3d[i].label == -1) {
               int max_index = 0;
               for (int j = 0; j < queue_availableSkeletons.back().skeleton_3d.size();++j) {
	            int temp = queue_availableSkeletons.back().skeleton_3d[j].label;
                       if(max_index < temp) {
                            max_index = temp;
                        }
	            }
                queue_availableSkeletons.back().skeleton_3d[i].label = max_index + 1;
            }
        }
    }
    //ROS_INFO("pair end!");



}

bool Track3DPose::getSkeletonsCorrespondence(SkeletonPair &skeleton_pair, double threshold){
    // int t1 = skeleton_pair.time1;
    // int t2 = skeleton_pair.time2;
    pthread_mutex_lock(&track_mutex);
    vector<KeyPoint_3d> skeleton_1, skeleton_2;
    //ROS_INFO("Pair: pre_frame :%d, cur_frame :%d", skeleton_pair.index_1, skeleton_pair.index_2);
    skeleton_1 = queue_availableSkeletons.back().skeleton_3d[skeleton_pair.index_1].a_skeleton;
    skeleton_2 = queue_availableSkeletons.front().skeleton_3d[skeleton_pair.index_2].a_skeleton;
    pthread_mutex_unlock(&track_mutex);
    //Skeleton skeleton_1 = queue_availableSkeletons.front();
    //Skeleton skeleton_2 = queue_availableSkeletons.back();
    //skeleton_pair
    //ROS_INFO("Pair will pairing: cur_frame :%d, pre_frame :%d", skeleton_pair.index_1, skeleton_pair.index_2);
    int count = 0;
    double sum_C = 0.0;
    for(int i=0; i < 8; i++){
        //Point3d point_ave;
        double delta = 0.0;
        delta = getDistance(cv::Point3d(skeleton_1[i].x, skeleton_1[i].y, skeleton_1[i].z),
                            cv::Point3d(skeleton_2[i].x, skeleton_2[i].y, skeleton_2[i].z));
        // point_ave = Point3d((skeleton_1[i].x + skeleton_2[i].x)/2,
        //                     (skeleton_1[i].y + skeleton_2[i].y)/2,
        //                     (skeleton_1[i].z + skeleton_2[i].z)/2);
        // skeleton_pair.skeleton[i].x = point_ave.x;
        // skeleton_pair.skeleton[i].y = point_ave.y;
        // skeleton_pair.skeleton[i].z = point_ave.z;
        ++count;
        sum_C += delta;
    }
    skeleton_pair.delta = sum_C / count;

    // skeleton_pair.delta = sum_C / count;
    if(skeleton_pair.delta <= threshold){
        return true;
    }
    else{
        return false;
    }
}

double Track3DPose::getDistance(cv::Point3d skeleton_point_1, cv::Point3d skeleton_point_2){
    return std::sqrt((skeleton_point_1.x - skeleton_point_2.x) * (skeleton_point_1.x - skeleton_point_2.x)
                    +(skeleton_point_1.y - skeleton_point_2.y) * (skeleton_point_1.y - skeleton_point_2.y)
                    +(skeleton_point_1.z - skeleton_point_2.z) * (skeleton_point_1.z - skeleton_point_2.z));
}
