#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <string>
#include <iostream>
#include <sstream>
#include <array>
#include <opencv2/core/core.hpp>
#include <pthread.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <fstream>
#include <math.h>
#include <openpose_ros_msgs/OpenPoseHumanList.h>
#include <openpose_ros_msgs/PointWithProb.h>
#include <human_identity.hpp>

using namespace cv;
using namespace std;
double min(double a, double b)
{
    return ((a < b) ? a:b);
}

class ID_Dictionary
{
    public:
        ID_Dictionary(int Dimension):index(Dimension), status(Dimension){
            for(int i = 0; i < Dimension; ++i){
                index[i] = -1;
                status[i] = 0;
            }
            size_ = Dimension;
        }
        vector<int> index;
        vector<int> status;
        int size(){
           return size_;
        }
    private:
        int size_;

};


class ImageProcessor
{
  public:
    ImageProcessor(bool arg):it(nh), sizeColor(960, 540), id_dictionary(20) //left_key_points_center(480,260) ,right_key_points_center(480,260), left_ROI(240), right_ROI(240)
    {


        image_1 = Mat(sizeColor, CV_8UC3);
        image_2 = Mat(sizeColor, CV_8UC3);
        human_joint_names.push_back("hip");
        //human_joint_names.push_back("lShoulder");
        human_joint_names.push_back("lArm");
        human_joint_names.push_back("lForeArm");
        human_joint_names.push_back("lWrist");
        //human_joint_names.push_back("rShoulder");
        human_joint_names.push_back("rArm");
        human_joint_names.push_back("rForeArm");
        human_joint_names.push_back("rWrist");

        string color_topic_1 = "/kinect2_1/qhd/image_color_rect";
        color_1 = it.subscribe(color_topic_1.c_str(), 1,&ImageProcessor::imageCallback_1,this);
        string color_topic_2 = "/kinect2_2/qhd/image_color_rect";
        color_2 = it.subscribe(color_topic_2.c_str(), 1,&ImageProcessor::imageCallback_2,this);

        human_keypoints_sub = nh.subscribe("/openpose_ros/human_list_1", 1, &ImageProcessor::human_keypoints_callback_1, this);
        human_keypoints_sub = nh.subscribe("/openpose_ros/human_list_2", 1, &ImageProcessor::human_keypoints_callback_2, this);
        initCalibration(0.5);
    }

    void run();

  private:
    ros::NodeHandle nh;
    image_transport::ImageTransport it;
    image_transport::Subscriber color_1;
    image_transport::Subscriber color_2;

    cv::Mat distortion_1;
    cv::Mat cameraMatrix_1;
    cv::Mat distortion_2;
    cv::Mat cameraMatrix_2;

    cv::Mat image_1;
    cv::Mat image_2;
    cv::Size sizeColor;

    vector<vector<poseVector> > pose_vectors_image_all;

    vector<human> human_detected_1;
    vector<human> human_detected_last_1;
    vector<poseVector> pose_imu_2d_1;
    ID_Dictionary id_dictionary_1;

    vector<human> human_detected_2;
    vector<human> human_detected_last_2;
    vector<poseVector> pose_imu_2d_2;
    ID_Dictionary id_dictionary_2;



    tf::TransformListener robot_pose_listener;
    tf::StampedTransform base_cam_transform;
    bool isCamBaseTransformAvailable;
    bool isCamHumanTransformAvailable;
    vector<std::string> joint_names;
    vector<std::string> human_joint_names;
    bool calibrationMode;
    bool firstDepth;



    vector<Point> human_joint_pos;


    tf::Transform robot_pose_tansform;
    void loadRobotPoseFile(string);

    bool keypoints_available;

    void imageCallback_1(const sensor_msgs::ImageConstPtr& msg);
    void imageCallback_2(const sensor_msgs::ImageConstPtr& msg);

    void initCalibration(double scale);

    void getImageCoordinate(Point3f& world_cord, KeyPoint_prob& image_cord, Mat& cameraMatrix);
    void loadCalibrationFiles(string& calib_path, cv::Mat& cameraMatrix, cv::Mat& distortion, double scale);
    void calculateHumanPose(vector<KeyPoint_prob>& joint_image_cords, int camera_index);
    bool getPoseVector(vector<KeyPoint_prob>& key_points, vector<poseVector>& pose_vectors);
    double getVectorLength(const poseVector& pose_vector);

    double getVectorSimilarity(const poseVector& pose_vector_1, const poseVector& pose_vector_2);

    double getPoseSimilarity(const vector<poseVector>& pose_vectors_1, const vector<poseVector>& pose_vectors_2);

    void human_keypoints_callback_1(openpose_ros_msgs::OpenPoseHumanList keypoints);
    void human_keypoints_callback_2(openpose_ros_msgs::OpenPoseHumanList keypoints);

    void identify(vector<KeyPoint_prob>& human_joints);

    void draw_human_pose(Mat& image,const vector<KeyPoint_prob>& human_joints, Scalar color = Scalar(0, 0, 255));
    int getAvailableId(ID_Dictionary& id_list);
    double distance(const KeyPoint_prob key_point_1, const KeyPoint_prob key_point_2);
    double getKeyPointsSimilarity(const vector<KeyPoint_prob>& key_points_1, const vector<KeyPoint_prob>& key_points_2);

};

int ImageProcessor::getAvailableId(ID_Dictionary& id_list){
    for(int i = 0; i < id_list.size(); ++i)
    {
        if(id_list.status[i] == 0)
        {
            return i;
        }
    }
    return -1;
}

void ImageProcessor::initCalibration(double scale){
    string calib_path_1 = "/home/agent/catkin_ws/src/iai_kinect2/kinect2_bridge/data/003415165047";
    string calib_path_2 = "/home/agent/catkin_ws/src/iai_kinect2/kinect2_bridge/data/092465240847";
    loadCalibrationFiles(calib_path_1, cameraMatrix_1, distortion_1, scale);
    loadCalibrationFiles(calib_path_2, cameraMatrix_2, distortion_2, scale);
}

void ImageProcessor::imageCallback_1(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {

      Mat color_mat = cv_bridge::toCvShare(msg, "bgr8")->image;

      image_1 = color_mat.clone();
    }
    catch (cv_bridge::Exception& e)
    {
    	ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

void ImageProcessor::imageCallback_2(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {

      Mat color_mat = cv_bridge::toCvShare(msg, "bgr8")->image;

      image_2 = color_mat.clone();
    }
    catch (cv_bridge::Exception& e)
    {
    	ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

void ImageProcessor::loadCalibrationFiles(string& calib_path, cv::Mat& cameraMatrix, cv::Mat& distortion, double scale)
{

    cv::FileStorage fs;

    cv::Mat cameraMatrix_origin;


  if(fs.open(calib_path + "/calib_color.yaml", cv::FileStorage::READ))
  {
    fs["cameraMatrix"] >> cameraMatrix_origin;
    cameraMatrix = cameraMatrix_origin.clone();
    cameraMatrix.at<double>(0, 0) *= scale;
    cameraMatrix.at<double>(1, 1) *= scale;
    cameraMatrix.at<double>(0, 2) *= scale;
    cameraMatrix.at<double>(1, 2) *= scale;

    distortion= cv::Mat::zeros(1, 5, CV_64F);

    //fs["distortionCoefficients"] >> distortion_color;
    cout << "color matrix load success"<< endl;
    fs.release();


  }
  else
  {
    cout << "No calibration file: calib_color.yalm, using default calibration setting" << endl;
    cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    distortion = cv::Mat::zeros(1, 5, CV_64F);


  }


}

void ImageProcessor::getImageCoordinate(Point3f& world_cord, KeyPoint_prob& image_cord, Mat& cameraMatrix)
{
    double fx = cameraMatrix.at<double>(0, 0);
    double fy = cameraMatrix.at<double>(1, 1);
    double cx = cameraMatrix.at<double>(0, 2);
    double cy = cameraMatrix.at<double>(1, 2);

    image_cord.x = (int)(world_cord.x * fx / world_cord.z + cx);
    image_cord.y = (int)(world_cord.y * fy / world_cord.z + cy);
    image_cord.p = 1;
}

void ImageProcessor::calculateHumanPose(vector<KeyPoint_prob>& joint_image_cords, int camera_index);
{
  //tf::TransformListener robot_pose_listener;
    string human_reference_frame;

    char human_reference_frame[30];
    sprintf(human_reference_frame, "camera_base_%d", camera_index);

    if(camera_index = 1){
        cameraMatrix = cameraMatrix_1.clone();
    }
    if(camera_index = 2){
        cameraMatrix = cameraMatrix_2.clone();
    }
    //human_reference_frame = "camera_base";


    tf::StampedTransform joint_transforms;
    tf::StampedTransform cam_hip_transform;
    try
    {
        robot_pose_listener.lookupTransform(human_reference_frame, "hip", ros::Time(0), cam_hip_transform);
    }

    catch(tf::TransformException ex)
    {
        //ROS_ERROR("%s", ex.what());
        isCamHumanTransformAvailable = false;
        return;
    }

    isCamHumanTransformAvailable = true;
    Point3f hip_location(cam_hip_transform.getOrigin().x(), cam_hip_transform.getOrigin().y(), cam_hip_transform.getOrigin().z());
    KeyPoint_prob hip_image_cord;

    getImageCoordinate(hip_location, hip_image_cord, cameraMatrix);
    //circle(color_mat, hip_image_cord, 2, Scalar(0, 0, 255), -1, 8);
    /***        else{
          getImageCoordinate(location, joint_image_cord, cameraMatrix_sub);
        }
    ostringstream cord_text;
    cord_text.str("");
    cord_text << "base_position:" << " at" << '(' << base_location.x << ',' << base_location.y << ',' << base_location.z << ')';
    putText(color_mat, cord_text.str(), Point(20,400), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,, 0));
    ***/
    //vector<Point3f> joint_3d_cords;
    //joint_3d_cords.push_back(hip_location);
    //vector<Point> joint_image_cords;
    joint_image_cords.push_back(hip_image_cord);
    for(int i = 1; i < human_joint_names.size(); i++)
    {
        try
        {
            robot_pose_listener.lookupTransform(human_reference_frame.c_str(), human_joint_names[i], ros::Time(0), joint_transforms);
        }
        catch(tf::TransformException ex)
        {
            //ROS_ERROR("%s", ex.what());
            continue;
        }
        Point3f location(joint_transforms.getOrigin().x(), joint_transforms.getOrigin().y(), joint_transforms.getOrigin().z());
        //joint_3d_cords.push_back(location);
        KeyPoint_prob joint_image_cord;

        getImageCoordinate(location, joint_image_cord, cameraMatrix);

        joint_image_cords.push_back(joint_image_cord);

    }
    //ROS_INFO("Human Pose calculated");
}



bool ImageProcessor::getPoseVector(vector<KeyPoint_prob>& key_points, vector<poseVector>& pose_vectors)
{
    if(key_points.size()!=7)
    {
        return false;
    }

    pose_vectors.clear();
    poseVector pose_vector_ite;
    //lShoulder vector
    pose_vector_ite.x = key_points[1].x - key_points[0].x;
    pose_vector_ite.y = key_points[1].y - key_points[0].y;
    pose_vector_ite.p = key_points[1].p;
    pose_vectors.push_back(pose_vector_ite);

    //lArm vector
    pose_vector_ite.x = key_points[2].x - key_points[1].x;
    pose_vector_ite.y = key_points[2].y - key_points[1].y;
    pose_vector_ite.p = (key_points[2].p + key_points[1].p) / 2;
    pose_vectors.push_back(pose_vector_ite);
    //lWrist vector
    pose_vector_ite.x = key_points[3].x - key_points[2].x;
    pose_vector_ite.y = key_points[3].y - key_points[2].y;
    pose_vector_ite.p = (key_points[3].p + key_points[2].p) / 2;
    pose_vectors.push_back(pose_vector_ite);

    //rShoulder vector
    pose_vector_ite.x = key_points[4].x - key_points[0].x;
    pose_vector_ite.y = key_points[4].y - key_points[0].y;
    pose_vector_ite.p = key_points[4].p;
    pose_vectors.push_back(pose_vector_ite);

    //lArm vector
    pose_vector_ite.x = key_points[5].x - key_points[4].x;
    pose_vector_ite.y = key_points[5].y - key_points[4].y;
    pose_vector_ite.p = (key_points[5].p + key_points[4].p) / 2;
    pose_vectors.push_back(pose_vector_ite);
    //lWrist vector
    pose_vector_ite.x = key_points[6].x - key_points[5].x;
    pose_vector_ite.y = key_points[6].y - key_points[5].y;
    pose_vector_ite.p = (key_points[6].p + key_points[5].p) / 2;
    pose_vectors.push_back(pose_vector_ite);

    return true;
}

double ImageProcessor::getVectorLength(const poseVector& pose_vector){
    return sqrt(pose_vector.x * pose_vector.x + pose_vector.y * pose_vector.y);
}

double ImageProcessor::getVectorSimilarity(const poseVector& pose_vector_1, const poseVector& pose_vector_2)
{
    double COS_Theta;
    COS_Theta = (pose_vector_1.x * pose_vector_2.x + pose_vector_2.y * pose_vector_1.y)/(getVectorLength(pose_vector_1) * getVectorLength(pose_vector_2));
    return (COS_Theta + 1.0) / 2.0;
}

double ImageProcessor::getPoseSimilarity(const vector<poseVector>& pose_vectors_1, const vector<poseVector>& pose_vectors_2)
{
    int dimension = pose_vectors_1.size();
    if (dimension != pose_vectors_2.size())
    {
        return 0;
    }
    double simliarity = 0.0;
    double denominator = 0.0;
    int count = 0;

    for (int i = 0; i < dimension; ++i){
        if(pose_vectors_1[i].p > 0 && pose_vectors_2[i].p > 0){
            denominator += pose_vectors_1[i].p;
            count++;
            //simliarity += getVectorSimilarity(pose_vectors_1[i], pose_vectors_2[i]);
            simliarity += getVectorSimilarity(pose_vectors_1[i], pose_vectors_2[i]) * pose_vectors_1[i].p;
        }
    }

    //return simliarity / count;
    return simliarity / denominator;
}

void ImageProcessor::human_keypoints_callback_1(openpose_ros_msgs::OpenPoseHumanList keypoints)
{
  int person_num = keypoints.num_humans;
  //pose_vectors_image_all.clear();
  human_detected_last_1 = human_detected_1;
  //std::array<bool, 20> id_list;
  //id_list.fill(true);
  //for(int i = 0; i < human_detected_last.size(); ++i){
  //    int index = human_detected_last[i].id();
  //    id_list[index] = false;
  //}

  ID_Dictionary id_dictionary_tmp = id_dictionary_1;

  human_detected_1.clear();
  if(person_num > 0){
      int person_index = 0;
      for(int person = 0;person < person_num; ++person)
      {
          auto body_keypoints = keypoints.human_list[person].body_key_points_with_prob;

          int count = 0;
          double prob_sum = 0.0;
          for(int i = 0; i < body_keypoints.size(); i++)
          {
            if(body_keypoints[i].prob > 0.0)
            {
              prob_sum += body_keypoints[i].prob;
              count ++;
            }
          }
          double prob_eval = prob_sum / count;

          if(prob_eval < 0.4){
              continue;
          }


          vector<KeyPoint_prob> key_points_basic;
          KeyPoint_prob key_point_ite;
          //hip
          key_point_ite.x = body_keypoints[8].x;
          key_point_ite.y = body_keypoints[8].y;
          key_point_ite.p = body_keypoints[8].prob;
          key_points_basic.push_back(key_point_ite);
          //lArm
          key_point_ite.x = body_keypoints[5].x;
          key_point_ite.y = body_keypoints[5].y;
          key_point_ite.p = body_keypoints[5].prob;
          key_points_basic.push_back(key_point_ite);
          //lElbow
          key_point_ite.x = body_keypoints[6].x;
          key_point_ite.y = body_keypoints[6].y;
          key_point_ite.p = body_keypoints[6].prob;
          key_points_basic.push_back(key_point_ite);
          //lWrist
          key_point_ite.x = body_keypoints[7].x;
          key_point_ite.y = body_keypoints[7].y;
          key_point_ite.p = body_keypoints[7].prob;
          key_points_basic.push_back(key_point_ite);
          //rArm
          key_point_ite.x = body_keypoints[2].x;
          key_point_ite.y = body_keypoints[2].y;
          key_point_ite.p = body_keypoints[2].prob;
          key_points_basic.push_back(key_point_ite);
          //rElbow
          key_point_ite.x = body_keypoints[3].x;
          key_point_ite.y = body_keypoints[3].y;
          key_point_ite.p = body_keypoints[3].prob;
          key_points_basic.push_back(key_point_ite);
          //rWrist
          key_point_ite.x = body_keypoints[4].x;
          key_point_ite.y = body_keypoints[4].y;
          key_point_ite.p = body_keypoints[4].prob;
          key_points_basic.push_back(key_point_ite);

          vector<poseVector> pose_vectors_image;
          getPoseVector(key_points_basic, pose_vectors_image);

          human human_tmp(-1);
          if(!human_detected_last_1.empty()){
            vector<double> distance_pool;
            for(int i = 0; i < human_detected_last_1.size(); ++i){
                double dis = getKeyPointsSimilarity(key_points_basic, human_detected_last_1[i].key_points());
                distance_pool.push_back(dis);
            }
            auto minDis = std::min_element(distance_pool.begin(), distance_pool.end());
            //Found coincide last component
            if(*minDis < 20){
                int index = std::distance(distance_pool.begin(), minDis);
                human_tmp.setId(human_detected_last_1[index].id());
                human_tmp.setTime(human_detected_last_1[index].time() + 1);
                human_tmp.setKeyPoints(key_points_basic);
                human_tmp.setPose(pose_vectors_image);
                human_tmp.setPosition(body_keypoints[0].x, body_keypoints[0].y);
                //id_list_tmp[human_detected_last_1[index].id()] = -1;
                id_dictionary_tmp.status[human_detected_last_1[index].id()] = -1;
                id_dictionary_tmp.index[human_detected_last_1[index].id()] = person_index;

            }
            //new person found
            else{
                int id = getAvailableId(id_dictionary_tmp);
                human_tmp.setId(id);
                human_tmp.setKeyPoints(key_points_basic);
                human_tmp.setPose(pose_vectors_image);
                human_tmp.setPosition(body_keypoints[0].x, body_keypoints[0].y);
                //id_list_tmp[human_detected_last_1[index].id()] = -2;
                id_dictionary_tmp.status[id] = -2;
                id_dictionary_tmp.index[id] = person_index;

            }
          }
          else{
            human_tmp.setId(person);
            human_tmp.setKeyPoints(key_points_basic);
            human_tmp.setPose(pose_vectors_image);
            human_tmp.setPosition(body_keypoints[0].x, body_keypoints[0].y);
            id_dictionary_tmp.status[person] = -2;
            id_dictionary_tmp.index[person] = person_index;
          }

          person_index ++;
          human_detected_1.push_back(human_tmp);
      }
      //Check status
      for(int i = 0; i < id_dictionary_tmp.size(); ++i)
      {
          if( id_dictionary_tmp.status[i] == 1){
              for(int j = 0; j < id_dictionary_tmp.size(); ++j)
              {
                  if(id_dictionary_tmp.status[j] == -2){
                      double dis = getKeyPointsSimilarity(human_detected_1[id_dictionary_tmp.index[j]].key_points(), human_detected_last_1[id_dictionary_tmp.index[i]].key_points());
                      if(dis < 50)
                      {
                          human_detected_1[id_dictionary_tmp.index[j]].setId(i);
                          human_detected_1[id_dictionary_tmp.index[j]].setTime(human_detected_last_1[id_dictionary_tmp.index[i]].time() + 1);
                          id_dictionary_tmp.index[i] = id_dictionary_tmp.index[j];
                          id_dictionary_tmp.status[j] = 0;
                          id_dictionary_tmp.index[j] = -1;
                      }
                  }
              }
          }
      }
      for(int i = 0; i < id_dictionary_tmp.size(); ++i)
      {
          if( id_dictionary_tmp.status[i] < 0)
          {
              id_dictionary_tmp.status[i] = 1;
          }
      }
      id_dictionary_1 = id_dictionary_tmp;

    }
}

void ImageProcessor::human_keypoints_callback_2(openpose_ros_msgs::OpenPoseHumanList keypoints)
{
  int person_num = keypoints.num_humans;
  //pose_vectors_image_all.clear();
  human_detected_last_2 = human_detected_2;
  //std::array<bool, 20> id_list;
  //id_list.fill(true);
  //for(int i = 0; i < human_detected_last.size(); ++i){
  //    int index = human_detected_last[i].id();
  //    id_list[index] = false;
  //}

  ID_Dictionary id_dictionary_tmp = id_dictionary_2;

  human_detected_2.clear();
  if(person_num > 0){
      int person_index = 0;
      for(int person = 0;person < person_num; ++person)
      {
          auto body_keypoints = keypoints.human_list[person].body_key_points_with_prob;

          int count = 0;
          double prob_sum = 0.0;
          for(int i = 0; i < body_keypoints.size(); i++)
          {
            if(body_keypoints[i].prob > 0.0)
            {
              prob_sum += body_keypoints[i].prob;
              count ++;
            }
          }
          double prob_eval = prob_sum / count;

          if(prob_eval < 0.4){
              continue;
          }


          vector<KeyPoint_prob> key_points_basic;
          KeyPoint_prob key_point_ite;
          //hip
          key_point_ite.x = body_keypoints[8].x;
          key_point_ite.y = body_keypoints[8].y;
          key_point_ite.p = body_keypoints[8].prob;
          key_points_basic.push_back(key_point_ite);
          //lArm
          key_point_ite.x = body_keypoints[5].x;
          key_point_ite.y = body_keypoints[5].y;
          key_point_ite.p = body_keypoints[5].prob;
          key_points_basic.push_back(key_point_ite);
          //lElbow
          key_point_ite.x = body_keypoints[6].x;
          key_point_ite.y = body_keypoints[6].y;
          key_point_ite.p = body_keypoints[6].prob;
          key_points_basic.push_back(key_point_ite);
          //lWrist
          key_point_ite.x = body_keypoints[7].x;
          key_point_ite.y = body_keypoints[7].y;
          key_point_ite.p = body_keypoints[7].prob;
          key_points_basic.push_back(key_point_ite);
          //rArm
          key_point_ite.x = body_keypoints[2].x;
          key_point_ite.y = body_keypoints[2].y;
          key_point_ite.p = body_keypoints[2].prob;
          key_points_basic.push_back(key_point_ite);
          //rElbow
          key_point_ite.x = body_keypoints[3].x;
          key_point_ite.y = body_keypoints[3].y;
          key_point_ite.p = body_keypoints[3].prob;
          key_points_basic.push_back(key_point_ite);
          //rWrist
          key_point_ite.x = body_keypoints[4].x;
          key_point_ite.y = body_keypoints[4].y;
          key_point_ite.p = body_keypoints[4].prob;
          key_points_basic.push_back(key_point_ite);

          vector<poseVector> pose_vectors_image;
          getPoseVector(key_points_basic, pose_vectors_image);

          human human_tmp(-1);
          if(!human_detected_last_2.empty()){
            vector<double> distance_pool;
            for(int i = 0; i < human_detected_last_2.size(); ++i){
                double dis = getKeyPointsSimilarity(key_points_basic, human_detected_last_2[i].key_points());
                distance_pool.push_back(dis);
            }
            auto minDis = std::min_element(distance_pool.begin(), distance_pool.end());
            //Found coincide last component
            if(*minDis < 20){
                int index = std::distance(distance_pool.begin(), minDis);
                human_tmp.setId(human_detected_last_2[index].id());
                human_tmp.setTime(human_detected_last_2[index].time() + 1);
                human_tmp.setKeyPoints(key_points_basic);
                human_tmp.setPose(pose_vectors_image);
                human_tmp.setPosition(body_keypoints[0].x, body_keypoints[0].y);
                //id_list_tmp[human_detected_last_2[index].id()] = -1;
                id_dictionary_tmp.status[human_detected_last_2[index].id()] = -1;
                id_dictionary_tmp.index[human_detected_last_2[index].id()] = person_index;

            }
            //new person found
            else{
                int id = getAvailableId(id_dictionary_tmp);
                human_tmp.setId(id);
                human_tmp.setKeyPoints(key_points_basic);
                human_tmp.setPose(pose_vectors_image);
                human_tmp.setPosition(body_keypoints[0].x, body_keypoints[0].y);
                //id_list_tmp[human_detected_last_2[index].id()] = -2;
                id_dictionary_tmp.status[id] = -2;
                id_dictionary_tmp.index[id] = person_index;

            }
          }
          else{
            human_tmp.setId(person);
            human_tmp.setKeyPoints(key_points_basic);
            human_tmp.setPose(pose_vectors_image);
            human_tmp.setPosition(body_keypoints[0].x, body_keypoints[0].y);
            id_dictionary_tmp.status[person] = -2;
            id_dictionary_tmp.index[person] = person_index;
          }

          person_index ++;
          human_detected_2.push_back(human_tmp);
      }
      //Check status
      for(int i = 0; i < id_dictionary_tmp.size(); ++i)
      {
          if( id_dictionary_tmp.status[i] == 1){
              for(int j = 0; j < id_dictionary_tmp.size(); ++j)
              {
                  if(id_dictionary_tmp.status[j] == -2){
                      double dis = getKeyPointsSimilarity(human_detected_2[id_dictionary_tmp.index[j]].key_points(), human_detected_last_2[id_dictionary_tmp.index[i]].key_points());
                      if(dis < 50)
                      {
                          human_detected_2[id_dictionary_tmp.index[j]].setId(i);
                          human_detected_2[id_dictionary_tmp.index[j]].setTime(human_detected_last_2[id_dictionary_tmp.index[i]].time() + 1);
                          id_dictionary_tmp.index[i] = id_dictionary_tmp.index[j];
                          id_dictionary_tmp.status[j] = 0;
                          id_dictionary_tmp.index[j] = -1;
                      }
                  }
              }
          }
      }
      for(int i = 0; i < id_dictionary_tmp.size(); ++i)
      {
          if( id_dictionary_tmp.status[i] < 0)
          {
              id_dictionary_tmp.status[i] = 1;
          }
      }
      id_dictionary_2 = id_dictionary_tmp;

    }
}



double ImageProcessor::distance(const KeyPoint_prob key_point_1, const KeyPoint_prob key_point_2)
{
    return sqrt((key_point_1.x - key_point_2.x) * (key_point_1.x - key_point_2.x) + (key_point_1.y - key_point_2.y) * (key_point_1.y - key_point_2.y));
}

double ImageProcessor::getKeyPointsSimilarity(const vector<KeyPoint_prob>& key_points_1, const vector<KeyPoint_prob>& key_points_2)
{
      double average_distance = 0.0;
      double denominator = 0.0;
      for(int i; i < key_points_1.size(); ++i){
          if(key_points_1[i].p > 0.2 && key_points_2[i].p > 0.2){
              average_distance += distance(key_points_1[i], key_points_2[i]) * min(key_points_1[i].p, key_points_2[i].p);
              denominator += min(key_points_1[i].p, key_points_2[i].p);
          }
      }
      average_distance = average_distance / denominator;
      return average_distance;
}
void ImageProcessor::identify(vector<KeyPoint_prob>& human_joints)
{
    {
      if(human_detected.empty())
          ROS_INFO("human not deteted");
      if(pose_imu_2d.empty())
          ROS_INFO("imu pose not received");
      if(image.empty())
          ROS_INFO("no image get!");
    }
    if (!human_detected.empty() && !pose_imu_2d.empty() && !image.empty()){

        //ROS_INFO("identifying..");
        //ostringstream cord_text;
        cv::Mat displyImg = image.clone();
        draw_human_pose(displyImg, human_joints);
        for (int i = 0; i < human_detected.size(); ++i){
            draw_human_pose(displyImg, human_detected[i].key_points(), Scalar(255,0,0));
            double similarity;
            similarity = getPoseSimilarity(human_detected[i].pose(), pose_imu_2d);
            auto position = human_detected[i].position();
            char text[100];
            //cord_text << "similarity:" << similarity;
            sprintf(text, "ID:%d", human_detected[i].id());
            putText(displyImg, text, Point(position.x + 40,position.y + 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 4);
            sprintf(text, "similarity: %1.3f", similarity);
            putText(displyImg, text, Point(position.x + 40,position.y + 80), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 4);
            sprintf(text, "time: %d", human_detected[i].time());
            putText(displyImg, text, Point(position.x + 40,position.y + 120), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 4);
        }
        cv::imshow("simliarity",displyImg);
        cv::waitKey(30);
    }
}

void ImageProcessor::draw_human_pose(Mat& image, const vector<KeyPoint_prob>& human_joints, Scalar color)
{
  circle(image, cv::Point(human_joints[0].x, human_joints[0].y), 3, color, -1, 8);
  //draw joints on image
  for(int i = 1; i < (human_joints.size()); i++)
  {
      if(human_joints[i].p < 0.01)
          continue;
      circle(image, cv::Point(human_joints[i].x, human_joints[i].y), 3, color, -1, 8);
      if (i != 4)
        line(image, cv::Point(human_joints[i-1].x, human_joints[i-1].y), cv::Point(human_joints[i].x, human_joints[i].y), color, 2);
      else
        line(image,cv::Point(human_joints[0].x, human_joints[0].y), cv::Point(human_joints[i].x, human_joints[i].y), color, 2);
  }
    //ROS_INFO("Human Pose drawed");
}

void ImageProcessor::run()
{
  ros::Rate rate(30);
  while(ros::ok())
  {
    vector<KeyPoint_prob> key_point_imu_1;
    calculateHumanPose(key_point_imu_1. 1);
    getPoseVector(key_point_imu_1, pose_imu_2d_1);

    vector<KeyPoint_prob> key_point_imu_2;
    calculateHumanPose(key_point_imu_2. 2);
    getPoseVector(key_point_imu_2, pose_imu_2d_2);
      //ROS_INFO("IMU Pose get!");
    identify(key_point_imu_1);
    ros::spinOnce();
    rate.sleep();
  }
}

int main(int argc, char** argv){
   ros::init(argc, argv, "human_identifier");
   ImageProcessor identifier(true);
   identifier.run();
   return 0;
}
