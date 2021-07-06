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

#include <inertial_poser/ROI_Package.h>
#include <inertial_poser/UpperBodyKeyPoints.h>
#include <inertial_poser/KeyPoint.h>

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

class punish_unit
{
  public:
    punish_unit(int id, double scale)
    {
        this->id = id;
        this->scale = scale;
    }
    int id;
    double scale;
};


class ImageProcessor
{
  public:
    ImageProcessor(int camera_index):it(nh), sizeColor(960, 540), id_dictionary(20) //left_key_points_center(480,260) ,right_key_points_center(480,260), left_ROI(240), right_ROI(240)
    {
        camera_index_ = camera_index;

        selected_id = -1;

        human_ns = "/human_imu";
        result_human_ns = "/human_1";

        image = Mat(sizeColor, CV_8UC3);
        human_joint_names.push_back("/hip");
        //human_joint_names.push_back("lShoulder");
        human_joint_names.push_back("/rArm");
        human_joint_names.push_back("/rForeArm");
        human_joint_names.push_back("/rWrist");
        //human_joint_names.push_back("rShoulder");
        human_joint_names.push_back("/lArm");
        human_joint_names.push_back("/lForeArm");
        human_joint_names.push_back("/lWrist");

        //string color_topic= "/kinect2_1/qhd/image_color_rect";
        char color_topic[50];
        sprintf(color_topic, "/kinect2_%d/qhd/image_color_rect", camera_index);
        color = it.subscribe(color_topic, 1,&ImageProcessor::imageCallback ,this);

        char key_points_topic[50];
        sprintf(key_points_topic, "/openpose_ros/human_list_%d", camera_index);

        human_keypoints_sub = nh.subscribe(key_points_topic, 1, &ImageProcessor::human_keypoints_callback, this);
        //human_keypoints_sub = nh.subscribe("/openpose_ros/human_list_2", 1, &ImageProcessor::human_keypoints_callback_2, this);
        initCalibration(0.5, camera_index_);
    }

    void run();
    bool getHuman_with_most_confidence(human& human_max);
    void punish(int id);

  private:
    string human_ns;
    string result_human_ns;
    int camera_index_;

    ros::NodeHandle nh;
    image_transport::ImageTransport it;
    image_transport::Subscriber color;
    ros::Subscriber human_keypoints_sub;


    cv::Mat distortion;
    cv::Mat cameraMatrix;

    cv::Mat image;
    cv::Size sizeColor;

    vector<vector<poseVector> > pose_vectors_image_all;

    vector<human> human_detected;
    vector<human> human_detected_last;
    vector<poseVector> pose_imu_2d;
    ID_Dictionary id_dictionary;

    tf::TransformListener robot_pose_listener;
    tf::StampedTransform base_cam_transform;
    bool isCamBaseTransformAvailable;
    bool isCamHumanTransformAvailable;
    vector<std::string> joint_names;
    vector<std::string> human_joint_names;
    bool calibrationMode;
    bool firstDepth;

    vector<punish_unit> punish_list;

    vector<Point> human_joint_pos;

    int selected_id;


    tf::Transform robot_pose_tansform;
    void loadRobotPoseFile(string);

    bool keypoints_available;

    void imageCallback(const sensor_msgs::ImageConstPtr& msg);

    void initCalibration(double scale, int camera_index);

    void getImageCoordinate(Point3f& world_cord, KeyPoint_prob& image_cord, Mat& cameraMatrix);
    void loadCalibrationFiles(string& calib_path, cv::Mat& cameraMatrix, cv::Mat& distortion, double scale);
    void calculateHumanPose(vector<KeyPoint_prob>& joint_image_cords, int camera_index, bool);
    bool getPoseVector(vector<KeyPoint_prob>& key_points, vector<poseVector>& pose_vectors);
    double getVectorLength(const poseVector& pose_vector);

    double getVectorSimilarity(const poseVector& pose_vector_1, const poseVector& pose_vector_2);

    double getPoseSimilarity(const vector<poseVector>& pose_vectors_1, const vector<poseVector>& pose_vectors_2);

    void human_keypoints_callback(openpose_ros_msgs::OpenPoseHumanList keypoints);

    void identify(vector<KeyPoint_prob>& human_joints, vector<KeyPoint_prob>& human_joints_result);

    void draw_human_pose(Mat& image,const vector<KeyPoint_prob>& human_joints, Scalar color = Scalar(0, 0, 255));
    void draw_imu_pose(Mat& image, const vector<KeyPoint_prob>& human_joints,  Scalar color = Scalar(0, 0, 255));
    int getAvailableId(ID_Dictionary& id_list);
    double distance(const KeyPoint_prob key_point_1, const KeyPoint_prob key_point_2);
    double getKeyPointsSimilarity(const vector<KeyPoint_prob>& key_points_1, const vector<KeyPoint_prob>& key_points_2);



};
void ImageProcessor::punish(int id)
{
  for(int i = 0; i < human_detected.size(); ++i)
  {
      if(human_detected[i].id() == id)
      {
          human_detected[i].updatePunishment();
          return;
      }
  }
  punish_list.push_back(punish_unit(id, 0.9));
}

bool ImageProcessor::getHuman_with_most_confidence(human& human_max)
{
    if(!human_detected.empty())
    {
        vector<double> similarity_list;
        int candidate_index = -1;
        for(int i = 0; i < human_detected.size();++i)
        {
            if(human_detected[i].id() == selected_id)
            {
                candidate_index = i;
            }
            similarity_list.push_back(human_detected[i].similarity());
        }
        auto maxSim = std::max_element(similarity_list.begin(), similarity_list.end());
        //Found coincide last component
        if(*maxSim > 0.88){
            int index = std::distance(similarity_list.begin(), maxSim);
            if(candidate_index == index)
            {
              human_max = human_detected[index];
              human_detected[index].resetPunish();
            }
            else if(human_detected[candidate_index].similarity() > 0.88)
            {
              human_max = human_detected[candidate_index];
              human_detected[index].resetPunish();
            }
            else
            {
              selected_id = human_detected[index].id();
              human_detected[index].resetPunish();

            }
            return true;
        }
        else
        {
            return false;
        }
    }
    else
    {
        return false;
    }
}


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

void ImageProcessor::initCalibration(double scale, int camera_index){
    string calib_path = "/home/agent/catkin_ws/src/iai_kinect2/kinect2_bridge/data/003415165047";
    loadCalibrationFiles(calib_path, cameraMatrix, distortion, scale);
}

void ImageProcessor::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {

      Mat color_mat = cv_bridge::toCvShare(msg, "bgr8")->image;

      image = color_mat.clone();
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

void ImageProcessor::calculateHumanPose(vector<KeyPoint_prob>& joint_image_cords, int camera_index, bool result = false)
{
  //tf::TransformListener robot_pose_listener;
    //string human_reference_frame;

    vector<std::string> human_joint_names_this;
    if(result)
    {
        for(int i = 0; i < human_joint_names.size(); ++i)
        {
            string name_temp = result_human_ns + human_joint_names[i];
            human_joint_names_this.push_back(name_temp);
        }
    }
    else
    {
        for(int i = 0; i < human_joint_names.size(); ++i)
        {
            string name_temp = human_ns + human_joint_names[i];
            human_joint_names_this.push_back(name_temp);
        }
    }

    char human_reference_frame[30];
    sprintf(human_reference_frame, "camera_base_%d", camera_index);

    //human_reference_frame = "camera_base";


    tf::StampedTransform joint_transforms;
    tf::StampedTransform cam_hip_transform;
    try
    {
        robot_pose_listener.lookupTransform(human_reference_frame, human_joint_names_this[0], ros::Time(0), cam_hip_transform);
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
    for(int i = 1; i < human_joint_names_this.size(); i++)
    {
        try
        {
            robot_pose_listener.lookupTransform(human_reference_frame, human_joint_names_this[i], ros::Time(0), joint_transforms);
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
    //rShoulder vector
    pose_vector_ite.x = key_points[1].x - key_points[0].x;
    pose_vector_ite.y = key_points[1].y - key_points[0].y;
    pose_vector_ite.p = key_points[1].p;
    pose_vectors.push_back(pose_vector_ite);

    //rArm vector
    pose_vector_ite.x = key_points[2].x - key_points[1].x;
    pose_vector_ite.y = key_points[2].y - key_points[1].y;
    pose_vector_ite.p = (key_points[2].p + key_points[1].p) / 2;
    pose_vectors.push_back(pose_vector_ite);
    //rWrist vector
    pose_vector_ite.x = key_points[3].x - key_points[2].x;
    pose_vector_ite.y = key_points[3].y - key_points[2].y;
    pose_vector_ite.p = (key_points[3].p + key_points[2].p) / 2;
    pose_vectors.push_back(pose_vector_ite);

    //lShoulder vector
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
    if(denominator > 0.0)
    {
      return simliarity / denominator;
    }
    else
    {
      return -1;
    }

}

void ImageProcessor::human_keypoints_callback(openpose_ros_msgs::OpenPoseHumanList keypoints)
{
  int person_num = keypoints.num_humans;
  //pose_vectors_image_all.clear();
  human_detected_last = human_detected;
  //std::array<bool, 20> id_list;
  //id_list.fill(true);
  //for(int i = 0; i < human_detected_last.size(); ++i){
  //    int index = human_detected_last[i].id();
  //    id_list[index] = false;
  //}

  ID_Dictionary id_dictionary_tmp = id_dictionary;

  human_detected.clear();

  vector< vector<double> > distance_pool;
  distance_pool.resize(person_num);

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


          vector<poseVector> pose_vectors_image;
          getPoseVector(key_points_basic, pose_vectors_image);

          human human_tmp(-1);
          if(!human_detected_last.empty()){
            vector<double> distance_pool;
            for(int i = 0; i < human_detected_last.size(); ++i){
                double dis = getKeyPointsSimilarity(key_points_basic, human_detected_last[i].key_points());
                distance_pool.push_back(dis);
            }
            auto minDis = std::min_element(distance_pool.begin(), distance_pool.end());
            //Found coincide last component
            if(*minDis < 20){
                int index = std::distance(distance_pool.begin(), minDis);
                human_tmp = human_detected_last[index];
                human_tmp.updateTime();
                human_tmp.setKeyPoints(key_points_basic);
                human_tmp.setPose(pose_vectors_image);
                human_tmp.setPosition(body_keypoints[0].x, body_keypoints[0].y);
                //id_list_tmp[human_detected_last[index].id()] = -1;
                id_dictionary_tmp.status[human_detected_last[index].id()] = -1;
                id_dictionary_tmp.index[human_detected_last[index].id()] = person_index;

            }
            //new person found
            else{
                int id = getAvailableId(id_dictionary_tmp);
                human_tmp.setId(id);
                human_tmp.setKeyPoints(key_points_basic);
                human_tmp.setPose(pose_vectors_image);
                human_tmp.setPosition(body_keypoints[0].x, body_keypoints[0].y);
                //id_list_tmp[human_detected_last[index].id()] = -2;
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
          human_detected.push_back(human_tmp);
      }
      //Check status
      for(int i = 0; i < id_dictionary_tmp.size(); ++i)
      {
          if( id_dictionary_tmp.status[i] == 1){
              bool found = false;
              for(int j = 0; j < id_dictionary_tmp.size(); ++j)
              {
                  if(id_dictionary_tmp.status[j] == -2){
                      double dis = getKeyPointsSimilarity(human_detected[id_dictionary_tmp.index[j]].key_points(), human_detected_last[id_dictionary_tmp.index[i]].key_points());
                      if(dis < 50)
                      {
                          //human_detected[id_dictionary_tmp.index[j]] = human_detected_last[id_dictionary_tmp.index[i]];
                          //human_detected[id_dictionary_tmp.index[j]].setId(i);
                          //human_detected[id_dictionary_tmp.index[j]].setTime(human_detected_last[id_dictionary_tmp.index[i]].time() + 1);
                          human_detected[id_dictionary_tmp.index[j]].inherit(human_detected_last[id_dictionary_tmp.index[i]]);
                          id_dictionary_tmp.index[i] = id_dictionary_tmp.index[j];
                          id_dictionary_tmp.status[j] = 0;
                          id_dictionary_tmp.index[j] = -1;
                          found = true;
                      }
                  }
              }
              if(!found)
              {
                  id_dictionary_tmp.index[i] = -1;
                  id_dictionary_tmp.status[i] = 0;
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
      id_dictionary = id_dictionary_tmp;

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
void ImageProcessor::identify(vector<KeyPoint_prob>& human_joints_imu, vector<KeyPoint_prob>& human_joints_result)
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
        int sub_length = 400;
        cv::Mat displyImg = image.clone();
        cv::Mat finalImg(540, 960 + sub_length, CV_8UC3);
        //draw_human_pose(displyImg, human_joints);
        for (int i = 0; i < human_detected.size(); ++i){
            Scalar color(0, 0, 0);
            double similarity;
            similarity = getPoseSimilarity(human_detected[i].pose(), pose_imu_2d);
            if(human_detected[i].punishment() < 1)
            {
              color = Scalar(0,0,150);
            }
            if(human_detected[i].id() == selected_id)
            {
              color = Scalar(255,0,0);
            }
            draw_human_pose(displyImg, human_detected[i].key_points(), color);
            if(similarity >= 0){
              human_detected[i].updateSimilarity(similarity);
            }

            auto position = human_detected[i].position();
            char text[100];
            //cord_text << "similarity:" << similarity;

            sprintf(text, "ID:%d", human_detected[i].id());
            putText(displyImg, text, Point(position.x + 40,position.y + 30), FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
            sprintf(text, "similarity: %1.3f", human_detected[i].similarity());
            putText(displyImg, text, Point(position.x + 40,position.y + 60), FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
            sprintf(text, "time: %d", human_detected[i].time());
            putText(displyImg, text, Point(position.x + 40,position.y + 90), FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
            sprintf(text, "confidence: %1.3f", human_detected[i].confidence());
            putText(displyImg, text, Point(position.x + 40,position.y + 120), FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
            sprintf(text, "buffer size: %d", int(human_detected[i].similarity_buffer().size()));
            putText(displyImg, text, Point(position.x + 40,position.y + 150), FONT_HERSHEY_SIMPLEX, 0.7, color, 2);

        }
        cv::Mat imuImg = Mat::zeros(540, sub_length, CV_8UC3);
        if(!human_joints_result.empty())
        {
            draw_human_pose(displyImg, human_joints_result, Scalar(0,255,0));
        }
        if(!human_joints_imu.empty())
        {
            draw_imu_pose(imuImg, human_joints_imu, Scalar(0,0,255));
        }
        Rect left_piece = Rect(0, 0, 960, 540);
        Rect right_piece = Rect(960, 0, sub_length, 540);
        displyImg.copyTo(finalImg(left_piece));
        imuImg.copyTo(finalImg(right_piece));
        cv::resize(finalImg, finalImg, cv::Size(finalImg.cols*0.7, finalImg.rows*0.7),0,0,INTER_LINEAR);

        char window_name[100];
        sprintf(window_name, "similarity_%d", camera_index_);
        cv::imshow(window_name, finalImg);
        cv::waitKey(30);
    }
}

void ImageProcessor::draw_imu_pose(Mat& image, const vector<KeyPoint_prob>& human_joints, Scalar color)
{
  int center_x = image.cols / 2;
  int diff = center_x - human_joints[0].x;

  circle(image, cv::Point(human_joints[0].x + diff, human_joints[0].y), 3, color, -1, 8);
  //draw joints on image
  for(int i = 1; i < (human_joints.size()); i++)
  {
      if(human_joints[i].p < 0.01)
          continue;
      circle(image, cv::Point(human_joints[i].x + diff, human_joints[i].y), 3, color, -1, 8);
      if (i != 4)
        line(image, cv::Point(human_joints[i-1].x + diff, human_joints[i-1].y), cv::Point(human_joints[i].x + diff, human_joints[i].y), color, 2);
      else
        line(image,cv::Point(human_joints[0].x + diff, human_joints[0].y), cv::Point(human_joints[i].x + diff, human_joints[i].y), color, 2);
  }
}

void ImageProcessor::draw_human_pose(Mat& image, const vector<KeyPoint_prob>& human_joints, Scalar color)
{
  if(human_joints[0].p < 0.01)
  {
    circle(image, cv::Point(human_joints[1].x, human_joints[1].y), 3, color, -1, 8);
    circle(image, cv::Point(human_joints[4].x, human_joints[2].y), 3, color, -1, 8);
    for(int i = 2; i < (human_joints.size()); i++)
    {
        if(human_joints[i].p < 0.01)
            continue;
        circle(image, cv::Point(human_joints[i].x, human_joints[i].y), 3, color, -1, 8);
        if (i != 4)
          line(image, cv::Point(human_joints[i-1].x, human_joints[i-1].y), cv::Point(human_joints[i].x, human_joints[i].y), color, 2);
        else
          continue;
    }
  }
  else
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
  }

    //ROS_INFO("Human Pose drawed");
}

void ImageProcessor::run()
{
    vector<KeyPoint_prob> key_point_imu;
    vector<KeyPoint_prob> key_point_result;
    calculateHumanPose(key_point_imu, camera_index_);
    calculateHumanPose(key_point_result, camera_index_, true);
    getPoseVector(key_point_imu, pose_imu_2d);
    //ROS_INFO("IMU Pose get!");
    identify(key_point_imu, key_point_result);
}

void publishSelectedKeypoints(ros::Publisher& pub, const vector<KeyPoint_prob>& keypoints_human_1, const vector<KeyPoint_prob>& keypoints_human_2)
{
    inertial_poser::UpperBodyKeyPoints keypoints_1;
    inertial_poser::UpperBodyKeyPoints keypoints_2;

    for(int i = 0; i < keypoints_human_1.size(); ++i)
    {
      keypoints_1.points[i].x = keypoints_human_1[i].x;
      keypoints_1.points[i].y = keypoints_human_1[i].y;
      keypoints_1.points[i].prob = keypoints_human_1[i].p;
    }

    for(int i = 0; i < keypoints_human_2.size(); ++i)
    {
      keypoints_2.points[i].x = keypoints_human_2[i].x;
      keypoints_2.points[i].y = keypoints_human_2[i].y;
      keypoints_2.points[i].prob = keypoints_human_2[i].p;
    }

    inertial_poser::ROI_Package roi_pack;
    roi_pack.packages.push_back(keypoints_1);
    roi_pack.packages.push_back(keypoints_2);
    roi_pack.header.stamp = ros::Time::now();
    pub.publish(roi_pack);
}

bool calculateHumanPose_3d(tf::TransformListener &robot_pose_listener, vector<string>& human_joint_names, vector<Point3f>& joint_pose_3d, bool result = false)
{
  //tf::TransformListener robot_pose_listener;
    //string human_reference_frame;
    joint_pose_3d.clear();
    string human_ns = "/human_imu";
    string result_human_ns = "/human_1";
    vector<Point3f> joint_3d_cords;

    vector<std::string> human_joint_names_this;
    if(result)
    {
        for(int i = 0; i < human_joint_names.size(); ++i)
        {
            string name_temp = result_human_ns + human_joint_names[i];
            human_joint_names_this.push_back(name_temp);
        }
    }
    else
    {
        for(int i = 0; i < human_joint_names.size(); ++i)
        {
            string name_temp = human_ns + human_joint_names[i];
            human_joint_names_this.push_back(name_temp);
        }
    }

    tf::StampedTransform joint_transforms;
    for(int i = 0; i < human_joint_names_this.size(); i++)
    {
        try
        {
            robot_pose_listener.lookupTransform("marker_0", human_joint_names_this[i], ros::Time(0), joint_transforms);
        }
        catch(tf::TransformException ex)
        {
            //ROS_ERROR("%s", ex.what());
            return false;
        }
        Point3f location(joint_transforms.getOrigin().x(), joint_transforms.getOrigin().y(), joint_transforms.getOrigin().z());
        joint_3d_cords.push_back(location);
    }

    Point3f vector_ite;
    vector_ite = joint_3d_cords[1] - joint_3d_cords[0];
    joint_pose_3d.push_back(vector_ite);
    vector_ite = joint_3d_cords[3] - joint_3d_cords[2];
    joint_pose_3d.push_back(vector_ite);
    vector_ite = joint_3d_cords[4] - joint_3d_cords[3];
    joint_pose_3d.push_back(vector_ite);
    vector_ite = joint_3d_cords[6] - joint_3d_cords[5];
    joint_pose_3d.push_back(vector_ite);
    vector_ite = joint_3d_cords[7] - joint_3d_cords[6];
    joint_pose_3d.push_back(vector_ite);
    return true;
}

double getVectorLength_3d(const Point3f& pose_vector){
    return sqrt(pose_vector.x * pose_vector.x + pose_vector.y * pose_vector.y + pose_vector.z * pose_vector.z);
}

double getVectorSimilarity_3d(const Point3f& pose_vector_1, const Point3f& pose_vector_2)
{
    double COS_Theta;
    COS_Theta = (pose_vector_1.x * pose_vector_2.x + pose_vector_2.y * pose_vector_1.y + pose_vector_2.z * pose_vector_1.z)/(getVectorLength_3d(pose_vector_1) * getVectorLength_3d(pose_vector_2));
    return (COS_Theta + 1.0) / 2.0;
}

double getPoseSimilarity_3d(const vector<Point3f>& pose_vectors_1, const vector<Point3f>& pose_vectors_2)
{
    int dimension = pose_vectors_1.size();
    if (dimension != pose_vectors_2.size())
    {
        return 0;
    }
    double simliarity = 0.0;
    int count = 0;

    for (int i = 0; i < dimension; ++i){
            count++;
            simliarity += getVectorSimilarity_3d(pose_vectors_1[i], pose_vectors_2[i]);
    }

    return simliarity / count;

}

int IndexInBuffer(int id_1, int id_2 ,const vector<human_linkage>& link_buf)
{
    for(int i = 0; i < link_buf.size(); ++i)
    {
      if(link_buf[i].isBetween(id_1, id_2))
      {
          return i;
      }
    }
    return -1;
}

int MaxInBuffer(int except_id, const vector<human_linkage>& link_buf)
{
    int count = link_buf.size();
    if (count > 1)
    {
      int max_id = except_id;
      double max_data = link_buf[except_id].confidence();
      for(int i = 0; i < link_buf.size(); ++i)
      {
        if(i != except_id && link_buf[i].confidence() > max_data)
        {
            max_data = link_buf[i].confidence();
            max_id = i;
        }
      }
    }
    else
    {
      return -1;
    }

}

int main(int argc, char** argv){

   vector<string> human_joint_names;
   human_joint_names.push_back("/neck");
   human_joint_names.push_back("/fetch_position");

   human_joint_names.push_back("/rArm");
   human_joint_names.push_back("/rForeArm");
   human_joint_names.push_back("/rWrist");
   human_joint_names.push_back("/lArm");
   human_joint_names.push_back("/lForeArm");
   human_joint_names.push_back("/lWrist");
   ros::init(argc, argv, "human_identifier");
   ImageProcessor identifier_camera_1(1);
   ImageProcessor identifier_camera_2(2);
   ros::NodeHandle nh_globle;
   ros::Publisher keypoints_pub = nh_globle.advertise<inertial_poser::ROI_Package>("/human_1/roi_package", 5);
   static tf::TransformListener human_pose_listener;
   ros::Rate rate(60);
   human human_selected_1;
   human human_selected_2;
   //vector<KeyPoint_prob> keypoints_1;
   //vector<KeyPoint_prob> keypoints_2;
   vector<Point3f> pose_3d_imu;
   vector<Point3f> pose_3d_result;

   vector<human_linkage> link_buf;

   int id_last_1 = -1;
   int id_last_2 = -1;

   while(ros::ok())
   {
      double conf_result = -1;
      ros::spinOnce();
      bool rc_imu = calculateHumanPose_3d(human_pose_listener, human_joint_names, pose_3d_imu);
      bool rc_result = calculateHumanPose_3d(human_pose_listener, human_joint_names, pose_3d_result, true);

      if(rc_imu && rc_result)
      {
          conf_result = getPoseSimilarity_3d(pose_3d_imu,pose_3d_result);
          if(id_last_1 > 0 && id_last_2 > 0)
          {
            int index = IndexInBuffer(id_last_1, id_last_2, link_buf);
            if(index < 0)
            {
                human_linkage link_tmp(id_last_1, id_last_2, conf_result);
                link_buf.push_back(link_tmp);
            }
            else
            {
              link_buf[index].updateSimilarity(conf_result);
            }
          }
      }

      ROS_INFO("result confidence: %.2f", conf_result);

      identifier_camera_1.run();
      identifier_camera_2.run();
      bool rc_1 = identifier_camera_1.getHuman_with_most_confidence(human_selected_1);
      bool rc_2 = identifier_camera_2.getHuman_with_most_confidence(human_selected_2);

      if(rc_1 && rc_2)
      {
        int index = IndexInBuffer(human_selected_1.id(), human_selected_2.id(), link_buf);
        if(index < 0)
        {
          id_last_1 = human_selected_1.id();
          id_last_2 = human_selected_2.id();
          publishSelectedKeypoints(keypoints_pub, human_selected_1.key_points(), human_selected_2.key_points());
        }
        else
        {
          if(link_buf[index].confidence() < 0.7)
          {
            int max_id = MaxInBuffer(index, link_buf);
            if(max_id >= 0)
            {
                if(human_selected_1.id() == link_buf[max_id].id_1())
                {
                  identifier_camera_2.punish(human_selected_2.id());
                }
                else if(human_selected_2.id() == link_buf[max_id].id_2())
                {
                  identifier_camera_1.punish(human_selected_1.id());
                }
                else
                {
                  if(human_selected_1.similarity() < human_selected_2.similarity())
                  {
                    identifier_camera_1.punish(human_selected_1.id());
                  }
                  else
                  {
                    identifier_camera_2.punish(human_selected_2.id());
                  }
                }
            }
            id_last_1 = human_selected_1.id();
            id_last_2 = human_selected_2.id();
          }
          else
          {
            publishSelectedKeypoints(keypoints_pub, human_selected_1.key_points(), human_selected_2.key_points());
            id_last_1 = human_selected_1.id();
            id_last_2 = human_selected_2.id();
          }

        }
      }

      rate.sleep();
   }
   return 0;
}
