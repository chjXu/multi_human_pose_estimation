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
#include <queue>
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



class ImageProcessor
{
  public:
    ImageProcessor(int camera_index, string nspace, string calib_path):it(nh), priv_nh("~"), sizeColor(960, 540) //left_key_points_center(480,260) ,right_key_points_center(480,260), left_ROI(240), right_ROI(240)
    {
        camera_index_ = camera_index;

        selected_id = -1;

        human_ns = "/human_imu";
        result_human_ns = "/human_1";

        joint_names.push_back("shoulder_link");
        joint_names.push_back("upper_arm_link");
        joint_names.push_back("forearm_left");
        joint_names.push_back("forearm_link");
        joint_names.push_back("wrist_1_link");
        joint_names.push_back("wrist_2_link");
        joint_names.push_back("wrist_3_link");


        human_joint_names.push_back("/hip");
        human_joint_names.push_back("/spine3");

        //human_joint_names.push_back("lShoulder");
        human_joint_names.push_back("/rArm");
        human_joint_names.push_back("/rForeArm");
        human_joint_names.push_back("/rWrist");
        //human_joint_names.push_back("rShoulder");
        human_joint_names.push_back("/lArm");
        human_joint_names.push_back("/lForeArm");
        human_joint_names.push_back("/lWrist");

        detected_key_points.reserve(10);
        string color_topic = nspace + "/qhd/image_color_rect";
        color = it.subscribe(color_topic.c_str(), 1,&ImageProcessor::imageCallback ,this);

        char key_points_topic[50];
        sprintf(key_points_topic, "/openpose_ros/human_list_%d", camera_index);

        human_keypoints_sub = nh.subscribe(key_points_topic, 1, &ImageProcessor::human_keypoints_callback, this);
        //human_keypoints_sub = nh.subscribe("/openpose_ros/human_list_2", 1, &ImageProcessor::human_keypoints_callback_2, this);
        loadCalibrationFiles(calib_path, cameraMatrix, distortion, 0.5);
        priv_nh.param("image_delay", image_delay, 2);
    }

    void run();
    bool getHuman_with_most_confidence(human& human_max);
    void punish(int id);



  private:
    string human_ns;
    string result_human_ns;
    int camera_index_;

    ros::NodeHandle nh;
    ros::NodeHandle priv_nh;
    image_transport::ImageTransport it;
    image_transport::Subscriber color;
    ros::Subscriber human_keypoints_sub;


    cv::Mat distortion;
    cv::Mat cameraMatrix;

    cv::Size sizeColor;

    int image_delay;

    std::queue<cv::Mat> image_queue;

    vector<vector<poseVector> > pose_vectors_image_all;

    vector<human> human_detected;
    vector<human> human_detected_last;
    vector<poseVector> pose_imu_2d;

    tf::TransformListener robot_pose_listener;
    tf::StampedTransform base_cam_transform;
    bool isCamBaseTransformAvailable;
    bool isCamHumanTransformAvailable;
    vector<std::string> joint_names;
    vector<std::string> human_joint_names;
    bool calibrationMode;
    bool firstDepth;


    vector<Point> human_joint_pos;

    vector<KeyPoint_prob> detected_key_points;

    int selected_id;


    tf::Transform robot_pose_tansform;
    void loadRobotPoseFile(string);

    bool keypoints_available;

    void imageCallback(const sensor_msgs::ImageConstPtr& msg);

    void initCalibration(double scale, int camera_index);

    void getImageCoordinate(Point3f& world_cord, KeyPoint_prob& image_cord, Mat& cameraMatrix);
    void getImageCoordinate(Point3f& world_cord, Point& image_cord, Mat& cameraMatrix);
    void loadCalibrationFiles(string& calib_path, cv::Mat& cameraMatrix, cv::Mat& distortion, double scale);
    void calculateHumanPose(vector<KeyPoint_prob>& joint_image_cords, int camera_index, bool);
    void calculateRobotPose(vector<Point>& joint_image_cords, int camera_index);
    bool getPoseVector(vector<KeyPoint_prob>& key_points, vector<poseVector>& pose_vectors);
    double getVectorLength(const poseVector& pose_vector);

    double getVectorSimilarity(const poseVector& pose_vector_1, const poseVector& pose_vector_2);

    double getPoseSimilarity(const vector<poseVector>& pose_vectors_1, const vector<poseVector>& pose_vectors_2);

    void human_keypoints_callback(openpose_ros_msgs::OpenPoseHumanList keypoints);

    void identify(vector<KeyPoint_prob>&, vector<Point>&);

    void draw_human_pose(Mat& image,const vector<KeyPoint_prob>& human_joints, Scalar color = Scalar(0, 0, 255));
    void draw_imu_pose(Mat& image, const vector<KeyPoint_prob>& human_joints,  Scalar color = Scalar(0, 0, 255));
    void draw_key_points(Mat& image, const vector<KeyPoint_prob>&,  Scalar color = Scalar(0, 0, 255));
    void drawRobotJoints(Mat& image, vector<Point>& joint_image_cords, Scalar color = Scalar(0, 0, 255));
    double distance(const KeyPoint_prob key_point_1, const KeyPoint_prob key_point_2);
    double getKeyPointsSimilarity(const vector<KeyPoint_prob>& key_points_1, const vector<KeyPoint_prob>& key_points_2);



};

void ImageProcessor::human_keypoints_callback(openpose_ros_msgs::OpenPoseHumanList keypoints)
{
      detected_key_points.clear();
      int person_num = keypoints.num_humans;
      if (person_num > 0)
      {
        int person_index = 0;
        for(int person = 0;person < person_num; ++person)
        {
          auto body_keypoints = keypoints.human_list[person].body_key_points_with_prob;
          KeyPoint_prob key_point_ite;

          for(int i = 2; i < 9; i++)
          {
            if(body_keypoints[i].prob > 0.0)
            {

              key_point_ite.x = body_keypoints[i].x;
              key_point_ite.y = body_keypoints[i].y;
              key_point_ite.p = body_keypoints[i].prob;
              detected_key_points.push_back(key_point_ite);
            }
          }
        }
      }
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

      Mat image = color_mat.clone();
      image_queue.push(image);

      if (image_queue.size()>image_delay)
      {
        image_queue.pop();
      }
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

void ImageProcessor::getImageCoordinate(Point3f& world_cord, Point& image_cord, Mat& cameraMatrix)
{
    double fx = cameraMatrix.at<double>(0, 0);
    double fy = cameraMatrix.at<double>(1, 1);
    double cx = cameraMatrix.at<double>(0, 2);
    double cy = cameraMatrix.at<double>(1, 2);

    image_cord.x = (int)(world_cord.x * fx / world_cord.z + cx);
    image_cord.y = (int)(world_cord.y * fy / world_cord.z + cy);
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

void ImageProcessor::identify(vector<KeyPoint_prob>& human_joints_result, vector<Point>& robot_joint)
{

          if(image_queue.size() == image_delay)
          {
            cv::Mat displyImg = (image_queue.front()).clone();
            if(!robot_joint.empty())
            {
                drawRobotJoints(displyImg, robot_joint, Scalar(0, 255, 0));
            }
            if(!human_joints_result.empty())
            {
                draw_human_pose(displyImg, human_joints_result, Scalar(0,0,255));
            }


            draw_key_points(displyImg, detected_key_points, Scalar(255,0,0));

            char window_name[100];
            sprintf(window_name, "similarity_%d", camera_index_);
            cv::imshow(window_name, displyImg);
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
void ImageProcessor::draw_key_points(Mat& image, const vector<KeyPoint_prob>& key_points, Scalar color)
{
    for(int i = 0; i < key_points.size(); ++i)
    {
        if(key_points[i].p > 0){
            int scale = 10.0 * key_points[i].p;
            circle(image, cv::Point((int)key_points[i].x, (int)key_points[i].y), scale, color, 3, 8);
        }

    }
}
void ImageProcessor::drawRobotJoints(Mat& image, vector<Point>& joint_image_cords, Scalar color)
{

    //cout << "x: " << joint_image_cords[0].x << "\n";
    //cout << "y: " << joint_image_cords[0].y << "\n";
    circle(image, joint_image_cords[0], 3, color, -1, 8);
    //draw joints on image
    for(int i = 1; i < (joint_image_cords.size()); i++)
    {
        circle(image, joint_image_cords[i], 3, color, -1, 8);
        line(image,joint_image_cords[i-1],joint_image_cords[i], color, 2);
    }
}

void ImageProcessor::draw_human_pose(Mat& image, const vector<KeyPoint_prob>& human_joints, Scalar color)
{
  {
    circle(image, cv::Point(human_joints[0].x, human_joints[0].y), 3, color, -1, 8);

    circle(image, cv::Point(human_joints[1].x, human_joints[1].y), 3, color, -1, 8);

    line(image, cv::Point(human_joints[1].x, human_joints[1].y), cv::Point(human_joints[0].x, human_joints[0].y), color, 2);
    //draw joints on image
    for(int i = 1; i < (human_joints.size()); i++)
    {
        if(human_joints[i].p < 0.01)
            continue;
        circle(image, cv::Point(human_joints[i].x, human_joints[i].y), 3, color, -1, 8);
        if (i != 5)
          line(image, cv::Point(human_joints[i-1].x, human_joints[i-1].y), cv::Point(human_joints[i].x, human_joints[i].y), color, 2);
        else
          line(image,cv::Point(human_joints[1].x, human_joints[1].y), cv::Point(human_joints[i].x, human_joints[i].y), color, 2);
    }
  }

    //ROS_INFO("Human Pose drawed");
}

void ImageProcessor::calculateRobotPose(vector<Point>& joint_image_cords, int camera_index)
{
  //tf::TransformListener robot_pose_listener;
    char robot_reference_frame[30];
    sprintf(robot_reference_frame, "camera_base_%d", camera_index);


    tf::StampedTransform joint_transforms;
    tf::StampedTransform cam_base_transform;
    try
    {
        robot_pose_listener.lookupTransform(robot_reference_frame, "base_link", ros::Time(0), cam_base_transform);
    }

    catch(tf::TransformException ex)
    {
        ROS_ERROR("%s", ex.what());
        isCamBaseTransformAvailable = false;
        return;
    }

    isCamBaseTransformAvailable = true;
    Point3f base_location(cam_base_transform.getOrigin().x(), cam_base_transform.getOrigin().y(), cam_base_transform.getOrigin().z());
    Point base_image_cord;
    getImageCoordinate(base_location, base_image_cord, cameraMatrix);

    //vector<Point> joint_image_cords;
    joint_image_cords.push_back(base_image_cord);
    for(int i = 0; i < joint_names.size(); i++)
    {
        try
        {
            robot_pose_listener.lookupTransform(robot_reference_frame, joint_names[i], ros::Time(0), joint_transforms);
        }
        catch(tf::TransformException ex)
        {
            ROS_ERROR("%s", ex.what());
            continue;
        }
        Point3f location(joint_transforms.getOrigin().x(), joint_transforms.getOrigin().y(), joint_transforms.getOrigin().z());
        Point joint_image_cord;
        getImageCoordinate(location, joint_image_cord, cameraMatrix);

        joint_image_cords.push_back(joint_image_cord);
        //ROS_INFO("Robot Pose Get!");
    }
}
void ImageProcessor::run()
{
    vector<KeyPoint_prob> key_point_result;
    vector<Point> robot_joint;
    calculateHumanPose(key_point_result, camera_index_, true);
    calculateRobotPose(robot_joint, camera_index_);
    identify(key_point_result, robot_joint);
}



int main(int argc, char** argv){

   string calib_path_1 = "/home/agent/catkin_ws/src/iai_kinect2/kinect2_bridge/data/003415165047";
   string calib_path_2 = "/home/agent/catkin_ws/src/iai_kinect2/kinect2_bridge/data/092465240847";
   string calib_path_3 = "/home/agent/catkin_ws/src/cv_camera/calibration_data";
   ros::init(argc, argv, "human_display");
   ImageProcessor identifier_camera_1(1, "/kinect2_1", calib_path_1);
   ImageProcessor identifier_camera_2(2, "/kinect2_2", calib_path_2);
   ImageProcessor identifier_camera_3(3, "/camera_3", calib_path_3);
   static tf::TransformListener human_pose_listener;
   ros::Rate rate(60);

   while(ros::ok())
   {
      ros::spinOnce();
      identifier_camera_1.run();
      identifier_camera_2.run();
      identifier_camera_3.run();
      rate.sleep();
   }
   return 0;
}
