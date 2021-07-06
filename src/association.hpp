#ifndef ASSOCIATION__H
#define ASSOCIATION__H

/***************
 * 在姿态匹配过程中：
 * 1.首先，要把每台相机中的姿态进行捕获，传到PerFrameAssociation类中
 * 2.建立不同相机中不同姿态之间的置信度关系，也就是得到deta的值
 * 3.根据deta计算r，并将r进行排序，取最小值并返回姿态的索引值
 * 4.把返回索引值的姿态进行优化计算得到3维点，设置label
 * 5.更新传入到类中的姿态信息，循环计算，直到姿态全部匹配完成或一台相机的姿态匹配完或
***************/
#include "pose.hpp"
#include "PosePair.hpp"
#include <ros/ros.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <chrono>
#include <pthread.h>

using namespace cv;
using namespace std;
using namespace ceres;




/************************非线性优化函数**********************************/
struct CostFunction_cam_2d_2d {//仿函数
  	CostFunction_cam_2d_2d(Point2d _observe_point1, Point2d _observe_point2,
	  				  Eigen::Matrix3d _camera_rot_1, Eigen::Matrix3d _camera_rot_2,
					  Eigen::Matrix<double,3,1> _camera_trans_1, Eigen::Matrix<double,3,1> _camera_trans_2,
					  vector<double> camera_params_1, vector<double> camera_params_2)
	   :observe_point1(_observe_point1),observe_point2(_observe_point2),
	  	camera_rot_1(_camera_rot_1), camera_rot_2(_camera_rot_2), camera_trans_1(_camera_trans_1), camera_trans_2(_camera_trans_2),
        fx_1(camera_params_1[0]), fy_1(camera_params_1[1]), cx_1(camera_params_1[2]), cy_1(camera_params_1[3]),
        fx_2(camera_params_1[0]), fy_2(camera_params_1[1]), cx_2(camera_params_1[2]), cy_2(camera_params_1[3]){}

          // 残差的计算
     template <typename T>
	  bool operator()(const T *const depths, T *residual) const {

		Eigen::Matrix<T,3,1> Cam_point1, Cam_point2;
		Eigen::Matrix<T,3,1> World_point1, World_point2;

			Cam_point1(2,0) = depths[0];
			Cam_point2(2,0) = depths[1];
			Cam_point1(0,0) = (T(observe_point1.x) - cx_1) * Cam_point1(2,0) / fx_1;
			Cam_point1(1,0) = (T(observe_point1.y) - cy_1) * Cam_point1(2,0) / fy_1;
			Cam_point2(0,0) = (T(observe_point2.x) - cx_2) * Cam_point2(2,0) / fx_2;
			Cam_point2(1,0) = (T(observe_point2.y) - cy_2) * Cam_point2(2,0) / fy_2;

			World_point1 = camera_rot_1.cast<T>() * Cam_point1 + camera_trans_1.cast<T>();
			World_point2 = camera_rot_2.cast<T>() * Cam_point2 + camera_trans_2.cast<T>();

    		residual[0] = ((World_point1(0,0) - World_point2(0,0)) * (World_point1(0,0) - World_point2(0,0))
						+(World_point1(1,0) - World_point2(1,0)) * (World_point1(1,0) - World_point2(1,0))
						+(World_point1(2,0) - World_point2(2,0)) * (World_point1(2,0) - World_point2(2,0)));


    	return true;

  	}

	Point3d observe_point1;
	Point3d observe_point2;
	Eigen::Matrix3d camera_rot_1, camera_rot_2;
	Eigen::Matrix<double,3,1> camera_trans_1, camera_trans_2;

	double fx_1,fy_1,cx_1,cy_1;
	double fx_2,fy_2,cx_2,cy_2;

};



struct CostFunction_cam_2d_3d {
  	CostFunction_cam_2d_3d(Point2d _observe_point1, Point3d _observe_point2,
	  				  Eigen::Matrix3d _camera_rot,
					  Eigen::Matrix<double,3,1> _camera_trans,
					  vector<double> camera_params)
	   :observe_point1(_observe_point1),observe_point2(_observe_point2),
	  	camera_rot(_camera_rot),camera_trans(_camera_trans),
        fx(camera_params[0]),fy(camera_params[1]),cx(camera_params[2]),cy(camera_params[3]){}

// 残差的计算
  	template <typename T>
	  bool operator()(const T *const depths, T *residual) const {

		Eigen::Matrix<T,3,1> Cam_point1;
		Eigen::Matrix<T,3,1> World_point1;


			Cam_point1(2,0) = depths[0];
			Cam_point1(0,0) = (T(observe_point1.x) - cx) * Cam_point1(2,0) / fx;
			Cam_point1(1,0) = (T(observe_point1.y) - cy) * Cam_point1(2,0) / fy;

			World_point1 = camera_rot.cast<T>() * Cam_point1 + camera_trans.cast<T>();

    		residual[0] = ((World_point1(0,0) - observe_point2.x) * (World_point1(0,0) - observe_point2.x)
						+(World_point1(1,0) - observe_point2.y) * (World_point1(1,0) - observe_point2.y)
						+(World_point1(2,0) - observe_point2.z) * (World_point1(2,0) - observe_point2.z));



    	return true;

  	}

	Point3d observe_point1;
	Point3d observe_point2;
	Eigen::Matrix3d camera_rot;
	Eigen::Matrix<double,3,1> camera_trans;

	double fx,fy,cx,cy;

};

struct CostFunction_3d_loc {
    CostFunction_3d_loc(
                      Eigen::Matrix3d& _camera_rot,
                      Eigen::Matrix<double,3,1>& _camera_trans,
                      vector<double>& camera_params,
                      KeyPoint_prob& _point_ref)
                      :camera_trans(_camera_trans), point_ref(_point_ref),
                      fx(camera_params[0]),fy(camera_params[1]),cx(camera_params[2]),cy(camera_params[3]),
                      camera_rot(_camera_rot.transpose()){}


    template <typename T>
      bool operator()(const T *const point, T *residual) const {

        Eigen::Map<const Eigen::Matrix<T, 3, 1> > eigen_point(point);
        Eigen::Matrix<T,3,1> cam_point;

        cam_point = camera_rot.cast<T>() * (eigen_point - camera_trans.cast<T>());

        T x = cam_point(0,0) * (T)fx / cam_point(2,0) + (T)cx;
        T y = cam_point(1,0) * (T)fy / cam_point(2,0) + (T)cy;
        //cout << "Cmaera2 is" << cam_point(2,0) << endl;
        //cout << "Cmaera0 is" << cam_point(0,0) << endl;
        //cout << "Cmaera1 is" << cam_point(1,0) << endl;

        T x_diff, y_diff;
        x_diff = x - (T)point_ref.x;
        y_diff = y - (T)point_ref.y;


        residual[0] = (x_diff * x_diff + y_diff * y_diff) * (T)point_ref.p;

        return true;

    }
      KeyPoint_prob point_ref;
      Eigen::Matrix3d camera_rot;
	  Eigen::Matrix<double,3,1> camera_trans;

	  double fx,fy,cx,cy;
};

class PerFrameAssociation
{
private:
    /*****************相机内参**********************/
     vector<vector<double> > cam_param;
	 vector<Eigen::Matrix3d> camera_rot;
	 vector<Eigen::Matrix<double, 3, 1>> camera_trans;
     bool check_;

public:
    list<PosePair> pose_pair;

    vector<Pose> AvailablePose;
public:
    PerFrameAssociation(bool check) {
      check_ = check;
      options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
      options.minimizer_progress_to_stdout = true;
      options.logging_type = ceres::SILENT;

      options_3d.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
      options_3d.minimizer_progress_to_stdout = true;
      options_3d.logging_type = ceres::SILENT;
      pose_mutex = PTHREAD_MUTEX_INITIALIZER;
    }

    void addCamParam(vector<double> cameraParam);

    void addCamPose(Eigen::Matrix3d& cam_rot, Eigen::Matrix<double, 3, 1>& cam_trans);

    void clearPoses() {
        AvailablePose.clear();
    }
    void addAvailablePose(vector<Pose> &availablePose);

	bool getFullCorrespondence(PosePair& pair, double threshold = 0.4);

    double optimizeWithCeres2D_2D(int cam_1, int cam_2, Point2d point_1, Point2d point_2, Point3d& point_ret);

    double optimizeWithCeres2D_3D(int cam, Point2d point_1, Point3d point_2, Point3d& point_ret);

    void optimize3DLoc(vector<int>& as_id, vector<KeyPoint_3d>& pose_joints_3d);

    double getSpacedistance(cv::Point3d cam_a_point, cv::Point3d cam_b_point);

    vector<vector<KeyPoint_3d>> extract3DPoses();

    void generatePair();

    void associate();

    vector<vector<int>> extract2DAssociation();

	ceres::Solver::Options options;

    ceres::Solver::Options options_3d;

    pthread_mutex_t pose_mutex;

};

void PerFrameAssociation::addCamParam(vector<double> cameraParam) {
    cam_param.push_back(cameraParam);
}
void PerFrameAssociation::addCamPose(Eigen::Matrix3d& cam_rot, Eigen::Matrix<double, 3, 1>& cam_trans) {
    camera_rot.push_back(cam_rot);
    camera_trans.push_back(cam_trans);
}

void PerFrameAssociation::addAvailablePose(vector<Pose> &availablePose)
{
    if(availablePose.empty()) {
        return;
    }

	for(int i = 0; i < availablePose.size(); ++i) {
        AvailablePose.push_back(availablePose[i]);
    }
    //std::cout << "Pose size:" << AvailablePose.size() << endl;
}

void PerFrameAssociation::optimize3DLoc(vector<int>& as_id, vector<KeyPoint_3d>& pose_joints_3d) {

    for(int p = 0; p < pose_joints_3d.size(); ++p) {
        double point[3] = {0.0, 0.0, 0.0};
        ceres::Problem problem;
        int count = 0;
        for(int i = 0; i < as_id.size(); ++i) {

            double prob = AvailablePose[as_id[i]].pose_joints[p].p;
            //ROS_INFO("prob:%.2f", prob);
            if (prob > 0) {

                count++;
                int cam = AvailablePose[as_id[i]].camera_index-1;
                problem.AddResidualBlock(     // 向问题中添加误差项
                      // 使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
                    new ceres::AutoDiffCostFunction<CostFunction_3d_loc, 1, 3>(
                        new CostFunction_3d_loc(camera_rot[cam], camera_trans[cam], cam_param[cam], AvailablePose[as_id[i]].pose_joints[p])
                    ),
                    nullptr,            // 核函数，这里不使用，为空
                    point               // 待估计参数
                    );
            }
        }
        if(count >= 2) {
            ceres::Solver::Summary summary;
    		ceres::Solve(options, &problem, &summary);
            //cout << summary.BriefReport() << endl;

            pose_joints_3d[p].available = true;
            pose_joints_3d[p].x = point[0];
            pose_joints_3d[p].y = point[1];
            pose_joints_3d[p].z = point[2];
        }
        else {
            //ROS_INFO("no enough angle");
            pose_joints_3d[p].available = false;
        }
    }
}



/*************************************计算r的值*********************************/
void PerFrameAssociation::generatePair() {
    pose_pair.clear();
    //ROS_INFO("Total_pose: %d", (int)AvailablePose.size());
    if(AvailablePose.empty()) {
        return;
    }

    int end[3] = {0,0,0};
    //int end[2] = {0,0};
    int i = 0;
    for(int c = 0; c < 3;++c) {
        int c_id = c+1;
        end[c] = i;
        while(i < AvailablePose.size() && AvailablePose[i].getCameraId() == c_id) {
            AvailablePose[i].label = -(i + 1);
            AvailablePose[i].as[0] = c;
            AvailablePose[i].as_id[0] = i;
            end[c] = i+1;
            ++i;
        }
    }
    ROS_INFO("image 1: %d image 2: %d image 3: %d\n", end[0], end[1] - end[0], end[2] - end[1]);
    //ROS_INFO("image 1: %d image 2: %d\n", end[0], end[1] - end[0]);
    //group 0
    for(int i = 0;i < end[0]; ++i) {
        for(int j = end[0];j < end[1]; ++j) {
            PosePair pair;
            pair.group = 0;
            pair.label_1 = AvailablePose[i].getLabel();
            pair.label_2 = AvailablePose[j].getLabel();
            pair.index_1 = i;
            pair.index_2 = j;
            pair.cam_1 = 0;
            pair.cam_2 = 1;
            pose_pair.push_back(pair);
        }
    }

    //group 1
    for(int i = 0;i < end[0]; ++i) {
        for(int j = end[1];j < end[2]; ++j) {
            PosePair pair;
            pair.group = 1;
            pair.label_1 = AvailablePose[i].getLabel();
            pair.label_2 = AvailablePose[j].getLabel();
            pair.index_1 = i;
            pair.index_2 = j;
            pair.cam_1 = 0;
            pair.cam_2 = 2;
            pose_pair.push_back(pair);
        }
    }

    //group 2
    for(int i = end[0];i < end[1]; ++i) {
        for(int j = end[1];j < end[2]; ++j) {
            PosePair pair;
            pair.group = 2;
            pair.label_1 = AvailablePose[i].getLabel();
            pair.label_2 = AvailablePose[j].getLabel();
            pair.index_1 = i;
            pair.index_2 = j;
            pair.cam_1 = 1;
            pair.cam_2 = 2;
            pose_pair.push_back(pair);
        }
    }
    //ROS_INFO("Total_pose pair: %d", (int)pose_pair.size());
}

struct crp_args{
    PerFrameAssociation* obj;
    list<PosePair>::iterator pair;
};

void* Correspondence_thread( void* args) {
    crp_args* m_args = (crp_args*)args;
    m_args->obj->getFullCorrespondence(*(m_args->pair));
}

void PerFrameAssociation::associate() {

    if(pose_pair.empty()){
        return;
    }
    // 不使用多线程
    // for(auto it=pose_pair.begin(); it!=pose_pair.end(); ++it){
    //     getFullCorrespondence(*it);
    // }
    pthread_t crp_thread[pose_pair.size()];
    int ret;
    int i = 0;
    crp_args args[pose_pair.size()];

    for(auto it = pose_pair.begin(); it != pose_pair.end(); ++it) {
        args[i] = {this, it};
        ret = pthread_create( &crp_thread[i], NULL, Correspondence_thread, &(args[i]));
        if(ret == 0){
            //ROS_INFO("Thread can't create!");
        }
        ++i;
    }

    for(int i = 0; i < pose_pair.size(); ++i) {
        pthread_join(crp_thread[i], NULL);
    }

    auto rank_min = min_element(pose_pair.begin(), pose_pair.end(), PosePair::comp);
    int label_ite = 1;
    double threshold  = 0.4;

    //auto t1 = chrono::steady_clock::now();
    while(!pose_pair.empty() && rank_min->delta <= threshold) {
        double delta_min = rank_min->delta;

        //ROS_INFO("iteration: %d", label_ite);
        /*
        for(int i = 0; i < AvailablePose.size(); ++i) {
            ROS_INFO("pose %d: %d", i, AvailablePose[i].label);
        }*/

        //ROS_INFO("pairs lefted: %d", (int)pose_pair.size());
        //int pair_id = 1;
        //for(auto it = pose_pair.begin(); it != pose_pair.end(); ++it) {
            //ROS_INFO("pair %d: g:%d, l%d:%d, i%d:%d, r:%.2f", pair_id, (*it).group, (*it).label_1, (*it).label_2, (*it).index_1, (*it).index_2, (*it).delta);
            //++pair_id;
        //}
        list<PosePair> pair_backup = pose_pair;
        vector<Pose> pose_backup = AvailablePose;


        int label_1_old = (*rank_min).label_1;
        int label_2_old = (*rank_min).label_2;
        int id_1 = (*rank_min).index_1;
        int id_2 = (*rank_min).index_2;
        int group = (*rank_min).group;
        //update Pose Label and pose with new 3D location
        vector<int> as_new;
        vector<int> as_id_new;
        int num_as_1 = AvailablePose[id_1].num_as;
        int num_as_2 = AvailablePose[id_2].num_as;
        int num_as_new = num_as_1 + num_as_2;
        as_new.resize(num_as_new);
        as_id_new.resize(num_as_new);
        for(int i = 0; i < num_as_1; ++i) {
            as_new[i] = AvailablePose[id_1].as[i];
            as_id_new[i] = AvailablePose[id_1].as_id[i];
        }
        for(int i = 0; i < num_as_2; ++i) {
            as_new[i+num_as_1] = AvailablePose[id_2].as[i];
            as_id_new[i+num_as_1] = AvailablePose[id_2].as_id[i];
        }

        //calculate new 3d location
        vector<KeyPoint_3d> pose_joints_3d;
        pose_joints_3d.resize(9);
        optimize3DLoc(as_id_new, pose_joints_3d);

        for(auto it = AvailablePose.begin(); it != AvailablePose.end(); ++it) {
            if((*it).label == label_1_old || (*it).label == label_2_old) {
                (*it).label = label_ite;
                (*it).num_as = num_as_new;
                (*it).pose_joints_3d = pose_joints_3d;
                (*it).as = as_new;
                (*it).as_id = as_id_new;
            }
        }

        //ROS_INFO("num_as: %d", num_as_new);

        pose_pair.erase(rank_min);
        //update Pair label and erase impossiple pairs
        for(auto it = pose_pair.begin();it != pose_pair.end();) {
            //camera_count
            if((*it).label_1 == label_1_old || (*it).label_2 == label_2_old || (*it).label_1 == label_2_old || (*it).label_2 == label_1_old) {
                if((*it).group == group || num_as_new >= 3) {
                    //ROS_INFO("pair: %d:%d erased in 1", (*it).index_1, (*it).index_2);
                    it = pose_pair.erase(it);
                    //continue;
                }
                else {
                    vector<int> &as_1 = AvailablePose[(*it).index_1].as;
                    vector<int> &as_2 = AvailablePose[(*it).index_2].as;
                    bool erase = false;
                    for(int i = 0; i < as_1.size(); ++i){
                        for(int j = 0; j < as_2.size(); ++j) {
                            if (as_1[i] == as_2[j]) {
                                erase = true;
                                //break;
                            }
                        }
                    }
                    if(erase) {
                        //ROS_INFO("pair: %d:%d erased in c", (*it).index_1, (*it).index_2);
                        it = pose_pair.erase(it);


                    }
                    else {
                        if((*it).label_1 == label_1_old || (*it).label_1 == label_2_old) {
                            (*it).label_1 = label_ite;
                        }
                        if((*it).label_2 == label_1_old || (*it).label_2 == label_2_old) {
                            int temp = (*it).label_1;
                            (*it).label_1 = label_ite;
                            (*it).label_2 = temp;
                            temp = (*it).index_1;
                            (*it).index_1 = (*it).index_2;
                            (*it).index_2 = temp;
                            temp = (*it).cam_1;
                            (*it).cam_1 = (*it).cam_2;
                            (*it).cam_2 = temp;
                        }
                        ++it;
                    }
                }
            }
            else {
                ++it;
            }

        }
        //update correspondence and erase duplicated pairs
        bool pass = false;
        for(auto it = pose_pair.begin();it != pose_pair.end();++it){
            if((*it).label_1 == label_ite) {
                pass = getFullCorrespondence(*it, threshold);
                int label = (*it).label_2;
                auto it_2 = it;
                ++it_2;
                for(;it_2 != pose_pair.end();) {
                    if((*it_2).label_1 == label_ite && (*it_2).label_2 == label) {
                        //ROS_INFO("pair: %d:%d erased in 2", (*it_2).index_1, (*it_2).index_2);
                        it_2 = pose_pair.erase(it_2);
                    }
                    else {
                        ++it_2;
                    }
                }
            }
        }
        //if not pass, return to iteration -1
        if(check_ && !pass && num_as_new < 3 && delta_min != threshold) {
            //ROS_INFO("iteration cannot pass, pairs recovered");
            pose_pair = pair_backup;
            AvailablePose = pose_backup;
            auto erase_min = min_element(pose_pair.begin(), pose_pair.end(), PosePair::comp);
            //pose_pair.erase(erase_min);
            (*erase_min).delta = threshold;
        }

        rank_min = min_element(pose_pair.begin(), pose_pair.end(), PosePair::comp);
        ++label_ite;
    }
    //auto t2 = chrono::steady_clock::now();
    //cout << "循环时间 :" << chrono::duration_cast<chrono::microseconds>(t2-t1).count() << "ms" << endl;
    //ROS_INFO("association over");
}

vector<vector<KeyPoint_3d>> PerFrameAssociation::extract3DPoses() {
    map<int, int> PoseList;
    vector<vector<KeyPoint_3d> > pose_3d;
    for(auto it = AvailablePose.begin(); it != AvailablePose.end(); ++it) {
        //ROS_INFO("label: %d\n", (*it).label);
        if((*it).label > 0 && PoseList.find((*it).label) == PoseList.end()){
            PoseList[(*it).label] = 0;
            pose_3d.push_back((*it).pose_joints_3d);
        }
    }

    // ROS_INFO("Pose generated: %d", (int)pose_3d.size());
    return pose_3d;

}


vector<vector<int>> PerFrameAssociation::extract2DAssociation() {
    map<int, int> PoseList;
    vector<vector<int> > pose_ass;
    int id_ite = 0;
    for(int i = 0; i < AvailablePose.size(); ++i) {
        if(AvailablePose[i].label > 0) {
            int label = AvailablePose[i].label;
            if (PoseList.find(label) == PoseList.end()) {
                PoseList[label] = id_ite;
                pose_ass.resize(id_ite+1);
                pose_ass[id_ite].push_back(i);
                id_ite++;
            }
            else {
                pose_ass[PoseList[label]].push_back(i);
            }
        }
    }

    //ROS_INFO("Pose generated: %d", (int)pose_ass.size());
    return pose_ass;

}

/***************************************计算deta**************************************/
bool PerFrameAssociation::getFullCorrespondence(PosePair& pair, double threshold)
{
    int cam_1, cam_2;
    cam_1 = pair.cam_1;
    cam_2 = pair.cam_2;

    //ROS_INFO("solving dis p%d:%d  c%d:%d", pair.index_1, pair.index_2, cam_1, cam_2);

    pthread_mutex_lock(&pose_mutex);
    //ROS_INFO("p%d:%d  c%d:%d gets the lock", pair.index_1, pair.index_2, cam_1, cam_2);
    Pose pose_1 = AvailablePose[pair.index_1];
    Pose pose_2 = AvailablePose[pair.index_2];
    pthread_mutex_unlock(&pose_mutex);
    //ROS_INFO("p%d:%d  c%d:%d release the lock", pair.index_1, pair.index_2, cam_1, cam_2);

	int count = 0;
	double sum_EX = 0.0;
    for(int i = 0; i < 17; ++i) {
        //if(i == 1 || i == 8){
            Point3d point_ret;
            double delta = 0.0;
            if(pose_1.pose_joints_3d[i].available && pose_2.pose_joints_3d[i].available) {
                ++count;
                delta = getSpacedistance(Point3d(pose_1.pose_joints_3d[i].x, pose_1.pose_joints_3d[i].y, pose_1.pose_joints_3d[i].z),
                                     Point3d(pose_2.pose_joints_3d[i].x, pose_2.pose_joints_3d[i].y, pose_2.pose_joints_3d[i].z));
                point_ret = Point3d((pose_1.pose_joints_3d[i].x + pose_2.pose_joints_3d[i].x) / 2,
                                (pose_1.pose_joints_3d[i].y + pose_2.pose_joints_3d[i].y) / 2,
                                (pose_1.pose_joints_3d[i].z + pose_2.pose_joints_3d[i].z) / 2);
                pair.pose_joints_3d[i].available = true;
                pair.pose_joints_3d[i].x = point_ret.x;
                pair.pose_joints_3d[i].y = point_ret.y;
                pair.pose_joints_3d[i].z = point_ret.z;

            }
            else if(pose_2.pose_joints_3d[i].available && pose_1.pose_joints[i].p > 0) {
                ++count;
                delta = optimizeWithCeres2D_3D(cam_1, Point2d(pose_1.pose_joints[i].x, pose_1.pose_joints[i].y),
                                                  Point3d(pose_2.pose_joints_3d[i].x, pose_2.pose_joints_3d[i].y,pose_2.pose_joints_3d[i].z),
                                                  point_ret);
                delta = delta / pose_1.pose_joints[i].p;

                pair.pose_joints_3d[i].available = true;
                pair.pose_joints_3d[i].x = point_ret.x;
                pair.pose_joints_3d[i].y = point_ret.y;
                pair.pose_joints_3d[i].z = point_ret.z;
            }
            else if(pose_1.pose_joints_3d[i].available && pose_2.pose_joints[i].p > 0) {
                ++count;
                delta = optimizeWithCeres2D_3D(cam_2, Point2d(pose_2.pose_joints[i].x, pose_2.pose_joints[i].y),
                                                  Point3d(pose_1.pose_joints_3d[i].x, pose_1.pose_joints_3d[i].y,pose_1.pose_joints_3d[i].z),
                                                  point_ret);
                delta = delta * pose_2.pose_joints[i].p;

                pair.pose_joints_3d[i].available = true;
                pair.pose_joints_3d[i].x = point_ret.x;
                pair.pose_joints_3d[i].y = point_ret.y;
                pair.pose_joints_3d[i].z = point_ret.z;

            }
            else if(pose_1.pose_joints[i].p > 0 && pose_2.pose_joints[i].p > 0) {
                ++count;
                delta = optimizeWithCeres2D_2D(cam_1, cam_2, Point2d(pose_1.pose_joints[i].x, pose_1.pose_joints[i].y),
                                                         Point2d(pose_2.pose_joints[i].x, pose_2.pose_joints[i].y),
                                                         point_ret);
                //delta = delta / std::sqrt(pose_1.pose_joints[i].p * pose_2.pose_joints[i].p);
                double p_min = (pose_1.pose_joints[i].p < pose_2.pose_joints[i].p) ? pose_1.pose_joints[i].p:pose_2.pose_joints[i].p;
                delta = delta * p_min;

                pair.pose_joints_3d[i].available = true;
                pair.pose_joints_3d[i].x = point_ret.x;
                pair.pose_joints_3d[i].y = point_ret.y;
                pair.pose_joints_3d[i].z = point_ret.z;
            }
            else {
                continue;
            }
            //ROS_INFO("delta point%d: %.3f", i, delta);
            sum_EX += delta;
        //}
        //pair.delta = sum_EX / (count * count * pose_1.num_as * pose_2.num_as);
        pair.delta = sum_EX / count;

        //ROS_INFO("ave delta:%.3f", pair.delta);

        if (pair.delta <= threshold) {
            return true;
        }
        else {
            return false;
        }

    }
}

//ceres_sol ver 输入keypoint_prob,输出point3d
double PerFrameAssociation::optimizeWithCeres2D_2D(int cam_1, int cam_2, Point2d point_1, Point2d point_2, Point3d& point_ret){

  double depth[2] = {0.5, 0.5};

	ceres::Problem problem;
	problem.AddResidualBlock(     // 向问题中添加误差项
    	  // 使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
      	new ceres::AutoDiffCostFunction<CostFunction_cam_2d_2d, 1, 2>(
        new CostFunction_cam_2d_2d(point_1, point_2, camera_rot[cam_1], camera_rot[cam_2], camera_trans[cam_1], camera_trans[cam_2], cam_param[cam_1], cam_param[cam_2])
      	),
      	nullptr,            // 核函数，这里不使用，为空
      	depth              // 待估计参数
    	);
    problem.SetParameterLowerBound(depth, 0, 0.0);
    problem.SetParameterLowerBound(depth, 1, 0.0);
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		//cout << summary.BriefReport() << endl;

    Eigen::Matrix<double,3,1> cam_1_point, cam_2_point;
    Eigen::Matrix<double,3,1> world_point1, world_point2;

		cam_1_point(2,0) = depth[0];
		cam_2_point(2,0) = depth[1];
		cam_1_point(0,0) = (point_1.x - cam_param[cam_1][2]) * cam_1_point(2,0) / cam_param[cam_1][0];
    cam_1_point(1,0) = (point_1.y - cam_param[cam_1][3]) * cam_1_point(2,0) / cam_param[cam_1][1];
    cam_2_point(0,0) = (point_2.x - cam_param[cam_2][2]) * cam_2_point(2,0) / cam_param[cam_2][0];
    cam_2_point(1,0) = (point_2.y - cam_param[cam_2][3]) * cam_2_point(2,0) / cam_param[cam_2][1];

    world_point1 = camera_rot[cam_1] * cam_1_point + camera_trans[cam_1];
    world_point2 = camera_rot[cam_2] * cam_2_point + camera_trans[cam_2];

    point_ret.x = (world_point1(0,0) + world_point2(0,0)) / 2;
    point_ret.y = (world_point1(1,0) + world_point2(1,0)) / 2;
    point_ret.z = (world_point1(2,0) + world_point2(2,0)) / 2;

    return getSpacedistance(Point3d(world_point1(0,0), world_point1(1,0), world_point1(2,0)),
                            Point3d(world_point2(0,0), world_point2(1,0), world_point2(2,0)));


}

double PerFrameAssociation::optimizeWithCeres2D_3D(int cam, Point2d point_1, Point3d point_2, Point3d& point_ret)
{

    ceres::Problem problem;
    double depth = 0.5;

	problem.AddResidualBlock(     // 向问题中添加误差项
        // 使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
        new ceres::AutoDiffCostFunction<CostFunction_cam_2d_3d, 1, 1>(
        		new CostFunction_cam_2d_3d(point_1, point_2,camera_rot[cam],camera_trans[cam], cam_param[cam])
      	),
      	nullptr,            // 核函数，这里不使用，为空
      	&depth                // 待估计参数
    	);

		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		//cout << summary.BriefReport() << endl;


    Eigen::Matrix<double,3,1> cam_point;
    Eigen::Matrix<double,3,1> world_point;

    cam_point(2,0) = depth;
    cam_point(0,0) = (point_1.x - cam_param[cam][2]) * cam_point(2,0) / cam_param[cam][0];
    cam_point(1,0) = (point_1.y - cam_param[cam][3]) * cam_point(2,0) / cam_param[cam][1];

    world_point = camera_rot[cam] * cam_point + camera_trans[cam];

    point_ret.x = (world_point(0,0) + point_2.x) / 2;
    point_ret.y = (world_point(1,0) + point_2.y) / 2;
    point_ret.z = (world_point(2,0) + point_2.z) / 2;

    return getSpacedistance(Point3d(world_point(0,0), world_point(1,0), world_point(2,0)),
                            point_2);
}

double PerFrameAssociation::getSpacedistance(cv::Point3d cam_a_point, cv::Point3d cam_b_point)
{
	return std::sqrt((cam_a_point.x - cam_b_point.x) * (cam_a_point.x - cam_b_point.x)
					+(cam_a_point.y - cam_b_point.y) * (cam_a_point.y - cam_b_point.y)
					+(cam_a_point.z - cam_b_point.z) * (cam_a_point.z - cam_b_point.z));
}

#endif
