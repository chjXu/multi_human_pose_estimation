#include <iostream>
#include <vector>

#include "skeleton.hpp"
#include <string>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>

using namespace std;

class HumanModel
{
public:
    HumanModel(const string &human_name, const int label){
        this->human_name = human_name;
        this->label = label;
        joint.resize(1);    //tf不能正常发出,resize(2)
        joint_name.resize(1);
        q.setRPY(0,0,0);
    }
    void setJointName();
    void getPose(frame_skeleton &skeleton);
    void setJoint();
    void setTF();
    void sendTF();

private:
    Skeleton skeleton;
    string human_name;
    int label;
    frame_skeleton one_skeleton;
    vector<string> joint_name;
    vector<tf::Transform> joint;
    tf::TransformBroadcaster br;
    tf::Quaternion q;

    tf::Transform lhz_head;
    tf::Transform lhz_neck;
    tf::Transform lhz_rShoulder;
    tf::Transform lhz_rArm;
    tf::Transform lhz_rWrist;
    tf::Transform lhz_lShoulder;
    tf::Transform lhz_lArm;
    tf::Transform lhz_lWrist;
    tf::Transform lhz_hip;

    tf::Transform xcj_head;
    tf::Transform xcj_neck;
    tf::Transform xcj_rShoulder;
    tf::Transform xcj_rArm;
    tf::Transform xcj_rWrist;
    tf::Transform xcj_lShoulder;
    tf::Transform xcj_lArm;
    tf::Transform xcj_lWrist;
    tf::Transform xcj_hip;
};



void HumanModel::getPose(frame_skeleton &skeleton){
    if(!skeleton.a_skeleton.empty()){
        this->one_skeleton = skeleton;
    }
    setJointName();
    setJoint();
    setTF();
    sendTF();

}

void HumanModel::setJointName(){
    if(one_skeleton.a_skeleton.empty()){
        return;
    }
    //joint_name.push_back(human_name + "/head");
    joint_name.push_back(human_name + "/neck");
    joint_name.push_back(human_name + "/rShoulder");
    joint_name.push_back(human_name + "/rArm");
    joint_name.push_back(human_name + "/rWrist");
    joint_name.push_back(human_name + "/lShoulder");
    joint_name.push_back(human_name + "/lArm");
    joint_name.push_back(human_name + "/lWrist");
    joint_name.push_back(human_name + "/hip");
}

void HumanModel::setJoint(){
    if(one_skeleton.a_skeleton.empty()){
        return;
    }
    if(this->human_name == "lhz"){
        //joint.emplace_back(lhz_head);
        joint.emplace_back(lhz_neck);
        joint.emplace_back(lhz_rShoulder);
        joint.emplace_back(lhz_rArm);
        joint.emplace_back(lhz_rWrist);
        joint.emplace_back(lhz_lShoulder);
        joint.emplace_back(lhz_lArm);
        joint.emplace_back(lhz_lWrist);
        joint.emplace_back(lhz_hip);
    }
    if(this->human_name == "xcj"){
        //joint.emplace_back(xcj_head);
        joint.emplace_back(xcj_neck);
        joint.emplace_back(xcj_rShoulder);
        joint.emplace_back(xcj_rArm);
        joint.emplace_back(xcj_rWrist);
        joint.emplace_back(xcj_lShoulder);
        joint.emplace_back(xcj_lArm);
        joint.emplace_back(xcj_lWrist);
        joint.emplace_back(xcj_hip);
    }

}

void HumanModel::setTF(){
    if(!one_skeleton.a_skeleton.empty()){
        for(int i=1; i<one_skeleton.a_skeleton.size(); i++){
            joint[i].setOrigin(tf::Vector3(one_skeleton.a_skeleton[i].x, one_skeleton.a_skeleton[i].y, one_skeleton.a_skeleton[i].z));
            joint[i].setRotation(q);
        }
    }
}

void HumanModel::sendTF(){
    if(!one_skeleton.a_skeleton.empty()){
        for(int i=1; i<one_skeleton.a_skeleton.size(); i++){
            br.sendTransform(tf::StampedTransform(joint[i], ros::Time::now(), "marker_0", joint_name[i]));
        }
    }
}
