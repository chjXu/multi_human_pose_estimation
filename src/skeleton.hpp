#pragma once
#include "pose3D.hpp"
//为队列中的每一帧姿态进行编号
struct frame_skeleton{
    int label = -1;
    vector<KeyPoint_3d> a_skeleton;
};


class Skeleton{
public:
    int time_index = 0;
    //int index = 0;
    vector<frame_skeleton> skeleton_3d;


public:
    Skeleton() {
        //skeleton_3d.clear();
        //a_skeleton.resize(8);
        skeleton_3d.resize(8);
    }
        //int size(){
        //    return skeleton_3d.size();
        //}

        //void setSkeletons(vector<vector<KeyPoint_3d>>& skeleton, int vi_time){
        //    skeleton_3d = skeleton;
        //    this->frame = vi_time;
            //all_skeletons_a_frame.clear();
            //for(int i=0; i<skeleton.size(); i++){
                //all_skeletons_a_frame[i].pre_skeleton = skeleton[i];
            //    all_skeletons_a_frame[i].time = vi_time;
            //}
        //}
        //vector<vector<KeyPoint_3d>> getSkeletons() const{return skeleton_3d;}

       // void setFrame(int frame){
        //    this->frame = frame;
       // }

        // void setIndex(int index){
        //     this->index = index;
        // }

        //int getFrame(){
        //    return this->frame;
        //}

        // int getIndex(){
        //     return this->index;
        // }

        //static bool comp(SkeletonPair& pair_1, SkeletonPair& pair_2) {
        //    return pair_1.delta < pair_2.delta;
        //}
};


