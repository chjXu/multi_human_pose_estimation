#pragma once
#include <iostream>
#include "pose.hpp"

using namespace std;

class SkeletonPair{
public:
    int time1;
    int time2;

    int index_1;
    int index_2;

    double delta;

    vector<KeyPoint_3d> skeleton;


    SkeletonPair(){
        skeleton.resize(8);
        for(int i=0; i<8; i++){
            skeleton[i].x = 0.0;
            skeleton[i].y = 0.0;
            skeleton[i].z = 0.0;
            skeleton[i].available = false;
        }
        delta = 100;
    }

    static bool comp(SkeletonPair& skeleton_pair_1, SkeletonPair& skeleton_pair_2){
        return skeleton_pair_1.delta < skeleton_pair_2.delta;
    }
private:

};
