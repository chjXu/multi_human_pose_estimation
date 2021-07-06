#ifndef CORRESPONDENCE_HPP
#define CORRESPONDENCE_HPP
#include "pose.hpp"

class PosePair{
    public:
        int group;
        int label_1;
        int label_2;
        int index_1;
        int index_2;
        int cam_1;
        int cam_2;

        double delta;

        vector<KeyPoint_3d> pose_joints_3d;

        PosePair() {
            pose_joints_3d.resize(8);
            for(int i = 0; i < 8; ++i) {
                pose_joints_3d[i].x = 0.0;
                pose_joints_3d[i].y = 0.0;
                pose_joints_3d[i].z = 0.0;
                pose_joints_3d[i].available = false;
            }
            delta = 1000;
        }

        static bool comp(PosePair& pair_1, PosePair& pair_2) {
            return pair_1.delta < pair_2.delta;
        }
};

#endif
