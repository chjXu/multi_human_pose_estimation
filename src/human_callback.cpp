{
  int person_num = keypoints.num_humans;

  //pose_vectors_image_all.clear();
  human_detected_last = human_detected;
  int person_num_last = human_detected_last.size();
  //std::array<bool, 20> id_list;
  //id_list.fill(true);
  //for(int i = 0; i < human_detected_last.size(); ++i){
  //    int index = human_detected_last[i].id();
  //    id_list[index] = false;
  //}

  ID_Dictionary id_dictionary_tmp = id_dictionary;

  human_detected.clear();

  vector< vector<double> > distance_pool;

  vector<int> index_min;

  vector< vector<KeyPoint_prob> > KeyPoints_selected;
  vector< KeyPoint_prob > KeyPoints_center_selected;
  vector< vector<poseVector> > poseVector_selected;

  distance_pool.resize(person_num_last);
  index_min.resize(person_num_last);

  if(person_num > 0){
      int person_comfirmed = 0;
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

          key_point_ite.x = body_keypoints[0].x;
          key_point_ite.y = body_keypoints[0].y;
          key_point_ite.p = body_keypoints[0].prob;
          KeyPoints_center_selected.push_back(key_point_ite);



          vector<poseVector> pose_vectors_image;
          getPoseVector(key_points_basic, pose_vectors_image);

          KeyPoints_selected.push_back(key_points_basic);
          poseVector_selected.push_back(pose_vectors_image);

          human human_tmp(-1);
          if(!human_detected_last.empty()){
            for(int i = 0; i < human_detected_last.size(); ++i){
                double dis = getKeyPointsSimilarity(key_points_basic, human_detected_last[i].key_points());
                distance_pool[i].push_back(dis);
            }
          }
          person_comfirmed++;
        }
        vector<vector<int>> index_hash(person_comfirmed);

        vector< vector<double>::iterator> minDis_vec;
        vector<int> person_lost_track_last;

        vector<int> person_paired;
        vector<int> person_overpaired;
        vector<int> person_notpaired;
        for(int i = 0; i < distance_pool.size(); ++i)
        {
            vector<double>::iterator minDis = std::min_element(distance_pool[i].begin(), distance_pool[i].end());
            if(*minDis < 30)
            {
              minDis_vec.push_back(minDis);
              int index = std::distance(distance_pool[i].begin(), minDis);
              index_hash[index].push_back(i);
            }
            person_lost_track.push_back(i);
        }

        for(int i = 0; i < index_hash.size(); ++i)
        {
          if(index_hash[i].size() == 1)
          {
            human human_tmp;
            int index = index_hash[i][0];
            human_tmp = human_detected_last[index];
            human_tmp.updateTime();
            human_tmp.setKeyPoints(KeyPoints_selected[i]);
            human_tmp.setPose(poseVector_selected[i]);
            human_tmp.setPosition(KeyPoints_center_selected[i].x, KeyPoints_center_selected[i].y);
            human_tmp.updateUnreliability();
            human_detected.push_back(human_tmp);

            id_dictionary_tmp.status[human_detected_last[index].id()] = -1;
            id_dictionary_tmp.index[human_detected_last[index].id()] = i;
          }
          else if(index_hash[i].size() > 1)
          {
            person_overpaired.push_back(i);
          }
          else if(index_hash[i].size() == 0)
          {
            person_notpaired.push_back(i);
          }
        }
        if(!person_overpaired.empty())
        {
          for(int i = 0;i < person_overpaired.size(); ++i)
          {
              int person = person_overpaired[i];
              int min_index;
              double distance_tmp = 9999;
              for(int j = 0; j < index_hash[person].size(); ++j)
              {
                int index = index_hash[person][j];
                if(*minDis_vec[index] < distance_tmp)
                {
                  min_index = j;
                  distance_tmp = *minDis_vec[index];
                }
              }
              vector<int>::iterator ite;
              for(ite = index_hash[person].begin();ite != index_hash[person].end();)
              {
                if(std::distance(ite - index_hash[person].begin() != min_index)
                {
                  person_lost_track_last.push_back(*ite);
                  index_hash[person].erase(ite);
                }
                else
                {
                  human human_tmp;
                  human_tmp = human_detected_last[*ite];
                  human_tmp.updateTime();
                  human_tmp.setKeyPoints(KeyPoints_selected[person]);
                  human_tmp.setPose(poseVector_selected[person]);
                  human_tmp.setPosition(KeyPoints_center_selected[person].x, KeyPoints_center_selected[person].y);
                  human_detected.push_back(human_tmp);

                  id_dictionary_tmp.status[human_detected_last[*ite].id()] = -1;
                  id_dictionary_tmp.index[human_detected_last[*ite].id()] = person;

                  ++ite;
                }
              }
              person_paired.push_back(person_overpaired[i]);
          }
        }
        if(!person_lost_track_last.empty())
        {
          id_dictionary_tmp.status[human_detected_last[*ite].id()] = 1;
          id_dictionary_tmp.index[human_detected_last[*ite].id()] = -1;
        }

        if(!person_notpaired.empty())
        {
            vector<int>::iterator ite;
            for(ite = person_notpaired.begin(); ite != person_notpaired.end();++ite)
            {
              human human_tmp;
              int id = getAvailableId(id_dictionary_tmp);
              human_tmp.setId(id);
              human_tmp.setKeyPoints(KeyPoints_selected[person]);
              human_tmp.setPose(poseVector_selected[person]);
              human_tmp.setPosition(KeyPoints_center_selected[person].x, KeyPoints_center_selected[person].y);
              human_detected.push_back(human_tmp);
              id_dictionary_tmp.status[id] = -1;
              id_dictionary_tmp.index[id] = *ite;
            }

        }

      }

}
