#include <vector>
#include <queue>
#include <math.h>
using std::vector;
using std::queue;
typedef struct poseVector
{
  double x;
  double y;
  double p;
}poseVector;

typedef struct point_2d
{
  int x;
  int y;
}point_2d;

typedef struct keypoints_with_prob
{
  double x;
  double y;
  double p;
}KeyPoint_prob;

class human
{
  public:
    human(){
        time_exist_ = 0;
        similarity_ = 0.0;
        similarity_sum_ = 0.0;
        confidence_ = 0.0;
        punishment_ = 1.0;
        unreliability_ = 0;

        distance_ = 999.0;
        distance_sum_ = 999.0;
        distance_buffer_.push_back(distance_);
        queue_max_ = 3;

    }
    human(int id){
      id_ = id;
      time_exist_ = 0;
      similarity_ = 0.0;
      similarity_sum_ = 0.0;
      confidence_ = 0.0;
      punishment_ = 1.0;

      distance_ = 500.0;
      distance_sum_ = 500.0;
      queue_max_ = 3;

    }

    int id(){
       return id_;
    }

    vector<poseVector> pose(){
        return human_pose_2d_;
    }

    vector<KeyPoint_prob> key_points(){
        return key_points_;
    }

    vector<KeyPoint_prob> key_points_front(){
        return key_points_queue_.front();
    }

    int unreliability()
    {
      return unreliability_;
    }

    point_2d position()
    {
        return position_;
    }

    int time(){
        return time_exist_;
    }


    double confidence()
    {
        return confidence_;
    }

    double punishment()
    {
        return punishment_;
    }

    void resetPunish()
    {
      punishment_ = 1.0;
    }

    vector<double> similarity_buffer()
    {
        return similarity_buffer_;
    }

    double similarity_sum()
    {
        return similarity_sum_;
    }

    double similarity()
    {
        return (similarity_ * punishment_);
    }

    double distance()
    {
        return (distance_ / punishment_);
    }

    vector<double> distance_buffer()
    {
        return distance_buffer_;
    }

    void setId(int id){
       id_ = id;
    }

    void setTime(const int new_time)
    {
        time_exist_ = new_time;
        if(time_exist_ > 100)
        {
            time_exist_ = 100;
        }
        confidence_ = -(exp(-double(time_exist_) * 5 /70.0)) + 1;
    }

    void setPose(vector<poseVector> pose_2d)
    {
        human_pose_2d_ = pose_2d;
    }

    void setKeyPoints(vector<KeyPoint_prob> key_points)
    {
        key_points_ = key_points;
        key_points_queue_.push(key_points);
        if(key_points_queue_.size() > queue_max_)
        {
          key_points_queue_.pop();
        }
    }

    void setPosition(int x, int y){
       position_.x = x;
       position_.y = y;
    }

    void updatePunishment()
    {
      punishment_ *= 0.9;
    }

    void setUnreliability()
    {
      unreliability_ = 10;
    }
    void updateUnreliability()
    {
      if(unreliability_ > 0)
        unreliability_--;
    }

    void updateSimilarity(double similarity_new)
    {
        similarity_buffer_.push_back(similarity_new);
        int buffer_size = similarity_buffer_.size();
        if(buffer_size <= 100)
        {
            similarity_sum_ += similarity_new;
            similarity_ = similarity_sum_ / buffer_size;
        }
        else if(buffer_size > 100)
        {
            similarity_sum_ -= similarity_buffer_.front();
            similarity_buffer_.erase(similarity_buffer_.begin());
            similarity_sum_ += similarity_new;
            similarity_ = similarity_sum_ / 100;
        }
        similarity_ *= confidence_;
    }

    void updateDistance(double distance_new)
    {
        distance_buffer_.push_back(distance_new);
        int buffer_size = distance_buffer_.size();
        if(buffer_size <= 100)
        {
            distance_sum_ += distance_new;
            distance_ = distance_sum_ / buffer_size;
        }
        else if(buffer_size > 100)
        {
            distance_sum_ -= distance_buffer_.front();
            distance_buffer_.erase(distance_buffer_.begin());
            distance_sum_ += distance_new;
            distance_ = distance_sum_ / 100;
        }
        //distance_ /= confidence_;
    }

    void updateTime()
    {
        setTime(time_exist_ + 1);
    }

    void inherit(human& human_father)
    {
        id_ = human_father.id();
        similarity_buffer_ = human_father.similarity_buffer();
        similarity_sum_ = human_father.similarity_sum();
        setTime(human_father.time() + 1);
        similarity_ = human_father.similarity();
        punishment_ = human_father.punishment();
    }


  private:
    int id_;
    vector<poseVector> human_pose_2d_;
    vector<KeyPoint_prob> key_points_;
    vector<double> similarity_buffer_;

    queue< vector<KeyPoint_prob> > key_points_queue_;

    double similarity_sum_;
    double similarity_;
    double confidence_;
    double punishment_;
    point_2d position_;
    int unreliability_;

    int time_exist_;
    double distance_;
    vector<double> distance_buffer_;
    double distance_sum_;
    int queue_max_;

};

class human_linkage
{
  public:
    human_linkage(int id_1, int id_2, int similarity)
    {
        members.push_back(id_1);
        members.push_back(id_2);
        similarity_sum_ = 0.0;
        updateSimilarity(similarity);
    }

    int time(){
        return time_exist_;
    }
    double similarity() const
    {
        return similarity_;
    }

    double confidence() const
    {
        return confidence_;
    }

    void updateSimilarity(double similarity_new)
    {
        similarity_buffer_.push_back(similarity_new);
        int buffer_size = similarity_buffer_.size();
        if(buffer_size <= 100)
        {
            similarity_sum_ += similarity_new;
            similarity_ = similarity_sum_ / buffer_size;
        }
        else if(buffer_size > 100)
        {
            similarity_sum_ -= similarity_buffer_.front();
            similarity_buffer_.erase(similarity_buffer_.begin());
            similarity_sum_ += similarity_new;
            similarity_ = similarity_sum_ / 100;
        }
        confidence_ = similarity_;
        //similarity_ *= confidence_;
    }

    bool isBetween (int id_1, int id_2) const
    {
        if(id_1 == members[0] && id_2 == members[1])
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    int id_1()
    {
      return members[0];
    }
    int id_2()
    {
      return members[1];
    }


  private:
    vector<int> members;
    vector<double> similarity_buffer_;
    double similarity_sum_;
    double similarity_;
    double confidence_;
    int time_exist_;



};
