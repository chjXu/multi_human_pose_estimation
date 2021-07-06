#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <vector>
#include <pose.hpp>

void* thread( void *poses )
{
  double arg;
  vector<Pose> &m_pose = *(vector<Pose>*)poses;
  while(1) {
        arg = m_pose[0].pose_joints[0].x;
        printf( "thread:%.1f\n", arg);
        pthread_testcancel();
        //sleep(1);
    }
}

int main( int argc, char *argv[] )
{
    pthread_t th;
    int ret;

    vector<Pose> poses;
    poses.resize(8);

    ret = pthread_create( &th, NULL, thread, &poses );
    if( ret != 0 ){
        printf( "Create thread error!\n");
        return -1;
    }
    while(1) {
        poses[0].pose_joints[0].x = 1.0;
        printf( "main:%.1f\n", poses[0].pose_joints[0].x);
        //sleep(1);
    }

    ret = pthread_cancel(th);

    return 0;
}
