#ifndef REGISTERS_H
#define REGISTERS_H
#define WIN_WIDTH 100
#define WIN_HEIGHT 100

/*
#define max_particle_instances 7000
#define urd_angle_dist_start -1
#define urd_angle_dist_stop 1
#define urd_speed_dist_start 0
#define urd_speed_dist_stop 0.1
#define urd_attr_dist_start 2
#define urd_attr_dist_stop 2
#define urd_rep_dist_start 1
#define urd_rep_dist_stop 3
*/



namespace registers {
    constexpr size_t threads_per_block = 256;
}
#endif //REGISTERS_H
