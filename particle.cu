#include <particle.h>

size_t tbb::instance_num = 0;
std::unique_ptr<tbb::particle> tbb::instances[max_particle_instances];
float * tbb::host_particles_init_stream = nullptr;
float * tbb::host_particles_pos_stream = nullptr;
void * tbb::host_extra_properties_stream = nullptr;
float * tbb::device_particles_data_stream = nullptr;
float * tbb::device_particles_pos_stream = nullptr;
void * tbb::device_extra_properties_stream = nullptr;

