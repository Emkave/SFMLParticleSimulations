#ifndef SIMULATIONS_H
#define SIMULATIONS_H

namespace simulation {
    __device__ void simulation1(float * particle_data_stream, float * particle_pos_stream, void * particle_extra_stream, const size_t particle_num);

    __global__ inline void launch_simulation(float * particle_data_stream, float * particle_pos_stream, void * particle_extra_stream, const size_t particles_num) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < particles_num) {
            simulation::simulation1(particle_data_stream, particle_pos_stream, particle_extra_stream, particles_num);
        }
    }
}

#endif //SIMULATIONS_H
