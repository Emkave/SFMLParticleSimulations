#ifndef SIMULATIONS_H
#define SIMULATIONS_H

namespace simulation {
    __device__ void simulation1(float * particle_data_stream, float * particle_pos_stream, const size_t particle_num);


    template <size_t Index> struct simulation_selector;


    template <> struct simulation_selector<0> {
        static constexpr void (*function)(float *, float *, const size_t) = simulation1;
    };


    template <size_t Index> __global__ void launch_simulation(float * particle_data_stream, float * particle_pos_stream, const size_t particles_num) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < particles_num) {
            simulation::simulation_selector<Index>::function(particle_data_stream, particle_pos_stream, particles_num);
        }
    }
}

#endif //SIMULATIONS_H
