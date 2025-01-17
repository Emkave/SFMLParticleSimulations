#ifndef PARTICLE_H
#define PARTICLE_H
#define UTILIZE_GPU 0
#include <cuda_runtime.h>
#include <memory>
#include <SFML/Graphics/CircleShape.hpp>

using namespace sf;

namespace tbb {
    constexpr size_t max_instances = 2;

    class particle {
        static tbb::particle * instances[max_instances];
        static float * particles_data_stream;
        static size_t instance_num;

        CircleShape shape;
        const size_t id = particle::instance_num;

    public:
         particle(const float x, const float y, const float dx, const float dy, const float mass, const float atr_dist, const float rep_dist) {
            if (!tbb::particle::particles_data_stream) {
                cudaMallocManaged(&tbb::particle::particles_data_stream, tbb::max_instances * 7 * sizeof(float));
            }

            tbb::particle::particles_data_stream[tbb::particle::instance_num*7] = x;
            tbb::particle::particles_data_stream[tbb::particle::instance_num*7+1] = y;
            tbb::particle::particles_data_stream[tbb::particle::instance_num*7+2] = dx;
            tbb::particle::particles_data_stream[tbb::particle::instance_num*7+3] = dy;
            tbb::particle::particles_data_stream[tbb::particle::instance_num*7+4] = mass;
            tbb::particle::particles_data_stream[tbb::particle::instance_num*7+5] = atr_dist;
            tbb::particle::particles_data_stream[tbb::particle::instance_num*7+6] = rep_dist;

            this->shape.setRadius(2.0f);
            this->shape.setPosition(x, y);
            this->shape.setFillColor(Color::White);
            tbb::particle::instances[tbb::particle::instance_num] = this;
            tbb::particle::instance_num++;
        }

        ~particle() {
             if (tbb::particle::instance_num == 0 && tbb::particle::particles_data_stream) {
                 cudaFree(tbb::particle::particles_data_stream);
             }
        }

        static inline const size_t get_instance_count() {
            return tbb::particle::instance_num;
        }

        static inline tbb::particle ** get_instances() {
            return tbb::particle::instances;
        }

        static float *& get_particles_data_stream() {
            return tbb::particle::particles_data_stream;
        }

        static constexpr inline size_t get_max_instance_count() {
            return tbb::max_instances;
        }

        CircleShape & get_shape() {
            return this->shape;
        }
    };
    size_t tbb::particle::instance_num = 0;
    float * tbb::particle::particles_data_stream = nullptr;
    tbb::particle * tbb::particle::instances[max_instances] = {nullptr};


    __global__ void launch_simulation(float * particle_data_stream, size_t particles_num);
}

#endif //PARTICLE_H
