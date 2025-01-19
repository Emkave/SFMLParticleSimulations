#ifndef PARTICLE_H
#define PARTICLE_H
#include <SFML/Graphics/CircleShape.hpp>
#include <cuda_runtime.h>
#include <registers.h>
#include <memory>
#include <random>


using namespace sf;

namespace tbb {
    constexpr size_t max_instances = 8192;

    class particle {
        static std::unique_ptr<tbb::particle> instances[max_instances];
        static float * device_particles_data_stream;
        static float * host_particles_data_stream;
        static size_t instance_num;

        CircleShape shape;
        const size_t id = particle::instance_num;

    public:
         particle(const float x, const float y, const float dx, const float dy, const float mass, const float atr_force, const float rep_force) {
            tbb::particle::host_particles_data_stream[tbb::particle::instance_num*7] = x;
            tbb::particle::host_particles_data_stream[tbb::particle::instance_num*7+1] = y;
            tbb::particle::host_particles_data_stream[tbb::particle::instance_num*7+2] = dx;
            tbb::particle::host_particles_data_stream[tbb::particle::instance_num*7+3] = dy;
            tbb::particle::host_particles_data_stream[tbb::particle::instance_num*7+4] = mass;
            tbb::particle::host_particles_data_stream[tbb::particle::instance_num*7+5] = atr_force;
            tbb::particle::host_particles_data_stream[tbb::particle::instance_num*7+6] = rep_force;

            this->shape.setRadius(1.0f);
            this->shape.setPosition(x, y);
            this->shape.setFillColor(Color::White);
            tbb::particle::instance_num++;
        }

        ~particle() {
             if (tbb::particle::instance_num == 0 && tbb::particle::host_particles_data_stream) {
                 cudaFree(tbb::particle::host_particles_data_stream);
             }
        }

        static void load_particles();

        static void initialize() {
             if (!tbb::particle::host_particles_data_stream) {
                tbb::particle::host_particles_data_stream = new float[tbb::max_instances*7];
                 cudaMalloc(&tbb::particle::device_particles_data_stream, tbb::max_instances * 7 * sizeof(float));
             }
         }

        static void cleanup() {
             delete[] tbb::particle::host_particles_data_stream;
             cudaFree(tbb::particle::device_particles_data_stream);
         }

        static inline const size_t get_instance_count() {
            return tbb::particle::instance_num;
        }

        static inline std::unique_ptr<tbb::particle> * get_instances() {
            return tbb::particle::instances;
        }

        static float *& get_host_particles_data_stream() {
            return tbb::particle::host_particles_data_stream;
        }

        static float *& get_device_particles_data_stream() {
             return tbb::particle::device_particles_data_stream;
         }

        static constexpr inline size_t get_max_instance_count() {
            return tbb::max_instances;
        }

        CircleShape & get_shape() {
            return this->shape;
        }
    };
    size_t tbb::particle::instance_num = 0;
    float * tbb::particle::host_particles_data_stream = nullptr;
    float * tbb::particle::device_particles_data_stream = nullptr;
    std::unique_ptr<tbb::particle> tbb::particle::instances[max_instances] = {nullptr};
    __global__ void launch_simulation(float * particle_data_stream, const size_t particles_num);

    inline void particle::load_particles() {
        /*tbb::particle::get_instances()[0] = std::make_unique<tbb::particle>(
            WIN_WIDTH / 2,
            WIN_HEIGHT / 2,
            .7f,
            -0.4f,
            10,
            200,
            0.8
            );
        tbb::particle::get_instances()[1] = std::make_unique<tbb::particle>(
            WIN_WIDTH / 2 + 33,
            WIN_HEIGHT / 2 - 13,
            .4f,
            .4f,
            14,
            200,
            0.8
            );
        tbb::particle::get_instances()[2] = std::make_unique<tbb::particle>(
            WIN_WIDTH / 2 - 47,
            WIN_HEIGHT / 2 + 69,
            -.9f,
            0.4f,
            10,
            200,
            0.8
            );
        tbb::particle::get_instances()[3] = std::make_unique<tbb::particle>(
            WIN_WIDTH / 2 - 102,
            WIN_HEIGHT / 2 - 33,
            -.2f,
            0.9f,
            10,
            200,
            0.8
            );*/

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> angle_dist(0, 2 * M_PI);
        std::uniform_real_distribution<float> speed_dist(0.1, 5.f);
        std::uniform_real_distribution<float> mass_dist (10, 100);
        std::uniform_real_distribution<float> attr_dist (30, 80);
        std::uniform_real_distribution<float> rep_dist (10, 50);
        std::uniform_real_distribution<float> displacement_dist(-1, 1);

        for (size_t i=0; i<tbb::max_instances; i++) {
            const float x = rd() % WIN_WIDTH;
            const float y = rd() % WIN_HEIGHT;
            const float theta = angle_dist(gen);
            const float speed = speed_dist(gen);
            const float dx = speed * std::cos(theta);
            const float dy = speed * std::sin(theta);
            const float mass = mass_dist(gen);
            const float atr = attr_dist(gen);
            const float rep = rep_dist(gen);
            tbb::particle::get_instances()[i] = std::make_unique<tbb::particle>(x, y, dx, dy, mass, atr, rep);
        }
    }

}

#endif //PARTICLE_H
