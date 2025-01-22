#ifndef PARTICLE_H
#define PARTICLE_H
#include <SFML/Graphics/CircleShape.hpp>
#include <cuda_runtime.h>
#include <registers.h>
#include <memory>
#include <random>


using namespace sf;

namespace tbb {
    constexpr size_t max_instances = 4096;


    class particle {
        static std::unique_ptr<tbb::particle> instances[max_instances];
        static float * device_particles_data_stream;
        static float * device_particles_pos_stream;
        static float host_particles_init_stream[];
        static float host_particles_pos_stream[tbb::max_instances*2];
        static size_t instance_num;

        CircleShape shape;
        const size_t id = particle::instance_num;

    public:
         particle(const float x, const float y, const float dx, const float dy, const float mass, const float atr_force, const float rep_force) {
             tbb::particle::host_particles_pos_stream[tbb::particle::instance_num*2] = x;
             tbb::particle::host_particles_pos_stream[tbb::particle::instance_num*2+1] = y;
             tbb::particle::host_particles_init_stream[tbb::particle::instance_num*5] = dx;
             tbb::particle::host_particles_init_stream[tbb::particle::instance_num*5+1] = dy;
             tbb::particle::host_particles_init_stream[tbb::particle::instance_num*5+2] = mass;
             tbb::particle::host_particles_init_stream[tbb::particle::instance_num*5+3] = atr_force;
             tbb::particle::host_particles_init_stream[tbb::particle::instance_num*5+4] = rep_force;

             this->shape.setRadius(1.0f);
             this->shape.setPosition(x, y);
             this->shape.setFillColor(Color::White);
             tbb::particle::instance_num++;
        }

        ~particle() {
        }

        static void load_particles();
        static void load_particles_manually();

        static void initialize() {
             if (!tbb::particle::device_particles_data_stream) {
                 cudaMalloc(&tbb::particle::device_particles_data_stream, tbb::max_instances * 5 * sizeof(float));
             }

            if (!tbb::particle::device_particles_pos_stream) {
                cudaMalloc(&tbb::particle::device_particles_pos_stream, tbb::max_instances * 2 * sizeof(float));
            }
        }

        static void cleanup() {
            cudaFree(tbb::particle::device_particles_data_stream);
            cudaFree(tbb::particle::device_particles_pos_stream);
        }

        static inline const size_t get_instance_count() {
            return tbb::particle::instance_num;
        }

        static inline std::unique_ptr<tbb::particle> * get_instances() {
            return tbb::particle::instances;
        }

        static float * get_host_particles_pos_stream() {
            return tbb::particle::host_particles_pos_stream;
        }

        static float * get_host_particles_init_stream() {
            return tbb::particle::host_particles_init_stream;
        }

        static float *& get_device_particles_data_stream() {
             return tbb::particle::device_particles_data_stream;
        }

        static float *& get_device_particles_pos_stream() {
            return tbb::particle::device_particles_pos_stream;
        }

        static constexpr inline size_t get_max_instance_count() {
            return tbb::max_instances;
        }

        CircleShape & get_shape() {
            return this->shape;
        }
    };
    size_t tbb::particle::instance_num = 0;
    float * tbb::particle::device_particles_pos_stream = nullptr;
    float * tbb::particle::device_particles_data_stream = nullptr;
    float tbb::particle::host_particles_init_stream[tbb::max_instances*5];
    float tbb::particle::host_particles_pos_stream[tbb::max_instances*2];
    std::unique_ptr<tbb::particle> tbb::particle::instances[max_instances] = {nullptr};


    inline void particle::load_particles() {
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


    inline void particle::load_particles_manually() {
        tbb::particle::get_instances()[0] = std::make_unique<tbb::particle>(WIN_WIDTH / 2, WIN_HEIGHT / 2, 0, 0, 100, 50, 10);
        tbb::particle::get_instances()[1] = std::make_unique<tbb::particle>(WIN_WIDTH / 2 + 20, WIN_HEIGHT / 2 - 10, 0, 0, 10, 10, 1);

    }


}

#endif //PARTICLE_H
