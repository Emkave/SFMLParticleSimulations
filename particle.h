#ifndef PARTICLE_H
#define PARTICLE_H
#include <SFML/Graphics/CircleShape.hpp>
#include <cuda_runtime.h>
#include <memory>
#include <random>
#include <registers.h>

using namespace sf;


template <typename T> constexpr auto struct_to_tuple(T& obj) {
    return std::apply([&](auto &&... args) {
        return std::make_tuple(std::ref(args)...);
    }, std::forward_as_tuple(obj));
}

template <typename T> constexpr size_t count_fields() {
    return std::tuple_size_v<std::decay_t<decltype(struct_to_tuple(std::declval<T>()))>>;
}

namespace tbb {
    extern size_t instance_num;

    class particle {
    protected:
        const size_t id = instance_num;
        CircleShape shape;

    public:
        particle(const float && x, const float y, const float dx, const float dy, const float attr_dst, const float rep_dst, const float mass);
        ~particle() = default;

        inline CircleShape & get_shape() noexcept;

        static inline float * get_host_particles_pos_stream() noexcept;
        static inline float * get_host_particles_init_stream() noexcept;
        static inline void * get_host_particles_extra_stream() noexcept;
        static inline float * get_device_particles_pos_stream() noexcept;
        static inline float * get_device_particles_data_stream() noexcept;
        static inline void * get_device_particles_extra_stream() noexcept;
        static constexpr inline size_t get_extra_prop_count() noexcept;
        static inline void cleanup() noexcept;
        static inline void load_to_device();
        static inline void load_particles();
        static inline size_t get_instance_count() noexcept;
        static inline std::unique_ptr<tbb::particle> * get_instances() noexcept;
    };

    extern std::unique_ptr<particle> instances[max_particle_instances];
    extern float * host_particles_init_stream;
    extern float * host_particles_pos_stream;
    extern void * host_extra_properties_stream;
    extern float * device_particles_data_stream;
    extern float * device_particles_pos_stream;
    extern void * device_extra_properties_stream;

    inline particle::particle(const float && x, const float y, const float dx, const float dy, const float attr_dst, const float rep_dst, const float mass) {
        constexpr size_t base_size = 4 * sizeof(float);
        constexpr size_t extra_size = sizeof(float);

        if (!host_particles_pos_stream) {
            cudaMallocHost(&host_particles_pos_stream, max_particle_instances * 2 * sizeof(float));
        }

        if (!host_particles_init_stream) {
            cudaMallocHost(&host_particles_init_stream, max_particle_instances * base_size);
        }

        if (!host_extra_properties_stream) {
            cudaMallocHost(&host_extra_properties_stream, max_particle_instances * sizeof(float));
        }

        const size_t pos_offset = instance_num * 2;
        host_particles_pos_stream[pos_offset] = x;
        host_particles_pos_stream[pos_offset + 1] = y;

        const size_t init_offset = instance_num * 4;
        host_particles_init_stream[init_offset] = dx;
        host_particles_init_stream[init_offset + 1] = dy;
        host_particles_init_stream[init_offset + 2] = attr_dst;
        host_particles_init_stream[init_offset + 3] = rep_dst;

        const size_t extra_offset = instance_num * sizeof(float);
        memcpy(static_cast<char*>(host_extra_properties_stream) + extra_offset, &mass, extra_size);

        this->shape.setRadius(1.0f);
        this->shape.setFillColor(Color::White);
        this->shape.setPosition(x, y);

        instance_num++;
    }

    inline CircleShape & particle::get_shape() noexcept {
        return this->shape;
    }

    inline float * particle::get_host_particles_pos_stream() noexcept {
        return host_particles_pos_stream;
    }

    inline float * particle::get_host_particles_init_stream() noexcept {
        return host_particles_init_stream;
    }

    void * particle::get_host_particles_extra_stream() noexcept {
        return host_extra_properties_stream;
    }

    inline float * particle::get_device_particles_data_stream() noexcept {
        return device_particles_data_stream;
    }

    inline float * particle::get_device_particles_pos_stream() noexcept {
        return device_particles_pos_stream;
    }

    void * particle::get_device_particles_extra_stream() noexcept {
        return device_extra_properties_stream;
    }

    constexpr inline size_t particle::get_extra_prop_count() noexcept {
        return 1;
    }

    inline void particle::cleanup() noexcept {
        cudaFreeHost(host_particles_init_stream);
        cudaFreeHost(host_particles_pos_stream);
        cudaFreeHost(host_extra_properties_stream);
        cudaFree(device_particles_data_stream);
        cudaFree(device_particles_pos_stream);
        cudaFree(device_extra_properties_stream);
    }

    inline void particle::load_to_device() {
        if (!device_particles_data_stream) {
            cudaMalloc(&device_particles_data_stream, max_particle_instances * 4 * sizeof(float));
            cudaMemcpy(device_particles_data_stream, host_particles_init_stream, max_particle_instances * 4 * sizeof(float), cudaMemcpyHostToDevice);
        }

        if (!device_particles_pos_stream) {
            cudaMalloc(&device_particles_pos_stream, max_particle_instances * 2 * sizeof(float));
            cudaMemcpy(device_particles_pos_stream, host_particles_pos_stream, max_particle_instances * 2 * sizeof(float), cudaMemcpyHostToDevice);
        }

        if (!device_extra_properties_stream) {
            cudaMalloc(&device_extra_properties_stream, max_particle_instances * sizeof(float));
            cudaMemcpy(device_extra_properties_stream, host_extra_properties_stream,
                max_particle_instances * sizeof(float), cudaMemcpyHostToDevice);
        }
    }

    inline size_t particle::get_instance_count() noexcept {
        return instance_num;
    }

    inline std::unique_ptr<tbb::particle> * particle::get_instances() noexcept {
        return instances;
    }

    inline void particle::load_particles() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> angle_dist(urd_angle_dist_start, urd_angle_dist_stop);
        std::uniform_real_distribution<float> speed_dist(urd_speed_dist_start, urd_speed_dist_stop);
        std::uniform_real_distribution<float> attr_dist (urd_attr_dist_start, urd_attr_dist_stop);
        std::uniform_real_distribution<float> rep_dist (urd_rep_dist_start, urd_rep_dist_stop);

        for (size_t i=0; i<max_particle_instances; i++) {
            const float x = rd() % WIN_WIDTH;
            const float y = rd() % WIN_HEIGHT;
            const float theta = angle_dist(gen);
            const float speed = speed_dist(gen);
            const float dx = speed * std::cos(theta);
            const float dy = speed * std::sin(theta);
            const float attr = attr_dist(gen);
            const float rep = rep_dist(gen);

            instances[i] = std::make_unique<particle>(static_cast<const float &&>(x), static_cast<const float &&>(y), static_cast<const float &&>(dx), static_cast<const float &&>(dy), static_cast<const float &&>(attr), static_cast<const float &&>(rep), 5.0f);
        }
    }
}

#endif //PARTICLE_H
