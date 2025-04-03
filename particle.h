#ifndef PARTICLE_H
#define PARTICLE_H
#include <SFML/Graphics/CircleShape.hpp>
#include <cuda_runtime.h>
#include <memory>
#include <random>
#include <registers.h>

using namespace sf;

template <typename T> T generate_random_value() {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    if constexpr (std::is_floating_point_v<T>) {
        static std::uniform_real_distribution<T> dist(0.1, 10.0);
        return dist(gen);
    } else if constexpr (std::is_integral_v<T>) {
        static std::uniform_int_distribution<T> dist(1, 10);
        return dist(gen);
    } else {
        return T{};
    }
}

template <typename T> constexpr auto struct_to_tuple(T& obj) {
    return std::apply([&](auto &&... args) {
        return std::make_tuple(std::ref(args)...);
    }, std::forward_as_tuple(obj));
}

template <typename T> constexpr size_t count_fields() {
    return std::tuple_size_v<std::decay_t<decltype(struct_to_tuple(std::declval<T>()))>>;
}

template <typename Struct, typename Tuple, size_t... I>
void fill_struct_with_random_values(Struct& obj, Tuple && t, std::index_sequence<I...>) {
    ((std::get<I>(t) = generate_random_value<std::remove_reference_t<decltype(std::get<I>(t))>>()), ...);
}

template <typename Struct> void generate_random_struct_values(Struct& obj) {
    auto tuple_rep = struct_to_tuple(obj);
    constexpr size_t num_fields = std::tuple_size_v<std::decay_t<decltype(tuple_rep)>>;
    fill_struct_with_random_values(obj, tuple_rep, std::make_index_sequence<num_fields>{});
}


namespace tbb {
    template <size_t Index> struct particle_class_selector;

    template <> struct particle_class_selector<0> {
        struct extra_properties {
            float extra1 = 1.0f;
        };
        using extra_tuple = std::tuple<float>;
        static constexpr extra_properties default_properties {};
        static constexpr size_t extra_prop_size = sizeof(extra_properties);
        static constexpr size_t extra_prop_count = std::tuple_size_v<extra_tuple>;
    };

    template <> struct particle_class_selector<1> {
        struct extra_properties {
            float mass = 1.0f;
        };
        static constexpr extra_properties default_properties {};
        static constexpr size_t extra_prop_size = sizeof(extra_properties);
        static constexpr size_t extra_prop_count = 1;
    };

    template <size_t Index> class particle final {
    protected:
        static std::unique_ptr<particle> instances[registers::max_particle_instances];
        static void * host_extra_properties_stream;
        static void * device_extra_properties_stream;
        static float * host_particles_init_stream;
        static float * host_particles_pos_stream;
        static float * device_particles_data_stream;
        static float * device_particles_pos_stream;
        static size_t instance_num;

        const size_t id = particle::instance_num;
        CircleShape shape;

    public:
        particle(const float x, const float y, const float dx, const float dy, const float attr_dst, const float rep_dst,
            const typename particle_class_selector<Index>::extra_properties & extra_props);
        ~particle() = default;

        inline CircleShape & get_shape() noexcept;

        static inline float * get_host_particles_pos_stream() noexcept;
        static inline float * get_host_particles_init_stream() noexcept;
        static inline void * get_host_particles_extra_stream() noexcept;
        static inline float * get_device_particles_pos_stream() noexcept;
        static inline float * get_device_particles_data_stream() noexcept;
        static inline void * get_device_particles_extra_stream() noexcept;
        static constexpr inline size_t get_extra_prop_count() noexcept;
        static inline void cleanup();
        static inline void load_to_device();
        static inline void load_particles();
        static inline size_t get_instance_count() noexcept;
        static inline std::unique_ptr<tbb::particle<Index>> * get_instances() noexcept;
    };

    template <size_t Index> std::unique_ptr<particle<Index>> particle<Index>::instances[] = {nullptr};
    template <size_t Index> float * particle<Index>::host_particles_init_stream = nullptr;
    template <size_t Index> float * particle<Index>::host_particles_pos_stream = nullptr;
    template <size_t Index> void * particle<Index>::host_extra_properties_stream = nullptr;
    template <size_t Index> float * particle<Index>::device_particles_data_stream = nullptr;
    template <size_t Index> float * particle<Index>::device_particles_pos_stream = nullptr;
    template <size_t Index> void * particle<Index>::device_extra_properties_stream = nullptr;
    template <size_t Index> size_t particle<Index>::instance_num = 0;

    template <size_t Index> inline particle<Index>::particle(const float x, const float y, const float dx, const float dy, const float attr_dst, const float rep_dst,
        const typename particle_class_selector<Index>::extra_properties & extra_props) {
        constexpr size_t base_size = 4 * sizeof(float);
        constexpr size_t extra_size = particle_class_selector<Index>::extra_prop_size;

        if (!particle::host_particles_pos_stream) {
            cudaMallocHost(&particle::host_particles_pos_stream, registers::max_particle_instances * 2 * sizeof(float));
        }

        if (!particle::host_particles_init_stream) {
            cudaMallocHost(&particle::host_particles_init_stream, registers::max_particle_instances * base_size);
        }

        if (!particle::host_extra_properties_stream) {
            cudaMallocHost(&particle::host_extra_properties_stream, registers::max_particle_instances * particle_class_selector<Index>::extra_prop_size);
        }

        const size_t pos_offset = particle::instance_num * 2;
        particle::host_particles_pos_stream[pos_offset] = x;
        particle::host_particles_pos_stream[pos_offset + 1] = y;

        const size_t init_offset = particle::instance_num * 4;
        particle::host_particles_init_stream[init_offset] = dx;
        particle::host_particles_init_stream[init_offset + 1] = dy;
        particle::host_particles_init_stream[init_offset + 2] = attr_dst;
        particle::host_particles_init_stream[init_offset + 3] = rep_dst;

        const size_t extra_offset = particle::instance_num * particle_class_selector<Index>::extra_prop_size;
        std::memcpy(static_cast<char*>(particle::host_extra_properties_stream) + extra_offset, &extra_props, extra_size);

        this->shape.setRadius(1.0f);
        this->shape.setFillColor(Color::White);
        this->shape.setPosition(x, y);

        particle::instance_num++;
    }

    template <size_t Index> inline CircleShape & particle<Index>::get_shape() noexcept {
        return this->shape;
    }

    template <size_t Index> inline float * particle<Index>::get_host_particles_pos_stream() noexcept {
        return particle::host_particles_pos_stream;
    }

    template <size_t Index> inline float * particle<Index>::get_host_particles_init_stream() noexcept {
        return particle::host_particles_init_stream;
    }

    template <size_t Index> void *particle<Index>::get_host_particles_extra_stream() noexcept {
        return particle::host_extra_properties_stream;
    }

    template <size_t Index> inline float * particle<Index>::get_device_particles_data_stream() noexcept {
        return particle::device_particles_data_stream;
    }

    template <size_t Index> inline float * particle<Index>::get_device_particles_pos_stream() noexcept {
        return particle::device_particles_pos_stream;
    }

    template <size_t Index> void *particle<Index>::get_device_particles_extra_stream() noexcept {
        return particle::device_extra_properties_stream;
    }

    template <size_t Index> constexpr inline size_t particle<Index>::get_extra_prop_count() noexcept {
        return particle_class_selector<Index>::extra_prop_count;
    }

    template <size_t Index> inline void particle<Index>::cleanup() noexcept {
        cudaFreeHost(particle::host_particles_init_stream);
        cudaFreeHost(particle::host_particles_pos_stream);
        cudaFreeHost(particle::host_extra_properties_stream);
        cudaFree(particle::device_particles_data_stream);
        cudaFree(particle::device_particles_pos_stream);
        cudaFree(particle::device_extra_properties_stream);
    }

    template <size_t Index> inline void particle<Index>::load_to_device() {
        if (!particle::device_particles_data_stream) {
            cudaMalloc(&particle::device_particles_data_stream, registers::max_particle_instances * 4 * sizeof(float));
            cudaMemcpy(particle::device_particles_data_stream, particle::host_particles_init_stream, registers::max_particle_instances * 4 * sizeof(float), cudaMemcpyHostToDevice);
        }

        if (!particle::device_particles_pos_stream) {
            cudaMalloc(&particle::device_particles_pos_stream, registers::max_particle_instances * 2 * sizeof(float));
            cudaMemcpy(particle::device_particles_pos_stream, particle::host_particles_pos_stream, registers::max_particle_instances * 2 * sizeof(float), cudaMemcpyHostToDevice);
        }

        if (!particle::device_extra_properties_stream) {
            cudaMalloc(&particle::device_extra_properties_stream, registers::max_particle_instances * particle_class_selector<Index>::extra_prop_size);
            cudaMemcpy(particle::device_extra_properties_stream, particle::host_extra_properties_stream,
                registers::max_particle_instances * particle_class_selector<Index>::extra_prop_size, cudaMemcpyHostToDevice);
        }
    }

    template <size_t Index> inline size_t particle<Index>::get_instance_count() noexcept {
        return particle::instance_num;
    }

    template <size_t Index> inline std::unique_ptr<tbb::particle<Index>> * particle<Index>::get_instances() noexcept {
        return particle::instances;
    }

    template <size_t Index> inline void particle<Index>::load_particles() {
        std::random_device rd;
        std::mt19937 gen(rd());
        const std::uniform_real_distribution<float> angle_dist(registers::urd_angle_dist_start, registers::urd_angle_dist_stop);
        const std::uniform_real_distribution<float> speed_dist(registers::urd_speed_dist_start, registers::urd_speed_dist_stop);
        const std::uniform_real_distribution<float> attr_dist (registers::urd_attr_dist_start, registers::urd_attr_dist_stop);
        const std::uniform_real_distribution<float> rep_dist (registers::urd_rep_dist_start, registers::urd_rep_dist_stop);

        for (size_t i=0; i<registers::max_particle_instances; i++) {
            const float x = rd() % WIN_WIDTH;
            const float y = rd() % WIN_HEIGHT;
            const float theta = angle_dist(gen);
            const float speed = speed_dist(gen);
            const float dx = speed * std::cos(theta);
            const float dy = speed * std::sin(theta);
            const float attr = attr_dist(gen);
            const float rep = rep_dist(gen);

            typename particle_class_selector<Index>::extra_properties extra_props;
            particle::instances[i] = std::make_unique<particle<Index>>(x, y, dx, dy, attr, rep, extra_props);
        }
    }
}

#endif //PARTICLE_H
