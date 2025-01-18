#include <SFML/Graphics.hpp>
#include <particle.h>
#include <thread_pool.h>
#include <thread_safe_queue.h>
#include <registers.h>
#include <vld.h>
#include <iostream>
#include <queue>
#include <mutex>



using namespace sf;

__global__ void tbb::launch_simulation(float * particle_data_stream, const size_t particles_num) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < particles_num) {
        // Current particle properties
        float x = particle_data_stream[idx * 7];
        float y = particle_data_stream[idx * 7 + 1];
        float dx = particle_data_stream[idx * 7 + 2];
        float dy = particle_data_stream[idx * 7 + 3];
        float mass = particle_data_stream[idx * 7 + 4];
        float atr_force_coeff = particle_data_stream[idx * 7 + 5];
        float rep_force_coeff = particle_data_stream[idx * 7 + 6];

        float force_x = 0.f;
        float force_y = 0.f;

        for (int i = 0; i < particles_num; i++) {
            if (idx == i) continue;

            float other_x = particle_data_stream[i * 7];
            float other_y = particle_data_stream[i * 7 + 1];
            float other_mass = particle_data_stream[i * 7 + 4];

            float dist_x = other_x - x;
            float dist_y = other_y - y;

            if (fabsf(dist_x) > WIN_WIDTH / 2) {
                dist_x = dist_x > 0 ? dist_x - WIN_WIDTH : dist_x + WIN_WIDTH;
            }

            if (fabsf(dist_y) > WIN_HEIGHT / 2) {
                dist_y = dist_y > 0 ? dist_y - WIN_HEIGHT : dist_y + WIN_HEIGHT;
            }

            float distance = sqrtf(dist_x * dist_x + dist_y * dist_y);
            distance = fmaxf(distance, .1f);

            float direction_x = dist_x / distance;
            float direction_y = dist_y / distance;

            if (distance <= atr_force_coeff) {
                float attraction_force = (atr_force_coeff - distance) * mass * other_mass / atr_force_coeff;
                force_x += attraction_force * direction_x;
                force_y += attraction_force * direction_y;
            }

            if (distance <= rep_force_coeff) {
                float repulsion_force = (rep_force_coeff - distance) * mass * other_mass / rep_force_coeff;
                force_x -= repulsion_force * direction_x;
                force_y -= repulsion_force * direction_y;
            }
        }

        dx += force_x / 500000;
        dy += force_y / 500000;

        float new_x = (x + dx);
        float new_y = (y + dy);

        if (new_x < 0) {
            new_x = WIN_WIDTH;
        }
        if (new_x > WIN_WIDTH) {
            new_x = 0;
        }
        if (new_y < 0) {
            new_y = WIN_HEIGHT;
        }
        if (new_y > WIN_HEIGHT) {
            new_y = 0;
        }

        particle_data_stream[idx*7] = new_x;
        particle_data_stream[idx*7+1] = new_y;
        particle_data_stream[idx*7+2] = dx;
        particle_data_stream[idx*7+3] = dy;
    }
}



void task_distributor(thread_pool & pool, std::atomic<bool> & running) {
    while (running) {
        cudaMemcpy(tbb::particle::get_host_particles_data_stream(), tbb::particle::get_device_particles_data_stream(), tbb::particle::get_instance_count() * 7 * sizeof(float), cudaMemcpyDeviceToHost);

        for (size_t i=0; i<tbb::particle::get_instance_count(); i++) {
            pool.enqueue([i] {
                const float x = tbb::particle::get_host_particles_data_stream()[i*7];
                const float y = tbb::particle::get_host_particles_data_stream()[i*7+1];
                tbb::particle::get_instances()[i]->get_shape().setPosition(x, y);
            });
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}


int main() {
    srand(time(nullptr));

    RenderWindow window (VideoMode(1920, 1080), "The Big Bang");

    thread_pool pool (std::thread::hardware_concurrency());
    std::atomic<bool> running (true);

    tbb::particle::initialize();
    tbb::particle::load_particles();

    cudaMemcpy(tbb::particle::get_device_particles_data_stream(), tbb::particle::get_host_particles_data_stream(), tbb::particle::get_instance_count() * 7 * sizeof(float), cudaMemcpyHostToDevice);

    constexpr size_t threads_per_block = 256;
    const size_t blocks_per_grid = (tbb::particle::get_instance_count() + threads_per_block - 1) / threads_per_block;

    std::thread distributor_thread(task_distributor, std::ref(pool), std::ref(running));

    while (window.isOpen()) {
        Event event {};
        while (window.pollEvent(event)) {
            if (event.type == Event::Closed) {
                window.close();
            }
        }

        tbb::launch_simulation<<<blocks_per_grid, threads_per_block>>>(
            tbb::particle::get_device_particles_data_stream(),
            tbb::particle::get_instance_count()
        );

        //cudaDeviceSynchronize();

        window.clear(Color::Black);

        for (size_t i=0; i<tbb::particle::get_instance_count(); i++) {
            window.draw(tbb::particle::get_instances()[i]->get_shape());
        }

        window.display();
    }

    running = false;
    distributor_thread.join();
    tbb::particle::cleanup();
    return 0;
}
