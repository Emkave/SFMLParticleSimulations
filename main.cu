#include <SFML/Graphics.hpp>
#include <particle.h>
#include <thread_pool.h>
#include <thread_safe_queue.h>
#include <vld.h>
#include <iostream>
#include <queue>
#include <mutex>

#define WIN_WIDTH 1920
#define WIN_HEIGHT 1080

using namespace sf;

__global__ void tbb::launch_simulation(const float * particle_data_stream, size_t particles_num) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < particles_num) {
        // Current particle properties
        float x = particle_data_stream[idx * 7];
        float y = particle_data_stream[idx * 7 + 1];
        float dx = particle_data_stream[idx * 7 + 2];
        float dy = particle_data_stream[idx * 7 + 3];
        float mass = particle_data_stream[idx * 7 + 4];
        float atr_dist = particle_data_stream[idx * 7 + 5];
        float rep_dist = particle_data_stream[idx * 7 + 6];





    }
}



void task_distributor(thread_pool & pool, std::atomic<bool> & running) {
    while (running) {
        cudaMemcpy(tbb::particle::get_host_particles_data_stream(), tbb::particle::get_device_particles_data_stream(), tbb::particle::get_instance_count() * 7 * sizeof(float), cudaMemcpyDeviceToHost);

        for (size_t i=0; i<tbb::particle::get_instance_count(); i++) {
            pool.enqueue([i] {
                float x = tbb::particle::get_host_particles_data_stream()[i*7];
                float y = tbb::particle::get_host_particles_data_stream()[i*7+1];
                tbb::particle::get_instances()[i]->get_shape().setPosition(x, y);
            });
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}


int main() {
    RenderWindow window (VideoMode(1920, 1080), "The Big Bang");

    thread_pool pool (std::thread::hardware_concurrency());
    std::atomic<bool> running (true);

    tbb::particle::initialize();
    for (size_t i = 0; i < tbb::max_instances; i++) {
        const float x = static_cast<float>(rand() % WIN_WIDTH);
        const float y = static_cast<float>(rand() % WIN_HEIGHT);

        const float dx = (rand() % 2) ? -1 : 1;
        const float dy = (rand() % 2) ? -1 : 1;

        const float mass = 0;//static_cast<float>((rand() % 50 + 100)) / 100.0f; // Range [0.1, 0.6]
        const float atr_dist = .0;  // Scale with mass
        const float rep_dist = .0;  // Fixed small repulsion

        tbb::particle::get_instances()[i] = std::make_unique<tbb::particle>(x, y, dx, dy, mass, atr_dist, rep_dist);
    }

    cudaMemcpy(tbb::particle::get_device_particles_data_stream(), tbb::particle::get_host_particles_data_stream(), tbb::particle::get_instance_count() * 7 * sizeof(float), cudaMemcpyHostToDevice);

    const size_t threads_per_block = 256;
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

        cudaDeviceSynchronize();

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
