#include <SFML/Graphics.hpp>
#include <particle.h>
#include <thread_pool.h>
#include <thread_safe_queue.h>
#include <registers.h>
#include <simulations.h>
#include <vld.h>
#include <mutex>

using namespace sf;


void task_distributor(thread_pool & pool, const std::atomic<bool> & running) {
    while (running) {
        cudaMemcpy(tbb::particle::get_host_particles_pos_stream(), tbb::particle::get_device_particles_pos_stream(), tbb::particle::get_instance_count() * 2 * sizeof(float), cudaMemcpyDeviceToHost);

        for (size_t i=0; i<tbb::particle::get_instance_count(); i++) {
            pool.enqueue([i] {
                const float x = tbb::particle::get_host_particles_pos_stream()[i*2];
                const float y = tbb::particle::get_host_particles_pos_stream()[i*2+1];
                tbb::particle::get_instances()[i]->get_shape().setPosition(x, y);
            });
        }
    }
}


template <size_t Index> void simulation_handler(const std::atomic<bool> & running, const size_t blocks_per_grid) {
    while (running) {
        simulation::launch_simulation<Index><<<blocks_per_grid, threads_per_block>>>(
            tbb::particle::get_device_particles_data_stream(),
            tbb::particle::get_device_particles_pos_stream(),
            tbb::particle::get_instance_count()
        );
        cudaDeviceSynchronize();
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
}


int main() {
    RenderWindow window (VideoMode(1920, 1080), "Particle Physics");
    window.setFramerateLimit(120);

    thread_pool pool (std::thread::hardware_concurrency());
    std::atomic<bool> running (true);

    tbb::particle::initialize();
    tbb::particle::load_particles();

    cudaMemcpy(tbb::particle::get_device_particles_data_stream(), tbb::particle::get_host_particles_init_stream(), tbb::particle::get_instance_count() * 5 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(tbb::particle::get_device_particles_pos_stream(), tbb::particle::get_host_particles_pos_stream(), tbb::particle::get_instance_count() * 2 * sizeof(float), cudaMemcpyHostToDevice);

    const size_t blocks_per_grid = (tbb::particle::get_instance_count() + threads_per_block - 1) / threads_per_block;

    std::thread distributor_thread1(task_distributor, std::ref(pool), std::ref(running));
    std::thread simulator_handler_thread(simulation_handler<0>, std::ref(running), std::ref(blocks_per_grid));

    while (window.isOpen()) {
        Event event {};
        while (window.pollEvent(event)) {
            if (event.type == Event::Closed) {
                window.close();
            }
        }

        window.clear(Color::Black);

        for (size_t i=0; i<tbb::particle::get_instance_count(); i++) {
            window.draw(tbb::particle::get_instances()[i]->get_shape());
        }

        window.display();
    }

    running = false;
    distributor_thread1.join();
    simulator_handler_thread.join();
    tbb::particle::cleanup();
    return 0;
}
