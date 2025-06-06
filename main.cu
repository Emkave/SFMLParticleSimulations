#include <SFML/Graphics.hpp>
#include <particle.h>
#include <thread_pool.h>
#include <thread_safe_queue.h>
#include <registers.h>
#include <simulations.h>
#include <vld.h>
#include <mutex>

using namespace sf;

using particle_class = tbb::particle<registers::particle_class_index>;

void task_distributor(thread_pool & pool, const std::atomic<bool> & running) {
    while (running) {
        cudaMemcpy(particle_class::get_host_particles_pos_stream(), particle_class::get_device_particles_pos_stream(), particle_class::get_instance_count() * 2 * sizeof(float), cudaMemcpyDeviceToHost);

        for (size_t i=0; i<particle_class::get_instance_count(); i++) {
            pool.enqueue([i] {
                const float x = particle_class::get_host_particles_pos_stream()[i*2];
                const float y = particle_class::get_host_particles_pos_stream()[i*2+1];
                particle_class::get_instances()[i]->get_shape().setPosition(x, y);
            });
        }
    }
}

template <size_t Index> void simulation_handler(const std::atomic<bool> & running, const size_t blocks_per_grid) {
    while (running) {
        simulation::launch_simulation<Index><<<blocks_per_grid, registers::threads_per_block>>>(
            particle_class::get_device_particles_data_stream(),
            particle_class::get_device_particles_pos_stream(),
            particle_class::get_device_particles_extra_stream(),
            particle_class::get_instance_count()
        );
        cudaDeviceSynchronize();
        std::this_thread::sleep_for(std::chrono::microseconds(5));
    }
}


int main() {
    RenderWindow window (VideoMode(1920, 1080), "Particle Physics");
    window.setFramerateLimit(120);

    thread_pool pool (std::thread::hardware_concurrency());
    std::atomic<bool> running (true);

    particle_class::load_particles();
    particle_class::load_to_device();

    const size_t blocks_per_grid = (particle_class::get_instance_count() + registers::threads_per_block - 1) / registers::threads_per_block;

    std::thread distributor_thread1(task_distributor, std::ref(pool), std::ref(running));
    std::thread simulator_handler_thread(simulation_handler<registers::simulation_index>, std::ref(running), std::ref(blocks_per_grid));

    while (window.isOpen()) {
        Event event {};
        while (window.pollEvent(event)) {
            if (event.type == Event::Closed) {
                window.close();
            }
        }

        window.clear(Color::Black);

        for (size_t i=0; i<particle_class::get_instance_count(); i++) {
            window.draw(particle_class::get_instances()[i]->get_shape());
        }

        window.display();
    }

    running = false;
    distributor_thread1.join();
    simulator_handler_thread.join();
    particle_class::cleanup();
    return 0;
}
