#include <SFML/Graphics.hpp>
#include <thread_pool.h>
#include <registers.h>
#include <particle.h>
#include <simulations.h>
#ifdef _WIN32
#include <vld.h>
#endif
#include <mutex>
#include <atomic>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <iostream>

using namespace sf;

std::atomic<size_t> sim_cycle = 0;


void task_distributor(thread_pool & pool, const bool & running) {
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


void simulation_handler(const bool & running, const size_t blocks_per_grid) {
    std::ofstream entropy_file ("../saves/entropy.txt", std::ios::trunc);
    entropy_file.clear();
    std::ofstream gravity_file ("../saves/gravity.txt", std::ios::trunc);
    gravity_file.clear();
    std::ofstream matter_density_file ("../saves/matter_density.txt", std::ios::trunc);
    matter_density_file.clear();

    gravity_file.rdbuf()->pubsetbuf(0, 0);

    float * analysis_particle_holder_host = static_cast<float*>(malloc(tbb::particle::get_instance_count() * sizeof(float) * 4));
    float * analysis_particle_pos_holder_host = static_cast<float*>(malloc(tbb::particle::get_instance_count() * sizeof(float) * 2));
    float * analysis_particle_extra_holder_host = static_cast<float*>(malloc(tbb::particle::get_instance_count() * sizeof(float)));
    float stat_grid[WIN_HEIGHT][WIN_WIDTH] = {0};

    while (running) {
        simulation::launch_simulation<<<blocks_per_grid, registers::threads_per_block>>>(
            tbb::particle::get_device_particles_data_stream(),
            tbb::particle::get_device_particles_pos_stream(),
            tbb::particle::get_device_particles_extra_stream(),
            tbb::particle::get_instance_count()
        );
        cudaDeviceSynchronize();

        cudaMemcpy(analysis_particle_pos_holder_host, tbb::particle::get_device_particles_pos_stream(), tbb::particle::get_instance_count() * sizeof(float) * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(analysis_particle_holder_host, tbb::particle::get_device_particles_data_stream(), tbb::particle::get_instance_count() * sizeof(float) * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(analysis_particle_extra_holder_host, tbb::particle::get_device_particles_extra_stream(), tbb::particle::get_instance_count() * sizeof(float), cudaMemcpyDeviceToHost);

        for (size_t p = 0; p < tbb::particle::get_instance_count(); ++p) {
            float x = analysis_particle_pos_holder_host[p * 2];
            float y = analysis_particle_pos_holder_host[p * 2 + 1];
            float attr = analysis_particle_holder_host[p * 4 + 2];  // attr_force_coeff
            float mass = analysis_particle_extra_holder_host[p];

            for (int gy = 0; gy < WIN_HEIGHT; ++gy) {
                for (int gx = 0; gx < WIN_WIDTH; ++gx) {
                    float dx = gx - x;
                    float dy = gy - y;

                    // Wrap screen around (toroidal space)
                    if (std::abs(dx) > WIN_WIDTH / 2) dx -= std::copysign(WIN_WIDTH, dx);
                    if (std::abs(dy) > WIN_HEIGHT / 2) dy -= std::copysign(WIN_HEIGHT, dy);

                    float dist_sq = dx * dx + dy * dy;
                    dist_sq = std::max(dist_sq, 1.0f); // prevent division by zero

                    float gravity = (attr * mass) / dist_sq;
                    stat_grid[gy][gx] += gravity;
                }
            }
        }

        for (size_t i=0; i<WIN_HEIGHT; i++) {
            for (size_t j=0; j<WIN_WIDTH; j++) {
                gravity_file << sim_cycle << ' ' << i << ' ' << j << ' ' << stat_grid[i][j] << '\n';
            }
            gravity_file.flush();
        }


        for (size_t i=0; i<WIN_HEIGHT; i++) {
            for (size_t j=0; j<WIN_WIDTH; j++) {
                stat_grid[i][j] = 0.0f;
            }
        }


        for (size_t i = 0; i < tbb::particle::get_instance_count(); ++i) {
            float x1 = analysis_particle_pos_holder_host[i * 2];
            float y1 = analysis_particle_pos_holder_host[i * 2 + 1];

            for (size_t j = i + 1; j < tbb::particle::get_instance_count(); ++j) {
                float x2 = analysis_particle_pos_holder_host[j * 2];
                float y2 = analysis_particle_pos_holder_host[j * 2 + 1];

                // Compute shortest toroidal distance (screen wrap)
                float dx = x2 - x1;
                float dy = y2 - y1;

                if (std::abs(dx) > WIN_WIDTH / 2) dx -= std::copysign(WIN_WIDTH, dx);
                if (std::abs(dy) > WIN_HEIGHT / 2) dy -= std::copysign(WIN_HEIGHT, dy);

                float dist_sq = dx * dx + dy * dy;
                float dist = std::sqrt(std::max(dist_sq, 1.0f)); // prevent div by zero

                // Midpoint of the two particles (modulo wrap to keep in bounds)
                float cx = std::fmod((x1 + dx / 2.0f + WIN_WIDTH), WIN_WIDTH);
                float cy = std::fmod((y1 + dy / 2.0f + WIN_HEIGHT), WIN_HEIGHT);

                // Density contribution
                float density = 1.0f / (dist * dist * dist);

                // Map to grid index
                size_t gx = static_cast<size_t>(cx);
                size_t gy = static_cast<size_t>(cy);

                if (gx < WIN_WIDTH && gy < WIN_HEIGHT) {
                    stat_grid[gy][gx] += density;
                }
            }
        }

        for (size_t i = 0; i < WIN_HEIGHT; ++i) {
            for (size_t j = 0; j < WIN_WIDTH; ++j) {
                matter_density_file << sim_cycle << ' ' << i << ' ' << j << ' ' << stat_grid[i][j] << '\n';
            }
            matter_density_file.flush();
        }


        for (size_t i=0; i<WIN_HEIGHT; i++) {
            for (size_t j=0; j<WIN_WIDTH; j++) {
                stat_grid[i][j] = 0.0f;
            }
        }

        for (size_t p = 0; p < tbb::particle::get_instance_count(); ++p) {
            float x = analysis_particle_pos_holder_host[p * 2];
            float y = analysis_particle_pos_holder_host[p * 2 + 1];

            size_t gx = static_cast<size_t>(x) % WIN_WIDTH;
            size_t gy = static_cast<size_t>(y) % WIN_HEIGHT;

            stat_grid[gy][gx]++;
        }

        float entropy = 0.0f;
        float total_particles = static_cast<float>(tbb::particle::get_instance_count());

        for (size_t i = 0; i < WIN_HEIGHT; ++i) {
            for (size_t j = 0; j < WIN_WIDTH; ++j) {
                float count = static_cast<float>(stat_grid[i][j]);
                if (count > 0) {
                    float p = count / total_particles;
                    entropy -= p * std::log2(p);
                }
            }
        }

        entropy_file << sim_cycle << ' ' << entropy << '\n';
        entropy_file.flush();

        //std::this_thread::sleep_for(std::chrono::microseconds(5));
        ++sim_cycle;
    }

    free(analysis_particle_holder_host);
    free(analysis_particle_extra_holder_host);
    free(analysis_particle_pos_holder_host);
    entropy_file.close();
    gravity_file.close();
    matter_density_file.close();
}


int main() {
    RenderWindow window (VideoMode(WIN_WIDTH, WIN_HEIGHT), "Particle Physics");
    window.setFramerateLimit(120);

    thread_pool pool (std::thread::hardware_concurrency());
    bool running = true;

    tbb::particle::load_particles();
    tbb::particle::load_to_device();

    const size_t blocks_per_grid = (tbb::particle::get_instance_count() + registers::threads_per_block - 1) / registers::threads_per_block;

    std::thread distributor_thread(task_distributor, std::ref(pool), std::ref(running));
    std::thread simulator_handler_thread(simulation_handler, std::ref(running), std::ref(blocks_per_grid));

    while (window.isOpen() && sim_cycle < simulation_cycles) {
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
    distributor_thread.join();
    simulator_handler_thread.join();
    tbb::particle::cleanup();

    return 0;
}
