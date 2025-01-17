#include <SFML/Graphics.hpp>
#include <particle.h>
#include <vld.h>
#include <iostream>

using namespace sf;

__global__ void tbb::launch_simulation(float * particle_data_stream, size_t particles_num) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < particles_num) {
        float x = particle_data_stream[idx * 7];
        float y = particle_data_stream[idx * 7 + 1];
        float dx = particle_data_stream[idx * 7 + 2];
        float dy = particle_data_stream[idx * 7 + 3];

        // Example: Update positions based on velocities
        x += dx;
        y += dy;

        // Reflect off boundaries (simple boundary conditions)
        if (x < 0 || x > 800) dx = -dx;
        if (y < 0 || y > 600) dy = -dy;

        // Write updated data back to Unified Memory
        particle_data_stream[idx * 7] = x;
        particle_data_stream[idx * 7 + 1] = y;
        particle_data_stream[idx * 7 + 2] = dx;
        particle_data_stream[idx * 7 + 3] = dy;
    }
}


int main() {
    RenderWindow window (VideoMode(800, 600), "The Big Bang");

    tbb::particle p1 (100.f, 100.f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    tbb::particle p2 (100.f, 100.f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    const size_t threads_per_block = 256;
    const size_t blocks_per_grid = (tbb::particle::get_instance_count() + threads_per_block - 1) / threads_per_block;

    while (window.isOpen()) {
        Event event {};
        while (window.pollEvent(event)) {
            if (event.type == Event::Closed) {
                window.close();
            }
        }

        tbb::launch_simulation<<<blocks_per_grid, threads_per_block>>>(
            tbb::particle::get_particles_data_stream(),
            tbb::particle::get_instance_count()
        );

        cudaDeviceSynchronize();

        window.clear(Color::Black);

        for (size_t i=0; i<tbb::particle::get_instance_count(); i++) {
            tbb::particle::get_instances()[i]->get_shape().setPosition(
                tbb::particle::get_particles_data_stream()[i*7],
                tbb::particle::get_particles_data_stream()[i*7+1]
            );

            window.draw(tbb::particle::get_instances()[i]->get_shape());
        }

        window.display();
    }

    return 0;
}
