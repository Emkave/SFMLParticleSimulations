#include <simulations.h>
#include <registers.h>

__device__ void simulation::simulation1(float * particle_data_stream, float * particle_pos_stream, void * particle_extra_stream, const size_t particles_num) {
    if (const unsigned idx = threadIdx.x + blockIdx.x * blockDim.x; idx < particles_num) {
        const float x = particle_pos_stream[idx*2];
        const float y = particle_pos_stream[idx*2+1];
        float dx = particle_data_stream[idx*4];
        float dy = particle_data_stream[idx*4+1];
        const float atr_force_coeff = particle_data_stream[idx*4+2];
        const float rep_force_coeff = particle_data_stream[idx*4+3];
        const float mass = static_cast<float*>(particle_extra_stream)[idx];

        float force_x = 0.f;
        float force_y = 0.f;

        for (int i = 0; i < particles_num; i++) {
            if (idx == i) continue;

            const float other_x = particle_pos_stream[i*2];
            const float other_y = particle_pos_stream[i*2+1];
            const float other_mass = static_cast<float*>(particle_extra_stream)[i];

            float dist_x = other_x - x;
            float dist_y = other_y - y;

            if (fabsf(dist_x) > static_cast<float>(WIN_WIDTH) / 2) {
                dist_x = dist_x > 0 ? dist_x - WIN_WIDTH : dist_x + WIN_WIDTH;
            }

            if (fabsf(dist_y) > static_cast<float>(WIN_HEIGHT) / 2) {
                dist_y = dist_y > 0 ? dist_y - WIN_HEIGHT : dist_y + WIN_HEIGHT;
            }

            float distance = sqrtf(dist_x * dist_x + dist_y * dist_y);
            distance = fmaxf(distance, .0001f);

            const float direction_x = dist_x / distance;
            const float direction_y = dist_y / distance;

            //if (distance <= atr_force_coeff) {
                const float attraction_force = (atr_force_coeff) / distance;
                force_x += attraction_force * direction_x;
                force_y += attraction_force * direction_y;
            //}

            //if (distance <= rep_force_coeff) {
                const float repulsion_force = (rep_force_coeff) / distance * 10;
                force_x -= repulsion_force * direction_x;
                force_y -= repulsion_force * direction_y;
            //}
        }

        dx += force_x / 10000;
        dy += force_y / 10000;

        if (const float speed = sqrtf(dx * dx + dy * dy); speed > 5) {
            const float scale = 1 / speed;
            dx *= scale;
            dy *= scale;
        }

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

        particle_pos_stream[idx*2] = new_x;
        particle_pos_stream[idx*2+1] = new_y;
        particle_data_stream[idx*4] = dx;
        particle_data_stream[idx*4+1] = dy;
    }
}
