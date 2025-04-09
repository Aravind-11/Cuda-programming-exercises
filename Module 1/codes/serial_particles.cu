
#include <thrust/universal_vector.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include <chrono>

void update_particles_serial(int num_particles, float dt,
                           const thrust::universal_vector<float> &in,
                                 thrust::universal_vector<float> &out) {
    const float *in_ptr = thrust::raw_pointer_cast(in.data());
    float *out_ptr = thrust::raw_pointer_cast(out.data());

    // Process each particle sequentially
    for(int i = 0; i < num_particles; i++) {
        // Update x position
        out_ptr[i*4 + 0] = in_ptr[i*4 + 0] + dt * in_ptr[i*4 + 2];
        // Update y position
        out_ptr[i*4 + 1] = in_ptr[i*4 + 1] + dt * in_ptr[i*4 + 3];
        // Keep velocities unchanged
        out_ptr[i*4 + 2] = in_ptr[i*4 + 2];
        out_ptr[i*4 + 3] = in_ptr[i*4 + 3];
    }
}

thrust::universal_vector<float> init_particles(int num_particles) {
    thrust::universal_vector<float> particles(num_particles * 4);
    for(int i = 0; i < num_particles; i++) {
        particles[i*4 + 0] = i * 0.1f;     // x position
        particles[i*4 + 1] = i * -0.1f;    // y position
        particles[i*4 + 2] = 1.0f;         // x velocity
        particles[i*4 + 3] = -0.5f;        // y velocity
    }
    return particles;
}

int main() {
    int num_particles = 1000000;  // 1 million particles
    float dt = 0.1f;

    // Initialize particles
    thrust::universal_vector<float> particles = init_particles(num_particles);
    thrust::universal_vector<float> output(particles.size());

    // Measure performance
    auto begin = std::chrono::high_resolution_clock::now();
    update_particles_serial(num_particles, dt, particles, output);
    auto end = std::chrono::high_resolution_clock::now();

    const double seconds = std::chrono::duration<double>(end - begin).count();
    const double gigabytes = static_cast<double>(particles.size() * sizeof(float)) / 1024 / 1024 / 1024;
    const double throughput = gigabytes / seconds;

    std::printf("Serial computation:\n");
    std::printf("Time: %g seconds\n", seconds);
    std::printf("Throughput: %g GB/s\n", throughput);

    return 0;
}
