#include <thrust/universal_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <cstdio>

struct Particle {
    cuda::std::pair<float, float> pos;  // position
    cuda::std::pair<float, float> vel;  // velocity
};

// Helper function for vector addition
__host__ __device__
cuda::std::pair<float, float> operator*(const cuda::std::pair<float, float>& vec, float scalar) {
    return cuda::std::make_pair(vec.first * scalar, vec.second * scalar);
}

__host__ __device__
cuda::std::pair<float, float> operator+(
    const cuda::std::pair<float, float>& a, 
    const cuda::std::pair<float, float>& b) {
    return cuda::std::make_pair(a.first + b.first, a.second + b.second);
}

int main() {
    float dt = 0.1f;  // time step

    // Create particles using pairs
    thrust::universal_vector<Particle> particles{
        {cuda::std::make_pair(0.0f, 0.0f), cuda::std::make_pair(1.0f, 0.5f)},
        {cuda::std::make_pair(1.0f, 2.0f), cuda::std::make_pair(-0.5f, 0.2f)},
        {cuda::std::make_pair(-1.0f, -1.0f), cuda::std::make_pair(0.3f, 0.7f)}
    };

    // Update function using vector operations
    auto update_position = [dt] __host__ __device__ (Particle p) {
        p.pos = p.pos + p.vel * dt;
        return p;
    };

    // Print initial state
    printf("Step 0:\n");
    for (int i = 0; i < particles.size(); i++) {
        printf("Particle %d: pos=(%.2f, %.2f) vel=(%.2f, %.2f)\n", 
               i, particles[i].pos.first, particles[i].pos.second, 
               particles[i].vel.first, particles[i].vel.second);
    }

    // Run simulation
    for (int step = 1; step <= 3; step++) {
        thrust::transform(
            thrust::device,
            particles.begin(), 
            particles.end(),
            particles.begin(),
            update_position
        );

        printf("\nStep %d:\n", step);
        for (int i = 0; i < particles.size(); i++) {
            printf("Particle %d: pos=(%.2f, %.2f) vel=(%.2f, %.2f)\n", 
                   i, particles[i].pos.first, particles[i].pos.second,
                   particles[i].vel.first, particles[i].vel.second);
        }
    }
}
