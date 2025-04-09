
#include <thrust/universal_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <cstdio>
#include <chrono>

struct Vector2D {
    float x, y;

    __host__ __device__
    Vector2D operator+(const Vector2D& other) const {
        return {x + other.x, y + other.y};
    }

    __host__ __device__
    Vector2D operator*(float scalar) const {
        return {x * scalar, y * scalar};
    }
};

// Serial implementation for comparison
void update_force_field_serial(int num_particles,
                             const thrust::universal_vector<float>& positions,
                             thrust::universal_vector<float>& forces) {
    const float* pos = thrust::raw_pointer_cast(positions.data());
    float* force = thrust::raw_pointer_cast(forces.data());

    const float k_repulsive = 0.1f;  // Repulsive force constant
    const float k_circular = 0.2f;    // Circular force constant

    for (int i = 0; i < num_particles; i++) {
        float x = pos[i*2];
        float y = pos[i*2 + 1];

        // Calculate distance from origin
        float r2 = x*x + y*y;
        float r = sqrt(r2);

        // Avoid division by zero
        if (r < 1e-6f) {
            force[i*2] = 0.0f;
            force[i*2 + 1] = 0.0f;
            continue;
        }

        // Repulsive force
        float fr_x = k_repulsive * r * x/r;
        float fr_y = k_repulsive * r * y/r;

        // Circular force (perpendicular to position)
        float fc_x = -k_circular * y;
        float fc_y = k_circular * x;

        // Combine forces
        force[i*2] = fr_x + fc_x;
        force[i*2 + 1] = fr_y + fc_y;
    }
}

// Parallel implementation using thrust
struct ForceFieldCalculator {
    float k_repulsive;
    float k_circular;

    ForceFieldCalculator(float kr, float kc) 
        : k_repulsive(kr), k_circular(kc) {}

    __host__ __device__
    thrust::tuple<float, float> operator()(const thrust::tuple<float, float>& pos) const {
        float x = thrust::get<0>(pos);
        float y = thrust::get<1>(pos);

        // Calculate distance from origin
        float r2 = x*x + y*y;
        float r = sqrt(r2);

        // Handle particles at origin
        if (r < 1e-6f) {
            return thrust::make_tuple(0.0f, 0.0f);
        }

        // Repulsive force
        float fr_x = k_repulsive * r * x/r;
        float fr_y = k_repulsive * r * y/r;

        // Circular force
        float fc_x = -k_circular * y;
        float fc_y = k_circular * x;

        // Return combined forces
        return thrust::make_tuple(fr_x + fc_x, fr_y + fc_y);
    }
};

void update_force_field_parallel(int num_particles,
                               const thrust::universal_vector<float>& positions,
                               thrust::universal_vector<float>& forces) {
    const float k_repulsive = 0.1f;
    const float k_circular = 0.2f;

    // Create zip iterators for positions
    auto pos_begin = thrust::make_zip_iterator(
        thrust::make_tuple(
            positions.begin(),
            positions.begin() + 1
        )
    );

    // Create transform iterator for force calculation
    auto force_it = thrust::make_transform_iterator(
        pos_begin,
        ForceFieldCalculator(k_repulsive, k_circular)
    );

    // Copy calculated forces to output
    thrust::copy(
        thrust::device,
        force_it,
        force_it + num_particles,
        thrust::make_zip_iterator(
            thrust::make_tuple(
                forces.begin(),
                forces.begin() + 1
            )
        )
    );
}

// Initialize particles in a grid pattern
thrust::universal_vector<float> init_particles(int num_particles) {
    thrust::universal_vector<float> positions(num_particles * 2);

    int grid_size = static_cast<int>(sqrt(num_particles));
    float spacing = 2.0f / grid_size;

    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            int idx = (i * grid_size + j) * 2;
            if (idx + 1 < positions.size()) {
                positions[idx] = (i - grid_size/2) * spacing;
                positions[idx + 1] = (j - grid_size/2) * spacing;
            }
        }
    }
    return positions;
}

int main() {
    int num_particles = 1000000;  // 1 million particles

    // Initialize particles
    thrust::universal_vector<float> positions = init_particles(num_particles);
    thrust::universal_vector<float> forces(positions.size());

    // Measure serial performance
    auto begin = std::chrono::high_resolution_clock::now();
    update_force_field_serial(num_particles, positions, forces);
    auto end = std::chrono::high_resolution_clock::now();

    double seconds = std::chrono::duration<double>(end - begin).count();
    double gigabytes = static_cast<double>(positions.size() * sizeof(float)) / 1024 / 1024 / 1024;
    double throughput = gigabytes / seconds;

    std::printf("Serial Force Field Computation:\n");
    std::printf("Time: %g seconds\n", seconds);
    std::printf("Throughput: %g GB/s\n\n", throughput);

    // Measure parallel performance
    begin = std::chrono::high_resolution_clock::now();
    update_force_field_parallel(num_particles, positions, forces);
    end = std::chrono::high_resolution_clock::now();

    seconds = std::chrono::duration<double>(end - begin).count();
    throughput = gigabytes / seconds;

    std::printf("Parallel Force Field Computation:\n");
    std::printf("Time: %g seconds\n", seconds);
    std::printf("Throughput: %g GB/s\n", throughput);

    return 0;
}
