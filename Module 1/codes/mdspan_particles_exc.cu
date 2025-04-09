#include <thrust/universal_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <cstdio>

void simulate_particles(int num_particles, float dt,
                       const thrust::universal_vector<float> &in,
                             thrust::universal_vector<float> &out) {
    const float *in_ptr = thrust::raw_pointer_cast(in.data());

    thrust::transform(
        thrust::device,
        in.begin(), 
        in.end() - num_particles,
        out.begin(),
        [in_ptr, num_particles, dt] __device__ (float val) {
            int idx = &val - in_ptr;
            int particle_idx = idx / 4;
            int component = idx % 4;

            if (component < 2) {  // Position components
                return val + dt * in_ptr[particle_idx * 4 + component + 2];
            }
            return val;  // Velocity components unchanged
        }
    );
}
