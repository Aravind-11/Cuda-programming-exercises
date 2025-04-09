
#include <thrust/universal_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_output_iterator.h>

// Structure to track force and count for averaging
struct ForceAccumulator {
    float force_sum;
    int count;

    __host__ __device__
    ForceAccumulator() : force_sum(0.0f), count(0) {}

    __host__ __device__
    ForceAccumulator operator+(const ForceAccumulator& other) const {
        return {force_sum + other.force_sum, count + other.count};
    }
};

// Current implementation that needs to be modified
struct mean_force_functor {
    __host__ __device__
    float operator()(const ForceAccumulator& acc) const {
        return acc.count > 0 ? acc.force_sum / acc.count : 0.0f;
    }
};

thrust::universal_vector<float> calculate_region_forces(
    int grid_size,
    const thrust::universal_vector<float>& positions,
    const thrust::universal_vector<float>& forces) 
{
    thrust::universal_vector<float> region_means(grid_size * grid_size);

    // TODO: Replace this with transform_output_iterator approach
    // Current inefficient two-step approach:
    // 1. First reduces forces into regions
    // 2. Then transforms totals into means

    // Your implementation here

    return region_means;
}
