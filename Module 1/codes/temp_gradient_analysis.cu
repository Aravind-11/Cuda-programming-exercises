#include <thrust/universal_vector.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cstdio>

struct CoreReading {
    float current_temp;    // Current temperature
    float previous_temp;   // Temperature from last reading
    float previous_emtg;   // Previous gradient value
    int core_id;          // Core identifier
};

struct EMTGResult {
    float gradient;
    int core_id;
};

// TODO: Implement this function
thrust::universal_vector<EMTGResult> calculate_emtg(
    const thrust::universal_vector<CoreReading>& readings,
    float alpha = 0.3f  // Smoothing factor
) {
    // Create results vector
    thrust::universal_vector<EMTGResult> results(readings.size());

    // TODO: Transform readings into EMTG results

    return results;
}

int main() {
    // Example temperature readings for 4 GPU cores
    thrust::universal_vector<CoreReading> readings{
        {85.0f, 80.0f, 0.0f, 0},  // Core 0
        {82.0f, 79.0f, 0.0f, 1},  // Core 1
        {88.0f, 81.0f, 0.0f, 2},  // Core 2
        {83.0f, 80.0f, 0.0f, 3}   // Core 3
    };

    // Calculate EMTG for all cores
    auto results = calculate_emtg(readings);

    // Find core with highest gradient
    auto it = thrust::max_element(
        thrust::device,
        results.begin(), 
        results.end(),
        [] __device__ (const EMTGResult& a, const EMTGResult& b) {
            return a.gradient < b.gradient;
        }
    );

    // Get index of max element
    size_t max_idx = it - results.begin();

    // Print results
    printf("Core %d has highest gradient: %.2f°C/s\n", 
           results[max_idx].core_id, 
           results[max_idx].gradient);
}
