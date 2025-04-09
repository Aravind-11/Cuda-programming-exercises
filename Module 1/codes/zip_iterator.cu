#include <cstdio>
#include <array>
#include <tuple>

// An iterator that combines elements from two arrays
struct ZipIterator {
    float* positions;
    float* velocities;

    // Returns a tuple containing the position and velocity
    __host__ __device__
    std::tuple<float, float> operator[](int i) {
        return {positions[i], velocities[i]};
    }
};

int main() {
    std::array<float, 3> positions{0.0f, 1.0f, 2.0f};
    std::array<float, 3> velocities{0.5f, -0.3f, 1.0f};

    ZipIterator zipIter{positions.data(), velocities.data()};

    for (int i = 0; i < 3; i++) {
        auto [pos, vel] = zipIter[i];
        printf("Particle %d: position=%.1f, velocity=%.1f\n", i, pos, vel);
    }

    return 0;
}
