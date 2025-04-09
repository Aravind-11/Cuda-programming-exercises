#include <cstdio>
#include <array>
#include <tuple>
#include <cmath>

struct Particle {
    float x, y;    // position
};

// Zip iterator to combine two particle arrays
struct ParticleZipIterator {
    Particle* before;
    Particle* after;

    // Returns a tuple containing both particle states
    __host__ __device__
    std::tuple<Particle, Particle> operator[](int i) {
        return {before[i], after[i]};
    }
};

// Displacement iterator that computes distance between particle positions
struct DisplacementIterator {
    ParticleZipIterator zip;

    // Computes distance without storing intermediate results
    __host__ __device__
    float operator[](int i) {
        auto [p1, p2] = zip[i];
        float dx = p2.x - p1.x;
        float dy = p2.y - p1.y;
        return sqrt(dx*dx + dy*dy);
    }
};

int main() {
    Particle before[3] = {
        {0.0f, 0.0f},
        {1.0f, 2.0f},
        {-1.0f, -1.0f}
    };

    Particle after[3] = {
        {0.1f, 0.05f},
        {0.95f, 2.02f},
        {-0.97f, -0.93f}
    };

    ParticleZipIterator zipIter{before, after};
    DisplacementIterator dispIter{zipIter};

    for (int i = 0; i < 3; i++) {
        printf("Particle %d displacement: %.4f\n", i, dispIter[i]);
    }

    // Find maximum displacement manually
    float max_disp = 0.0f;
    for (int i = 0; i < 3; i++) {
        max_disp = fmax(max_disp, dispIter[i]);
    }

    printf("Maximum displacement: %.4f\n", max_disp);

    return 0;
}
