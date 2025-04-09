#include <cstdio>
#include <array>

// An iterator that transforms values from an array
struct TransformIterator {
    float* data;

    // Multiplies each value by 2
    __host__ __device__
    float operator[](int i) {
        return data[i] * 2.0f;
    }
};

int main() {
    std::array<float, 3> values{1.0f, 2.0f, 3.0f};

    TransformIterator transformer{values.data()};

    printf("Original: %.1f, Transformed: %.1f\n", values[0], transformer[0]);  // 1.0, 2.0
    printf("Original: %.1f, Transformed: %.1f\n", values[1], transformer[1]);  // 2.0, 4.0
    printf("Original: %.1f, Transformed: %.1f\n", values[2], transformer[2]);  // 3.0, 6.0

    return 0;
}
