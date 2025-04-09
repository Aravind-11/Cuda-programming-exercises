#include <cstdio>

// A simple iterator that generates a sequence of integers
struct CountingIterator {
    // Returns the value at index i
    __host__ __device__
    int operator[](int i) {
        return i;
    }
};

int main() {
    CountingIterator counter;

    printf("counter[0]: %d\n", counter[0]);  // Prints 0
    printf("counter[5]: %d\n", counter[5]);  // Prints 5
    printf("counter[10]: %d\n", counter[10]);  // Prints 10

    return 0;
}
