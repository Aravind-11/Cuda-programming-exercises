
#include <iostream>
#include <cuda_runtime.h>

// Define the size of the arrays
const int N = 1 << 20; // 1,048,576 elements

// --- KERNEL DEFINITION (Executed on Device) ---
// The kernel is the parallel function
__global__ void vectorAdd(const int *A, const int *B, int *C, int size) {
    // Calculate the unique global index 'i' for this thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Safety check: ensure the index is within the array bounds
    if (i < size) {
        C[i] = A[i] + B[i];
    }
}

// --- HOST MAIN FUNCTION (Executed on CPU) ---
int main() {
    // --- 0. Setup and Initialization ---
    // Memory size in bytes
    const int BYTES = N * sizeof(int);

    // Host Pointers (CPU memory, allocated with standard malloc)
    int *h_A, *h_B, *h_C;
    // Device Pointers (GPU memory, allocated with cudaMalloc)
    int *d_A, *d_B, *d_C;

    // Allocate Host memory
    h_A = (int*)malloc(BYTES);
    h_B = (int*)malloc(BYTES);
    h_C = (int*)malloc(BYTES);

    // Initialize Host arrays A and B
    for (int i = 0; i < N; i++) {
        h_A[i] = i;      // A = [0, 1, 2, 3, ...]
        h_B[i] = i * 2;  // B = [0, 2, 4, 6, ...]
    }
    std::cout << "Data initialized on Host (CPU)." << std::endl;

    // 1. MEMORY ALLOCATION (Device/GPU)
    cudaMalloc((void**)&d_A, BYTES);
    cudaMalloc((void**)&d_B, BYTES);
    cudaMalloc((void**)&d_C, BYTES);
    std::cout << "Memory allocated on Device (GPU)." << std::endl;

    // 2. DATA TRANSFER (Host -> Device)
    // Copy the initialized input arrays A and B from CPU to GPU
    cudaMemcpy(d_A, h_A, BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, BYTES, cudaMemcpyHostToDevice);
    std::cout << "Input data copied to Device." << std::endl;

    // --- 3. KERNEL LAUNCH SETUP ---
    // Define the execution configuration
    int threadsPerBlock = 256; 
    // Calculate the grid size (number of blocks) using ceiling division
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; 

    // 4. KERNEL LAUNCH (The work begins on the GPU)
    std::cout << "Launching kernel: " << blocksPerGrid << " blocks, " 
              << threadsPerBlock << " threads/block." << std::endl;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Wait for the GPU to finish execution
    cudaDeviceSynchronize();

    // 5. DATA TRANSFER (Device -> Host)
    // Copy the result array C back from GPU to CPU
    cudaMemcpy(h_C, d_C, BYTES, cudaMemcpyDeviceToHost);
    std::cout << "Result copied back to Host." << std::endl;

    // --- 6. Validation and Output (Host) ---
    // Check first few results: C[i] = i + 2*i = 3*i
    if (h_C[0] == 0 && h_C[1] == 3 && h_C[N-1] == 3 * (N-1)) {
        std::cout << "SUCCESS! Vector addition verified." << std::endl;
        std::cout << "Example: C[1] = " << h_C[1] << " (Expected 3)" << std::endl;
        std::cout << "Example: C[" << N-1 << "] = " << h_C[N-1] << std::endl;
    } else {
        std::cerr << "FAILURE! Results did not match." << std::endl;
    }

    // 7. CLEANUP
    // Free Host memory
    free(h_A); 
    free(h_B); 
    free(h_C); 
    // Free Device memory
    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_C); 
    std::cout << "Cleanup complete. Exiting." << std::endl;

    return 0;
}

