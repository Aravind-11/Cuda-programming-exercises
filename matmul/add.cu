#include <cstdio>

__global__ void add(int n, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] + 1.0f;
}

int main() {
  int N = 16;
  float *x, *y;
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
  for (int i=0;i<N;i++) x[i]=i;
  add<<<1,32>>>(N,x,y);
  cudaDeviceSynchronize();
  for (int i=0;i<8;i++) printf("%f ", y[i]);
  printf("\n");
  cudaFree(x); cudaFree(y);
  return 0;
}
