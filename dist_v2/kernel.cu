#include "kernel.h"
#include <cstdio>
#define TPB 32

__device__ float distance(float x1, float x2)
{
    return sqrt( (x2 - x1)*(x2 - x1) );
}

__global__ void distanceKernel(float *d_out, float *d_in, float ref)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const float x = d_in[i];

    d_out[i] = distance(x, ref);

    printf("%2d %18.10f %18.10f %18.10f\n", i, ref, x, d_out[i]);
}

void distanceArray(float *out, float *in, float ref, int len)
{
    float *d_in = 0;
    float *d_out = 0;

    cudaMalloc(&d_in, len*sizeof(float));
    cudaMalloc(&d_out, len*sizeof(float));

    // copy input data from host to device
    cudaMemcpy(d_in, in, len*sizeof(float), cudaMemcpyHostToDevice);

    // launch the kernel
    distanceKernel<<<len/TPB, TPB>>>(d_out, d_in, ref);

    // copy the result
    cudaMemcpy(out, d_out, len*sizeof(float), cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_in);
    cudaFree(d_out);
}
