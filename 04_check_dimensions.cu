#include <cuda_runtime.h>
#include <stdio.h>

__global__ void check_index() {
    printf("threadIdx: (%d,%d,%d) blockIdx: (%d,%d,%d) blockDim: (%d,%d,%d) gridDim: (%d,%d,%d)\n",
            threadIdx.x, threadIdx.y, threadIdx.z, 
            blockIdx.x, blockIdx.y, blockIdx.z,
            blockDim.x, blockDim.y, blockDim.z,
            gridDim.x, gridDim.y, blockDim.z
    );
    //printf("\n");
}

int main( int argc, char **argv) {

    int Ndata = 10;

    dim3 block(3);
    dim3 grid( (Ndata+block.x-1)/block.x );

    printf("grid: (%d,%d,%d)\n", grid.x, grid.y, grid.z);
    printf("blocks: (%d,%d,%d)\n", block.x, block.y, block.z);

    check_index<<<grid, block>>>();

    cudaDeviceReset();

    return 0;
}