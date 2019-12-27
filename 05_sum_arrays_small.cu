#include <stdio.h>
#include <cuda_runtime.h>

void sum_arrays_on_host( const int N, float *A, float *B, float *C )
{
    for( int idx = 0; idx < N; idx++ ) {
        C[idx] = A[idx] + B[idx];
    }
}

void initial_data( int Ndata, float *dat )
{
    // Generate different seed for random number
    time_t t;
    srand( (unsigned int) time(&t) );

    for( int i = 0; i < Ndata; i++ ) {
        dat[i] = (float)( rand() & 0xFF )/10.0f;
    }
}

__global__ void sum_arrays_on_gpu( float *A, float *B, float *C )
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int check_result( const int N, float *hostRef, float *gpuRef )
{
    double epsilon = 1e-8;
    for( int i = 0; i < N; i++ ) {
        if( abs(hostRef[i] - gpuRef[i]) > epsilon ) {
            printf("Arrays do not match!\n");
            printf("i = %3d , host = %5.2f , device = %5.2f\n", i, hostRef[i], gpuRef[i]);
            return -1;
        }
    }
    return 0;
}


int main( int argc, char** argv )
{
    printf("%s starting ...\n", argv[0]);

    // Set up device
    int dev = 0;
    cudaSetDevice(dev);

    int Ndata = 33;

    // host memory
    size_t Nbytes = Ndata*sizeof(float);

    float *h_A, *h_B, *hostRef, *gpuRef;

    h_A     = (float*) malloc(Nbytes); 
    h_B     = (float*) malloc(Nbytes); 
    hostRef = (float*) malloc(Nbytes);
    gpuRef  = (float*) malloc(Nbytes);

    // Initialize data at host side
    initial_data( Ndata, h_A );
    initial_data( Ndata, h_B );

    memset( hostRef, 0, Nbytes );
    memset( gpuRef, 0, Nbytes );

    // Initialize device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc( (float**)&d_A, Nbytes );
    cudaMalloc( (float**)&d_B, Nbytes );
    cudaMalloc( (float**)&d_C, Nbytes );

    // transfer data
    cudaMemcpy( d_A, h_A, Nbytes, cudaMemcpyHostToDevice );
    cudaMemcpy( d_B, h_B, Nbytes, cudaMemcpyHostToDevice );

    // invoke kernel
    dim3 block(Ndata);
    dim3 grid(Ndata/block.x);

    sum_arrays_on_gpu<<<grid,block>>>( d_A, d_B, d_C );

    printf("Execution configuration: <<<%d,%d>>>\n", grid.x, block.x);

    // 
    sum_arrays_on_host( Ndata, h_A, h_B, hostRef );

    cudaMemcpy( gpuRef, d_C, Nbytes, cudaMemcpyDeviceToHost );

    int err = check_result( Ndata, hostRef, gpuRef );
    if( err == 0 ) {
        printf("Test passed\n");
    }

    cudaFree( d_A );
    cudaFree( d_B );
    cudaFree( d_B );

    cudaDeviceReset();

    return 0;
}