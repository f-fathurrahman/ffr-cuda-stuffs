#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

void sum_arrays_on_host( const int N, double *A, double *B, double *C )
{
    for( int idx = 0; idx < N; idx++ ) {
        C[idx] = A[idx] + B[idx];
    }
}

void initial_data( int Ndata, double *dat )
{
    // Generate different seed for random number
    time_t t;
    srand( (unsigned int) time(&t) );

    for( int i = 0; i < Ndata; i++ ) {
        dat[i] = (double)( rand() & 0xFF )/10.0f;
    }
}

void print_data( int Ndata, double *A )
{
    for( int i = 0; i < Ndata; i++ ) {
        printf("%8d %18.10f\n", i, A[i]);
    }
}

int main( int argc, char **argv)
{
    int Ndata = 10;
    size_t Nbytes = Ndata * sizeof(double);

    double *h_A, *h_B, *h_C;

    h_A = (double*) malloc(Nbytes);
    h_B = (double*) malloc(Nbytes);
    h_C = (double*) malloc(Nbytes);

    initial_data( Ndata, h_A );
    initial_data( Ndata, h_B );

    sum_arrays_on_host( Ndata, h_A, h_B, h_C );

    for( int i = 0; i < Ndata; i++ ) {
        printf("%3d %18.10f %18.10f %18.10f\n", i, h_A[i], h_B[i], h_C[i]);
    }

    return 0;
}
