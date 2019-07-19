#include "aux_functions.h"

#include <cstdio>
#include <cstdlib>

#define N 200

int main()
{
    float *in = (float*)calloc(N, sizeof(float));
    float *out = (float*)calloc(N, sizeof(float));

    const float ref = 0.5f;

    for(int i = 0; i < N; i++)
    {
        in[i] = scale(i, N);
    }

    distanceArray(out, in, ref, N);

    // free memory
    free(in);
    free(out);

    printf("Program ended normally\n");

    return 0;
}