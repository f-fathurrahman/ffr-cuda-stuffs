#include <cstdio>
#include <cmath>

#define N 10

// convert integers 0, 1, ..., N-1 to evenly spaced floats ranging
// from 0 to 1
float scale( int i, int n )
{
    return ((float)i)/(n - 1);
}

// compute distance between 2 points on a line
float distance( float x1, float x2 )
{
    return sqrt( (x2 - x1)*(x2 - x1) );
}

int main()
{
    // Create an array of N floats
    float out[N] = {0.0f};

    float ref = 0.5f;

    printf("ref = %f\n", ref);
    for( int i = 0; i < N; i++ )
    {
        float x = scale(i, N);
        out[i] = distance(x, ref);
        printf("%d %f %f\n", i, x, out[i]);
    }

    printf("Program ended normally\n");
}
