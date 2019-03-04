#include <stdio.h>

__global__ void hello_from_gpu()
{
  printf("This is hello from GPU\n");
}

int main()
{
  printf("This is hello from CPU\n");

  hello_from_gpu <<<1,10>>> ();

  cudaDeviceReset();

  return 0;
}
