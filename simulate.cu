#include <iostream>

#include "cuda.h"

#include "animal.h"

#define DIM 1024
#define ANIMAL_START_COUNT 100

int iDivUp(int hostPtr, int b){ return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b); }
// https://stackoverflow.com/questions/5029920/how-to-use-2d-arrays-in-cuda
__global__ void KernelRunSim(int * devPtr, size_t pitch)
{
    int tidx = blockIdx.x*blockDim.x + threadIdx.x;
   int tidy = blockIdx.y*blockDim.y + threadIdx.y;
   
   *(devPtr + (tidx * tidy)) -= 1;
}

int main(void)
{
    int * world_grid_h = new int[DIM, DIM];
    int * out = nullptr;
    int * world_grid_d = nullptr;
    size_t pitch;

    for(int i = 0; i < DIM; i++)
    {
     for( int j = 0; j < DIM; j++)
     {
        world_grid_h[i,j] = j;
     }   
    }

    // http://www.orangeowlsolutions.com/archives/613
    // 2D pitched allocation and host->device memcopy
    cudaMallocPitch(&world_grid_d, &pitch, DIM * sizeof(int), DIM);
    cudaMemcpy2D(world_grid_d, pitch, world_grid_h, DIM*sizeof(int), DIM*sizeof(int), DIM, cudaMemcpyHostToDevice);

    dim3 gridSize(iDivUp(DIM, 1), iDivUp(DIM, 1));
    dim3 blockSize(1, 1);
    KernelRunSim<< <DIM, DIM >> >(world_grid_d, pitch);

    std::cout << cudaPeekAtLastError();
    cudaDeviceSynchronize();
    out = (int *)malloc(DIM*DIM*sizeof(int));
    cudaMemcpy2D(out, DIM * sizeof(int), world_grid_d, pitch, DIM * sizeof(int), DIM, cudaMemcpyDeviceToHost);

    for(int i = 0; i < DIM; i++)
    {
     for( int j = 0; j < DIM; j++)
     {
        std::cout << *(out + (i * j)) << "\n"; 
     }   
    }

    Animal * a = new Animal[ANIMAL_START_COUNT];

    return 0;
}