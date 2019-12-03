#include <iostream>

#include "cuda.h"

#include "animal.h"

#define DIM 1024
#define ANIMAL_START_COUNT 100

__global__ void KernelRunSim(void)
{

}

int main(void)
{
    int * world_grid = new int[DIM, DIM];
    for(int i = 0; i < DIM; i++)
    {
     for( int j = 0; j < DIM; j++)
     {
         world_grid[i,j] = j;
     }   
    }
    Animal * a = new Animal[ANIMAL_START_COUNT];

    return 0;
}