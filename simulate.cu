#include <iostream>

#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "animal.h"

int const DIM = 1024;
int const ANIMAL_START_COUNT = 100;

__global__ void KernelRunSim(int * devPtr, size_t pitch)
{

}

int main(void)
{
    World * world = new World(1, 1, 1);

    Animal * a = new Animal[ANIMAL_START_COUNT];

    return 0;
}