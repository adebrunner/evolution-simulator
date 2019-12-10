#include <string>

#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "../include/world.h"
#include "../include/animal.h"

using std::string;

int getNumberInput(string message);

/// Assigns animals to available starting locations (home) sequentially.
void setAnimalStartingLocation(thrust::host_vector<Animal> & animals, int house_dim);

/// Given animals location, the world notes which spaces have an animal present
void setWorldSpaceAnimalPresent(thrust::host_vector<Animal> & animals, World * world);

/// Sets all ContainsAnimal spaces in world to false.
void clearWorldSpaceAnimalPresent(World * world);

int test();

__global__ void KernelRunSim(Animal * animals_vec_d, World * world_d);