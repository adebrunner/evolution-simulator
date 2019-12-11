
#include <thrust/device_vector.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>

#include "../include/animal.cuh"

__device__ void Animal::move(World* world) {
    int options_left = 4;
	// Check surrounding squares to determine which are available
	int * available = new int(options_left);

	available[0] = getLocation() + 1;
    available[1] = getLocation() - 1;
    available[2] = getLocation() - world->getHouseDim();
    available[3] = getLocation() + world->getHouseDim();

	for (int i = 0; i < options_left; i++)
	{
		// If the space is outside bounds or has an animal in it, we can't move there
		if (available[i] < 0 &&
		available[i] > world->getHouseBoardSize())
		{
            removeIndex(i, available, options_left);
		}
	}

	int newLocation = this->pickNewLocation(world, available, options_left);

	// Update the world to reflect that an animal is now in this space
	while (!(*(world->getBoard() + newLocation)).putAnimal()) {
		for (int i = 0; i < options_left; i++)
		{
			if (available[i] == newLocation)
			{
                removeIndex(i, available, options_left);

				// If you can't move to any immediate spaces, decrease energy and stay put
				if (options_left == 0) {
					return;
				}
			}
		}

		newLocation = this->pickNewLocation(world, available, options_left);
	}

	// Pick a open location and assign an animal to it
	this->setLocation(newLocation);

	// If there was food in the space, the animal picks it up
	if ((*(world->getBoard() + this->getLocation())).getContainsFood()) {
		(*(world->getBoard() + this->getLocation())).setContainsFood(false);
		this->pickupFood();
	}
}

__device__ int Animal::pickNewLocation(World* world, int * available, int options_left) {
	for (int i = 0; i < options_left; i++) {
		if ((*(world->getBoard() + available[i])).getContainsFood()) {
			// move to the first location found that contains food
			return available[i];
		}
	}

	// https://thrust.github.io/doc/classthrust_1_1random_1_1uniform__int__distribution.html#details
    // create a minstd_rand object to act as our source of randomness
    thrust::minstd_rand rng;
    // create a uniform_int_distribution to produce ints from [0,options_left]
    thrust::uniform_int_distribution<int> dist(0,options_left);
    // dist(rng) returns rand int in distribution
	return available[dist(rng)];
}

__device__ void Animal::mutateAnimal(void)
{
    int thirty_percent = 0;
    int current_speed = this->getSpeed();
    int new_val = 0;

    // https://thrust.github.io/doc/classthrust_1_1random_1_1uniform__int__distribution.html#details
    // create a minstd_rand object to act as our source of randomness
    thrust::minstd_rand rng;

    // Get thirty percent of speed
    thirty_percent = current_speed * 0.3;
    // Ensure it is at least 1
    (thirty_percent == 0) ? 1 : thirty_percent;

    // To allow for a negative change [-thirty percent, thirty percent],
    // we generate a value from 0 to 2*thirty percent then subtract thirty percent
    // create a uniform_int_distribution to produce ints from [-thirty_percent,thirty_percent]
    thrust::uniform_int_distribution<int> dist(-thirty_percent,thirty_percent);
    // dist(rng) returns rand int in distribution
    new_val = dist(rng);
    new_val -= thirty_percent;

    this->setSpeed(current_speed + new_val);

    // We inversely change energy by the amount speed changed * const factor.
    this->setEnergy(this->getEnergy() - (thirty_percent * SPEED_TO_ENERGY_FACTOR));
}

__device__ Animal Animal::produceOffspring(void)
{
    // TODO: need to enforce this only happening when the animal has 2 food.
    // TODO: Need to figure out where to put animal as well.
    // Can do in simulation.

    // Give parent's stats to child
    Animal new_animal(this->getSpeed(), this->getEnergy());
    // But add a mutation
    new_animal.mutateAnimal();
    return new_animal;
}

__device__ void removeIndex(int index, int * arr, int & options_left)
{
    int it = 0;
    int * temp = nullptr;
    temp = new int(options_left);
    for(int j = 0; j < options_left; j++)
    {
        if (it == index)
        {
            it++;
        }
        else
        {
            temp[it] = arr[it];
            it++;
        }
    }
    options_left -= 1;
    delete [] arr;
    arr = temp;
    temp = nullptr;
}