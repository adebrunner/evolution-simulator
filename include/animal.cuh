#ifndef ANIMAL_H
#define ANIMAL_H

#include <thrust/device_vector.h>

#include "world.cuh"

// For every increase of 1 in speed, we decrease energy by 3 and vice versa.
int const SPEED_TO_ENERGY_FACTOR = 3;

class Animal
{
public:
    Animal(int sp=1, int e=1, int l=0)
    {
		setFood(0);
        setSpeed(sp);
        setEnergy(e);
        setLocation(l);
    }

    __device__ void setFood(int food) { this->food = food; }
    __device__ void setSpeed(int speed) { this->speed = speed; }
    __device__ void setEnergy(int energy) { this->energy = energy; }
    __device__ void setLocation(int l) { this->location = l; }
    __device__ void pickupFood() { this->food += 1; }

    __device__ int getFood() { return this->food; }
    __device__ int getSpeed() { return this->speed; }
    __device__ int getEnergy() { return this->energy; }
    __device__ int getLocation() { return this->location; }

    __device__ void move(World* world);
    __device__ Animal produceOffspring(void);

private:
    __device__ int pickNewLocation(World* world, int * available, int options_left);
    __device__ void mutateAnimal(void);

    int food;          // Keeps track of how many pieces of food an animal has
    int speed;         // Determines how many blocks an animal can move per unit of energy
    int energy;        // Determines how many moves an animal can make. 
    int location;      // Memory index of world of current animal position.
	bool hasFood;
};

#endif