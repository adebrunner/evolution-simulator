#ifndef ANIMAL_H
#define ANIMAL_H

#include "world.h"
#include <vector>

// For every increase of 1 in speed, we decrease energy by 3 and vice versa.
int const SPEED_TO_ENERGY_FACTOR = 2;

class Animal
{
public:
    Animal(int sp=1, int e=5, int l=0)
    {
		setFood(0);
        setSpeed(sp);
        setEnergy(e);
		setCurrentEnergy(e);
        setLocation(l);
    }

    void setFood(int food) { this->food = food; }
    void setSpeed(int speed) { this->speed = speed; }
    void setEnergy(int energy) { this->energy = energy; }
	void setCurrentEnergy(int energy) { this->curr_energy = energy; }
    void setLocation(int l) { this->location = l; }
	void pickupFood() { this->food += 1; }
	void decreaseEnergy() { this->curr_energy = this->curr_energy - 1; }
	void resetEnergy() { this->curr_energy = this->energy; }

	int getFood() { return this->food; }
	int getSpeed() { return this->speed; }
	int getEnergy() { return this->energy; }
	int getCurrentEnergy() { return this->curr_energy; }
    int getLocation() { return this->location; }

    void move(World* world);
    Animal produceOffspring(void);

private:
	int pickNewLocation(World* world, std::vector<int> available);
	void mutateAnimal(void);

    int food;          // Keeps track of how many pieces of food an animal has
    int speed;         // Determines how many blocks an animal can move per unit of energy
    int energy;        // Determines how many moves an animal can make.
	int curr_energy;   // Reflects the current energy of the animal
    int location;      // Memory index of world of current animal position.
	bool hasFood;
};

#endif