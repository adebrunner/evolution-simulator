#ifndef ANIMAL_H
#define ANIMAL_H

#include "world.h"
#include <vector>

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

    void setFood(int food) { this->food = food; }
    void setSpeed(int speed) { this->speed = speed; }
    void setEnergy(int energy) { this->energy = energy; }
    void setLocation(int l) { this->location = l; }
	void pickupFood() { this->food += 1; }

	int getFood() { return this->food; }
	int getSpeed() { return this->speed; }
	int getEnergy() { return this->energy; }
    int getLocation() { return this->location; }

    void move(World* world);

private:
	int pickNewLocation(World* world, std::vector<int> available);

    int food;          // Keeps track of how many pieces of food an animal has
    int speed;         // Determines how many blocks an animal can move per unit of energy
    int energy;        // Determines how many moves an animal can make. 
    int location;      // Memory index of world of current animal position.
	bool hasFood;
};

#endif