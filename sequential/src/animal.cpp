#include <vector>
#include <stdlib.h>
#include <time.h>
#include <iostream>

#include "../include/animal.h"

void Animal::move(World* world) {
	// Check surrounding squares to determine which are available
	std::vector<int> surrounding = { getLocation() + 1,
		getLocation() - 1, getLocation() - world->getHouseDim(), getLocation() + world->getHouseDim() };

	std::vector<int> available;
	for (int i = 0; i < surrounding.size(); i++) {
		// If the space is outside bounds or has an animal in it, we can't move there
		if (surrounding.at(i) >= 0 &&
			surrounding.at(i) < world->getHouseBoardSize()) {

			available.push_back(surrounding.at(i));
		}
	}

	int newLocation = this->pickNewLocation(world, available);

	// Update the world to reflect that an animal is now in this space
	while (!(*(world->getBoard() + newLocation)).putAnimal()) {
		for (int i = 0; i < available.size(); i++) {
			if (available.at(i) == newLocation) {
				available.erase(available.begin() + i);

				// If you can't move to any immediate spaces, decrease energy and stay put
				if (available.size() == 0) {
					return;
				}
			}
		}

		newLocation = this->pickNewLocation(world, available);
	}

	// Pick a open location and assign an animal to it
	(*(world->getBoard() + this->getLocation())).setContainsAnimal(false);
	this->setLocation(newLocation);	

	// If there was food in the space, the animal picks it up
	if ((*(world->getBoard() + this->getLocation())).getContainsFood()) {
		(*(world->getBoard() + this->getLocation())).setContainsFood(false);
		this->pickupFood();
	}
}

int Animal::pickNewLocation(World* world, std::vector<int> available) {
	for (int i = 0; i < available.size(); i++) {
		if ((*(world->getBoard() + available.at(i))).getContainsFood()) {
			// move to the first location found that contains food
			return available.at(i);
		}
	}

	// Pick a random available spot
	srand(time(NULL));
	return available.at(rand() % available.size());
}

void Animal::mutateAnimal(void)
{
    int thirty_percent = 0;
    int current_speed = this->getSpeed();
    int new_val = 0;

    srand(time(NULL));
    // Get thirty percent of speed
    thirty_percent = current_speed * 0.3;
    // Ensure it is at least 1
	if (thirty_percent == 0) thirty_percent = 1;

    // To allow for a negative change [-thirty percent, thirty percent],
    // we generate a value from 0 to 2*thirty percent then subtract thirty percent
    new_val = rand() % (thirty_percent * 2);
    new_val -= thirty_percent;
	if ((current_speed + new_val) < 1) {
		this->setSpeed(1);
	}
	else {
		this->setSpeed(current_speed + new_val);
	}

    // We inversely change energy by the amount speed changed * const factor.
	if ((this->getEnergy() - (new_val * SPEED_TO_ENERGY_FACTOR)) < 1) {
		this->setEnergy(1);
	}
	else {
		this->setEnergy(this->getEnergy() - (new_val * SPEED_TO_ENERGY_FACTOR));
	}
}

Animal Animal::produceOffspring(void)
{
    // Give parent's stats to child
    Animal new_animal(this->getSpeed(), this->getEnergy());
    // But add a mutation
    new_animal.mutateAnimal();
    return new_animal;
}

void Animal::printAnimal() {
	std::cout << "Animal in location " << this->getLocation() << ":" << std::endl;
	std::cout << "Energy: " << this->getEnergy() << ", Speed: " << this->getSpeed() << std::endl;
}