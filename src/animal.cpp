#include "../include/animal.h"
#include <vector>

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