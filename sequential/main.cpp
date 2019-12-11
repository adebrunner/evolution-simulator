#include <iostream>
#include <string>
#include <vector>
#include "include\animal.h"
#include "include\world.h"

using namespace std;

int getNumberInput(string message);

/// Assigns animals to available starting locations (home) sequentially.
void setAnimalStartingLocation(vector<Animal> & animals, int houseDim);

/// Given animals location, the world notes which spaces have an animal present
void setWorldSpaceAnimalPresent(vector<Animal> & animals, World * world);

/// Sets all ContainsAnimal spaces in world to false.
void clearWorldSpaceAnimalPresent(World * world);

void resetAnimalEnergyAndFood(vector<Animal> & animals);

void outputResults(vector<Animal> animals);

int main()
{
	cout << "Welcome to the Evolution Simulator. Please enter your parameters to begin the simulation..." << endl;
	int rounds = getNumberInput("Enter the number of rounds: ");
	int dim = getNumberInput("Enter the dimension of the world board: ");
	int num_animals = getNumberInput("Enter the number of animals: ");
	while (num_animals > (dim * 4 + 4)) {
		num_animals = getNumberInput("The number of animals must be less than " + to_string((dim * 4 + 4)) + ". Enter the number of animals: ");
	}
	int food = getNumberInput("Enter the number of spaces that have food: ");
	while (food > (dim*dim / 2)) {
		food = getNumberInput("The number of spaces that have food must be less than " + to_string(dim*dim / 4) + ". Enter the number of spaces that have food: ");
	}

	// Initialize world
	World * world = new World(food, num_animals, dim);

	// Put animals in the world
	vector<Animal> animals(num_animals);
	setAnimalStartingLocation(animals, world->getHouseDim());
	setWorldSpaceAnimalPresent(animals, world);

	// Populate the world with food
	world->populateFood();

	// Now we're ready to start the simulation...
	for (int i = 0; i < rounds; i++) { // for every requested round

		int num_active_animals = animals.size();

		while (num_active_animals > 0) {

			for (int a = 0; a < animals.size(); a++) { // for every animal
				// If the animal still has enough energy, allow it to take a turn
				if (animals.at(a).getCurrentEnergy() > 0) {
					// For an animal's turn, move it the # of spaces equal to its speed
					for (int s = 0; s < animals.at(a).getSpeed(); s++) {
						animals.at(a).move(world);

						// See if the animal has achieved its task for the round (getting 2 food)
						if (animals.at(a).getFood() == 2) {
							// Set current energy to zero so it stops moving and just chills until next round
							animals.at(a).setCurrentEnergy(0);
							s = animals.at(a).getSpeed();
							num_active_animals -= 1;
						}
					}

					animals.at(a).decreaseEnergy();
					if (animals.at(a).getCurrentEnergy() < 1 && animals.at(a).getFood() != 2) {
						num_active_animals -= 1;
					}
				}
			}
		}

		// See which animals survive/procreate/die after the turns are complete
		int numChecked = 0;
		int numToCheck = animals.size();
		int checkIndex = 0;
		while (numChecked < numToCheck) {
			if (animals.at(checkIndex).getFood() == 2) {
				// Procreate!
				animals.push_back(animals.at(checkIndex).produceOffspring());
				checkIndex++;
			}
			else if (animals.at(checkIndex).getFood() == 1) {
				// Nothing happens - the animal just stays in the list (lives on)
				checkIndex++;
			}
			else {
				// Kill the animal :(
				animals.erase(animals.begin() + checkIndex);
			}
			numChecked++;
		}

		// Reset animal starting locations
		setAnimalStartingLocation(animals, world->getHouseDim());
		setWorldSpaceAnimalPresent(animals, world);
		resetAnimalEnergyAndFood(animals);
		world->resetBoard(animals.size());
		world->populateFood();

		// Enter the next simulation round...
	}

	outputResults(animals);

	return 0;
}

void outputResults(vector<Animal> animals) {
	double averageSpeed = 0.0;
	double averageEnergy = 0.0;
	int maxEnergy = 0;
	int minEnergy = 10000000;
	int maxSpeed = 0;
	int minSpeed = 10000000;

	for (int i = 0; i < animals.size(); i++) {
		animals.at(i).printAnimal();
		cout << endl;

		averageEnergy += animals.at(i).getEnergy();
		averageSpeed += animals.at(i).getSpeed();

		if (animals.at(i).getEnergy() > maxEnergy) {
			maxEnergy = animals.at(i).getEnergy();
		}
		else if (animals.at(i).getEnergy() < minEnergy) {
			minEnergy = animals.at(i).getEnergy();
		}

		if (animals.at(i).getSpeed() > maxSpeed) {
			maxSpeed = animals.at(i).getSpeed();
		}
		else if (animals.at(i).getSpeed() < minSpeed) {
			minSpeed = animals.at(i).getSpeed();
		}
	}

	averageEnergy = averageEnergy / (double)animals.size();
	averageSpeed = averageSpeed / (double)animals.size();

	cout << endl;
	cout << "Final number of animals: " << animals.size() << endl;
	cout << "Average energy: " << averageEnergy << endl;
	cout << "Average speed: " << averageSpeed << endl;
	cout << "Maximum energy: " << maxEnergy << endl;
	cout << "Maximum speed: " << maxSpeed << endl;
}

int getNumberInput(string message)
{
	cout << message;
	string input = "";
	getline(cin, input);
	return stoi(input);
}

void resetAnimalEnergyAndFood(vector<Animal> & animals) {
	for (int i = 0; i < animals.size(); i++) {
		animals.at(i).setCurrentEnergy(animals.at(i).getEnergy());
		animals.at(i).setFood(0);
	}
}

void setAnimalStartingLocation(vector<Animal> & animals, int house_dim)
{
	int row = 0;
	int counter = 0;
	int temp_location = 0;
	for (int i = 0; i < animals.size(); i++)
	{
		if (i < house_dim)
		{
			animals[i].setLocation(i);
		}
		else if (i == house_dim)
		{
			animals[i].setLocation(i);
			row++;
			counter++;
		}
		else if (row == (house_dim - 1))
		{
			if (counter == house_dim)
			{
				cout << "Too many animals to put on board.\n";
				cout << "Only " << i << " animals placed.\n";
				break;
			}

			temp_location = (house_dim * row) + counter;
			animals[i].setLocation(temp_location);
			counter++;
		}
		else
		{
			temp_location = (house_dim * row) + (counter * (house_dim - 1));
			animals[i].setLocation(temp_location);
			if (counter == 1)
			{
				row++;
				counter--;
			}
			else
			{
				counter++;
			}
		}
	}
}

void setWorldSpaceAnimalPresent(vector<Animal> & animals, World * world)
{
	// Ensure everything is clear to begin
	clearWorldSpaceAnimalPresent(world);

	for (int i = 0; i < animals.size(); i++)
	{
		(world->getBoard() + animals[i].getLocation())->setContainsAnimal(true);
	}
}

void clearWorldSpaceAnimalPresent(World * world)
{
	for (int i = 0; i < world->getHouseDim(); i++)
	{
		(world->getBoard() + i)->setContainsAnimal(false);
	}
}