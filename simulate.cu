#include <iostream>
#include <string>

#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "world.h"
#include "animal.h"

using std::cin;
using std::cout;
using std::endl;
using std::string;

__global__ void KernelRunSim(int * devPtr, size_t pitch)
{

}

int getNumberInput(string message);

/// Assigns animals to available starting locations (home) sequentially.
void setAnimalStartingLocation(thrust::host_vector<Animal> & animals, int houseDim);

/// Given animals location, the world notes which spaces have an animal present
void setWorldSpaceAnimalPresent(thrust::host_vector<Animal> & animals, World * world);

/// Sets all ContainsAnimal spaces in world to false.
void clearWorldSpaceAnimalPresent(World * world);

int main() {
	cout << "Welcome to the Evolution Simulator. Please enter your parameters to begin the simulation..." << endl;
	int rounds = getNumberInput("Enter the number of rounds: ");
	int numAnimals = getNumberInput("Enter the number of animals: ");
	int dim = getNumberInput("Enter the dimension of the world board: ");
	int food = getNumberInput("Enter the number of spaces that have food: ");

    // Initialize world
    World * world = new World(food, numAnimals, dim);

	thrust::host_vector<Animal> animals_h(numAnimals);

	setAnimalStartingLocation(animals_h, world->getHouseDim());

	setWorldSpaceAnimalPresent(animals_h, world);

    thrust::device_vector<Animal> animals_d = animals_h;

    // This function currently causes a hang on my pc
	//world->populateFood();

	// Debugging purposes - used to see where animals are located.
//	for(int i = 0; i < world->getHouseBoardSize(); i++)
//    {
//	    if((world->getBoard() + i)->getContainsAnimal())
//        {
//	        cout << i << endl;
//        }
//    }

	return 0;
}

int getNumberInput(string message)
{
	cout << message;
	string input = "";
	getline(cin, input);
	return stoi(input);
}

void setAnimalStartingLocation(thrust::host_vector<Animal> & animals, int houseDim)
{
    int row = 0;
    int counter = 0;
    int tempLocation = 0;
    for(int i = 0; i < animals.size(); i++)
    {
        if (i < houseDim)
        {
            animals[i].setLocation(i);
        }
        else if(i == houseDim)
        {
            animals[i].setLocation(i);
            row++;
            counter++;
        }
        else if(row == (houseDim - 1))
        {
            if(counter == houseDim)
            {
                cout << "Too many animals to put on board.\n";
                cout << "Only " << i << " animals placed.\n";
                break;
            }

            tempLocation = (houseDim * row) + counter;
            animals[i].setLocation(tempLocation);
            counter++;
        }
        else
        {
            tempLocation = (houseDim * row) + (counter * (houseDim - 1));
            animals[i].setLocation(tempLocation);
            if(counter == 1)
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

void setWorldSpaceAnimalPresent(thrust::host_vector<Animal> & animals, World * world)
{
    // Ensure everything is clear to begin
    clearWorldSpaceAnimalPresent(world);

    for(int i = 0; i < animals.size(); i++)
    {
        (world->getBoard() + animals[i].getLocation())->setContainsAnimal(true);
    }
}

void clearWorldSpaceAnimalPresent(World * world)
{
    for(int i = 0; i < world->getHouseDim(); i++)
    {
        (world->getBoard() + i)->setContainsAnimal(false);
    }
}