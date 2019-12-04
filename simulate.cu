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

    thrust::device_vector<Animal> animals_d = animals_h;

	world->populateFood();

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
        else if(row == houseDim)
        {
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
