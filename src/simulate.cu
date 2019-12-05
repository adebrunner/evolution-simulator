#include <iostream>
#include <string>

#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "../include/world.h"
#include "../include/animal.h"
#include "../include/simulate.cuh"

using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::to_string;

//__global__ void KernelRunSim(int * devPtr, size_t pitch)
//{
//
//}
//
//int getNumberInput(string message);
//
///// Assigns animals to available starting locations (home) sequentially.
//void setAnimalStartingLocation(thrust::host_vector<Animal> & animals, int houseDim);
//
///// Given animals location, the world notes which spaces have an animal present
//void setWorldSpaceAnimalPresent(thrust::host_vector<Animal> & animals, World * world);
//
///// Sets all ContainsAnimal spaces in world to false.
//void clearWorldSpaceAnimalPresent(World * world);

int test()
{
	cout << "Welcome to the Evolution Simulator. Please enter your parameters to begin the simulation..." << endl;
	int rounds = getNumberInput("Enter the number of rounds: ");
	int dim = getNumberInput("Enter the dimension of the world board: ");
	int num_animals = getNumberInput("Enter the number of animals: ");
	while (num_animals > (dim * 4 + 4)) {
		num_animals = getNumberInput("The number of animals must be less than " + to_string((dim * 4 + 4)) + ". Enter the number of animals: ");
	}
	int food = getNumberInput("Enter the number of spaces that have food: ");
	while (food > (dim*dim/4)) {
		food = getNumberInput("The number of spaces that have food must be less than " + to_string(dim*dim / 4) + ". Enter the number of spaces that have food: ");
	}

    // Initialize world
    World * world = new World(food, num_animals, dim);

	thrust::host_vector<Animal> animals_h(num_animals);

	setAnimalStartingLocation(animals_h, world->getHouseDim());

	setWorldSpaceAnimalPresent(animals_h, world);

    thrust::device_vector<Animal> animals_d = animals_h;

	world->populateFood();
	// https://stackoverflow.com/questions/40682163/cuda-copy-inherited-class-object-to-device

	return 0;
}

int getNumberInput(string message)
{
	cout << message;
	string input = "";
	getline(cin, input);
	return stoi(input);
}

void setAnimalStartingLocation(thrust::host_vector<Animal> & animals, int house_dim)
{
    int row = 0;
    int counter = 0;
    int temp_location = 0;
    for(int i = 0; i < animals.size(); i++)
    {
        if (i < house_dim)
        {
            animals[i].setLocation(i);
        }
        else if(i == house_dim)
        {
            animals[i].setLocation(i);
            row++;
            counter++;
        }
        else if(row == (house_dim - 1))
        {
            if(counter == house_dim)
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