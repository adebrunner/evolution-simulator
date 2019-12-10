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

__global__ void KernelRunSim(Animal * animals_vec_d, World * world_d)
{
    //cout << world_d->getBoard()->getIsHome() << endl;
    bool x = world_d->getBoard()->getIsHome();
    printf("%s\n", x ? "true" : "false");
    //printf(world_d->getBoard()->getIsHome() + "\n");
    //printf("\tCopying data\n");
    return;
}

// TODO: Make it so this doesn't need to be done.
__global__ void SetSpace(World * world_d, Space * b)
{
    world_d->setBoard(b);
    return;
}

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
    World * world_h = new World(food, num_animals, dim);

	thrust::host_vector<Animal> animals_h(num_animals);

	setAnimalStartingLocation(animals_h, world_h->getHouseDim());

	setWorldSpaceAnimalPresent(animals_h, world_h);

    thrust::device_vector<Animal> animals_d = animals_h;
    Animal * animals_pointer_d = thrust::raw_pointer_cast(animals_d.data());

	world_h->populateFood();

	// https://stackoverflow.com/questions/40682163/cuda-copy-inherited-class-object-to-device
    //Allocate storage for object onto GPU and copy host object to device
    World * world_d;
    cudaMalloc(&world_d,sizeof(World));
    cudaMemcpy(world_d,&world_h,sizeof(World),cudaMemcpyHostToDevice);

//Copy dynamically allocated Space objects to GPU
    Space ** d_par;
    int length = world_h->getHouseBoardSize();
    d_par = new Space*[length];
    for(int i = 0; i < length; ++i) {
        cudaMalloc(&d_par[i],sizeof(Space));
        cudaMemcpy(d_par[i],(world_h->getBoard() + i),sizeof(Space),cudaMemcpyHostToDevice);
    }

    // Not the best way, but we set the spaces on the device world.
    SetSpace<<<1,1>>>(world_d, *d_par);

    bool x = world_h->getBoard()->getIsHome();
    printf("%s\n", x ? "true" : "false");
	KernelRunSim<<<1,1>>>(animals_pointer_d, world_d);
    printf("\t3\n");
    cudaDeviceSynchronize();

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