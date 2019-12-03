#include <iostream>
#include <string>

#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "world.h"

__global__ void KernelRunSim(int * devPtr, size_t pitch)
{

}

int getNumberInput(string message);

int main() {
	cout << "Welcome to the Evolution Simulator. Please enter your parameters to begin the simulation..." << endl;
	int rounds = getNumberInput("Enter the number of rounds: ");
	int numAnimals = getNumberInput("Enter the number of animals: ");
	int dim = getNumberInput("Enter the dimension of the world board: ");
	int food = getNumberInput("Enter the number of spaces that have food: ");

	// Initialize world
	World* world = new World(food, numAnimals, dim);


	return 0;
}

int getNumberInput(string message) {
	cout << message;
	string input = "";
	getline(cin, input);
	return stoi(input);
}