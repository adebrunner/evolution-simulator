#ifndef WORLD_H
#define WORLD_H

#include <stdlib.h>
#include <time.h>

#include "space.cuh"

// TODO: Create cpp file for this and update makefile
class World 
{
public:
    World(int f, int a, int d) 
    {
        setFood(f);
        setNumAnimals(a);
        setPlayableDim(d);
		setHouseDim(d + 2);

		// Initialize the board with open spaces. Allocate d+2 to have border for homes
        this->board = (Space*)malloc(sizeof(Space)*getHouseBoardSize());
        for (int i = 0; i < getHouseBoardSize(); i++) {
			Space space;

			// Make border spaces houses
			int row = i / getHouseDim();
			int col = i % getHouseDim();
			if (row == 0 || row == (getHouseDim() - 1) || col == 0 || col == (getHouseDim() - 1)) {
				space.makeHome();
			}

            *(this->board + i) = space;
        }
    }

    void setFood(int food) { this->food = food; }
    void setNumAnimals(int num_animals) { this->num_animals = num_animals; }
    void setPlayableDim(int dim) { this->playable_dim = dim; }
	void setHouseDim(int dim) { this->house_dim = dim; }
	__device__ __host__ void setBoard(Space * b){this->board = b;}

    __device__ __host__ Space * getBoard() { return this->board; }
	int getFood() { return this->food; }
	int getNumAnimals() { return this->num_animals; }
	int getPlayableDim() { return this->playable_dim; }
	int getHouseDim() { return this->house_dim; }
	int getPlayableBoardSize() { return this->playable_dim*this->playable_dim; }
	int getHouseBoardSize() { return this->house_dim*this->house_dim; }

    // Place food in random spaces on board
    void populateFood() {
        // Random seed
        srand(time(NULL));
        int numFood = 0;
        while(numFood < this->food) {
            int index = rand() % this->getHouseBoardSize();

			// Put food at any open space that is not a house
            if (!(*(this->board + index)).getIsHome() && (*(this->board + index)).getContainsFood() == false) {
                (*(this->board + index)).putFood();
                numFood += 1;
            }
        }
    }

private:
    int food; // Determines the amount of spaces that contain food
    int num_animals; // Determines the starting number of animals
	int house_dim; // Dim of world including houses
    int playable_dim; // Dim of world not including houses
    Space* board;

};

#endif