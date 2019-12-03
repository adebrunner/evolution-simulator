#ifndef WORLD_H
#define WORLD_H

#include "space.h"

class World 
{
public:
    World(int f, int a, int d) 
    {
        setFood(f);
        setNumAnimals(a);
        setDim(d);
        this->board = (Space*)malloc(sizeof(Space)*this->dim*this->dim);
        for (int i = 0; i < d*d; i++) {
			Space space;
            *(this->board + i) = space;
        }
    }

    void setFood(int food) { this->food = food; }
    void setNumAnimals(int num_animals) { this->num_animals = num_animals; }
    void setDim(int dim) { this->dim = dim; }

    Space* getBoard() { return this->board; }

private:
    int food; // Determines the amount of spaces that contain food
    int num_animals; // Determines the starting number of animals
    int dim; // Determines the dimension of the world; world will be dim*dim spaces
    Space* board;

};

#endif