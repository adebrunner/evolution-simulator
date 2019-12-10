#ifndef SPACE_H
#define SPACE_H

#include <atomic>

using std::atomic;

class Space
{
public:
    Space(bool f = false, bool a = false, bool h = false);

    /// Copy constructor that supports moving atomic variables.
    /// Note, this should only be done by one thread as our memory order is relaxed.
    Space(const Space& s);

    /// Needed to overload this due to atomic variables.
    Space operator=(const Space& s);

    __device__ __host__ bool putAnimal(void);
    void putFood(void) { this->contains_food = true; }
    void makeHome(void) { this->is_home = true; }

    void setContainsFood(bool contains_food) { this->contains_food = contains_food; }
    void setContainsAnimal(bool contains_animal) {this->contains_animal = contains_animal; }
    void setIsHome(bool is_home) {this->is_home = is_home; }

    __device__ __host__ bool getContainsFood(void) { return this->contains_food; }
    __device__ __host__ bool getContainsAnimal(void) { return this->contains_animal; }
    __device__ __host__ bool getIsHome(void) { return this->is_home; }


private:
    atomic<bool> contains_food;
    atomic<bool> contains_animal;
    bool is_home;
};

#endif