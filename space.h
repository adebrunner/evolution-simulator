#ifndef SPACE_H
#define SPACE_H

#include <atomic>

using std::atomic;

class Space
{
public:
    Space(bool f = false, bool a = false, bool h = false)
    {
        setContainsFood(f);
        setContainsAnimal(a);
        setIsHome(h);
    }

    /// Copy constructor that supports moving atomic variables.
    /// Note, this should only be done by one thread as our memory order is relaxed.
    Space(const Space& s):
            contains_animal(s.contains_animal.load()),
            contains_food(s.contains_food.load())
    {
        this->is_home = s.is_home;
    }

    /// Needed to overload this due to atomic variables.
    Space operator=(const Space& s)
    {
        this->contains_animal.store(s.contains_animal.load());
        this->contains_food.store(s.contains_food.load());
        this->is_home = s.is_home;
        return this;
    }

    // TODO, perhaps we can remove these. Should just use setters below imo.
    void putFood() { this->contains_food = true; }
    bool putAnimal() { this->contains_animal = true; return true; }
    void makeHome() { this->is_home = true; }

    void setContainsFood(bool contains_food) { this->contains_food = contains_food; }
    void setContainsAnimal(bool contains_animal) {this->contains_animal = contains_animal; }
    void setIsHome(bool is_home) {this->is_home = is_home; }

    bool getContainsFood() { return this->contains_food; }
    bool getContainsAnimal() { return this->contains_animal; }
    bool getIsHome() { return this->is_home; }


private:
    atomic<bool> contains_food;
    atomic<bool> contains_animal;
    bool is_home;
};

#endif