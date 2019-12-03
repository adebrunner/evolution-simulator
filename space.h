#ifndef SPACE_H
#define SPACE_H

class Space 
{
public:
    Space(bool f = false, bool a = false, bool h = false) 
    {
        setContainsFood(f);
        setContainsAnimal(a);
        setIsHome(h);
    }

    void putFood() { this->contains_food = true; }
    void putAnimal() { this->contains_animal = true; }
    void makeHome() { this->is_home = true; }

    void setContainsFood(bool contains_food) { this->contains_food = contains_food; }
    void setContainsAnimal(bool contains_animal) {this->contains_animal = contains_animal; }
    void setIsHome(bool is_home) {this->is_home = is_home; }

    bool getContainsFood() { return this->contains_food; }
    bool getContainsAnimal() { return this->contains_animal; }
    bool getIsHome() { return this->is_home; }


private:
    bool contains_food;
    bool contains_animal;
    bool is_home;
};

#endif