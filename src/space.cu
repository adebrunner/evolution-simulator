#include "../include/space.cuh"

Space::Space(bool f, bool a, bool h)
{
    setContainsFood(f);
    setContainsAnimal(a);
    setIsHome(h);
}

Space::Space(const Space& s)
{
    this->is_home = s.is_home;
    this->contains_animal = s.contains_animal;
    this->contains_food = s.contains_food;
}

Space Space::operator=(const Space &s)
{
    this->contains_animal = s.contains_animal;
    this->contains_food = s.contains_food;
    this->is_home = s.is_home;
    return this;
}

__device__  bool Space::putAnimal(void)
{
    int val_to_store = 1;
    int val_already_stored = 0;

    val_already_stored = atomicExch(&(this->contains_animal), val_to_store);

    // This allows us to check if an animal was already stored. If val_ready_store == 0 (false)
    // we know we successfully placed an animal in this space. If val_already_store != false, an animal is
    // already in this space. putAnimal failed.
    return val_already_stored == false;
}
