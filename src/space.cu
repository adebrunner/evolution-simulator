#include "../include/space.cuh"

Space::Space(bool f, bool a, bool h)
{
    setContainsFood(f);
    setContainsAnimal(a);
    setIsHome(h);
}

Space::Space(const Space& s):
contains_animal(s.contains_animal.load()),
contains_food(s.contains_food.load())
{
    this->is_home = s.is_home;
}

Space Space::operator=(const Space &s)
{
    this->contains_animal.store(s.contains_animal.load());
    this->contains_food.store(s.contains_food.load());
    this->is_home = s.is_home;
    return this;
}

__device__ __host__ bool Space::putAnimal(void)
{
    bool val = false;

    // Checks to see if contains_animal == val. If they do match,
    // contains_animal is set to true. If they do not match (contains_animal must be true already)
    // val is set to true.
    this->contains_animal.compare_exchange_strong(val, true);

    // Val is true only when contains_animal was already true and
    // put animal failed.
    return val == false;
}
