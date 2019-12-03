#ifndef ANIMAL_H
#define ANIMAL_H

class Animal
{
public:
    Animal(int s=1, int sp=1, int e=1)
    {
        setStrength(s);
        setSpeed(sp);
        setEnergy(e);
    }

    void setStrength(int strength) { this->strength = strength; }
    void setSpeed(int speed) { this->speed = speed; }
    void setEnergy(int energy) { this->energy = energy; }

private:
    int strength; // Determines if an animal can eat another encountered anima;
    int speed;    // Determines how many blocks an animal can move per unit of energy
    int energy;   // Determines how many moves an animal can make. Should be inversely proportional
                  // to strength
};

#endif