#ifndef ANIMAL_H
#define ANIMAL_H

class Animal
{
public:
    Animal(int s=1, int sp=1, int e=1, int l = 0)
    {
        setStrength(s);
        setSpeed(sp);
        setEnergy(e);
        setLocation(l);
    }

    void setStrength(int strength) { this->strength = strength; }
    void setSpeed(int speed) { this->speed = speed; }
    void setEnergy(int energy) { this->energy = energy; }
    void setLocation(int l) { this->location = l; }

	int getStrength() { return this->strength; }
	int getSpeed() { return this->speed; }
	int getEnergy() { return this->energy; }
    int getLocation() { return this->location; }


private:
    int strength;      // Determines if an animal can eat another encountered animal;
    int speed;         // Determines how many blocks an animal can move per unit of energy
    int energy;        // Determines how many moves an animal can make. Should be inversely proportional
                       // to strength
    int location;      // Memory index of world of current animal position.
};

#endif