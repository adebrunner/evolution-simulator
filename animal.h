class Animal
{
    public:
    Animal(int s=1, int sp=1, int e=1)
    {
        strength = s;
        speed = sp;
        e = e;
    }

    int strength; // Determines if an animal can eat another encountered anima;
    int speed;    // Determines how many blocks an animal can move per unit of energy
    int energy;   // Determines how many moves an animal can make. Should be inversely proportional
                  // to strength
};