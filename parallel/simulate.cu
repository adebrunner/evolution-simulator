#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <chrono>


#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>

using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::vector;
// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

class Space
{
public:
    __device__ __host__ Space(bool f=false, bool a=false, bool h=false)
    {
        setContainsFood(f);
        setContainsAnimal(a);
        setIsHome(h);
    }

    /// Copy constructor that supports moving atomic variables.
    /// Note, this should only be done by one thread as our memory order is relaxed.
    __device__ __host__ Space(const Space& s)
    {
        this->is_home = s.is_home;
        this->contains_animal = s.contains_animal;
        this->contains_food = s.contains_food;
    }

    /// Needed to overload this due to atomic variables.
    __device__ __host__  Space operator=(const Space &s)
    {
        this->contains_animal = s.contains_animal;
        this->contains_food = s.contains_food;
        this->is_home = s.is_home;
        return this;
    }

    __device__  bool putAnimal(void)
    {
        int val_to_store = 1;
        int val_already_stored = 0;

        val_already_stored = atomicExch(&(this->contains_animal), val_to_store);

        // This allows us to check if an animal was already stored. If val_ready_store == 0 (false)
        // we know we successfully placed an animal in this space. If val_already_store != false, an animal is
        // already in this space. putAnimal failed.
        return val_already_stored == false;
    }
    __device__ __host__ void putFood(void) { this->contains_food = true; }
    __device__ __host__ void makeHome(void) { this->is_home = true; }

    __device__ __host__ void setContainsFood(bool contains_food) { this->contains_food = contains_food; }
    __device__ __host__ void setContainsAnimal(bool contains_animal) {this->contains_animal = (contains_animal == true); }
    __device__ __host__ void setIsHome(bool is_home) {this->is_home = is_home; }

    __device__ __host__ bool getContainsFood(void) { return this->contains_food; }
    __device__ __host__ bool getContainsAnimal(void) { return this->contains_animal == 1; }
    __device__ __host__ bool getIsHome(void) { return this->is_home; }


private:
    bool contains_food;
    int contains_animal; // This is being set to an int as cuda does not have atomic exchange on boolean.
    bool is_home;
};

class World
{
public:
    World(int f=0, int a=0, int d=0)
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

    __device__ __host__ void setFood(int food) { this->food = food; }
    __device__ __host__ void setNumAnimals(int num_animals) { this->num_animals = num_animals; }
    __device__ __host__ void setPlayableDim(int dim) { this->playable_dim = dim; }
    __device__ __host__ void setHouseDim(int dim) { this->house_dim = dim; }
    __device__ __host__ void setBoard(Space * b){this->board = b;}

    __device__ __host__ Space * getBoard() { return this->board; }
    __device__ __host__ int getFood() { return this->food; }
    __device__ __host__ int getNumAnimals() { return this->num_animals; }
    __device__ __host__ int getPlayableDim() { return this->playable_dim; }
    __device__ __host__ int getHouseDim() { return this->house_dim; }
    __device__ __host__ int getPlayableBoardSize() { return this->playable_dim*this->playable_dim; }
    __device__ __host__ int getHouseBoardSize() { return this->house_dim*this->house_dim; }

    // Place food in random spaces on board
    __host__ void populateFood(void)
    {
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

// For every increase of 1 in speed, we decrease energy by 3 and vice versa.
int const SPEED_TO_ENERGY_FACTOR = 2;
// Max amount of food animal can acquire
int const MAX_FOOD = 2;

class Animal
{
public:
    __device__ __host__ Animal(int sp=1, int e=1, int l=0)
    {
        setFood(0);
        setSpeed(sp);
        setEnergy(e);
        setLocation(l);
    }

    __device__ __host__ void setFood(int food) { this->food = food; }
    __device__ __host__ void setSpeed(int speed) { this->speed = speed; }
    __device__ __host__ void setEnergy(int energy) { this->energy = energy; }
    __device__ __host__ void setLocation(int l) { this->location = l; }
    __device__ __host__ void pickupFood() { this->food += 1; }

    __device__ __host__ int getFood() { return this->food; }
    __device__ __host__ int getSpeed() { return this->speed; }
    __device__ __host__ int getEnergy() { return this->energy; }
    __device__ __host__ int getLocation() { return this->location; }

    __device__ void move(World* world) {
        int options_left = 4;
        // Check surrounding squares to determine which are available
        int * available = new int(options_left);
        available[0] = getLocation() + 1;
        available[1] = getLocation() - 1;
        available[2] = getLocation() - world->getHouseDim();
        available[3] = getLocation() + world->getHouseDim();

        for (int i = 0; i < options_left; i++)
        {
            // If the space is outside bounds or has an animal in it, we can't move there
            if (available[i] < 0 ||
                available[i] >= world->getHouseBoardSize())
            {
                removeIndex(i, available, options_left);
                // Need to reduce i by 1 so loops properly
                i--;
            }
        }

        int newLocation = this->pickNewLocation(world, available, options_left);
        // Update the world to reflect that an animal is now in this space

        while (!(*(world->getBoard() + newLocation)).putAnimal()) {
            for (int i = 0; i < options_left; i++)
            {
                if (available[i] == newLocation)
                {
                    removeIndex(i, available, options_left);
                    i--;
                    break;
                }
            }

            // If you can't move to any immediate spaces, decrease energy and stay put
            if (options_left == 0)
            {
                delete [] available;
                return;
            }

            newLocation = this->pickNewLocation(world, available, options_left);
        }
        // Pick a open location and assign an animal to it
        (*(world->getBoard() + this->getLocation())).setContainsAnimal(false);
        this->setLocation(newLocation);

        // If there was food in the space, the animal picks it up
        if ((*(world->getBoard() + this->getLocation())).getContainsFood()) {
            (*(world->getBoard() + this->getLocation())).setContainsFood(false);
            this->pickupFood();
        }
        delete [] available;
    }
    __host__ Animal produceOffspring(void)
    {
        // TODO: need to enforce this only happening when the animal has 2 food.
        // TODO: Need to figure out where to put animal as well.
        // Can do in simulation.

        // Give parent's stats to child
        Animal new_animal(this->getSpeed(), this->getEnergy());
        // But add a mutation
        new_animal.mutateAnimal();
        return new_animal;
    }

    __host__ void printAnimal(void)
    {
        cout << "Animal in location " << this->getLocation() << ":" << endl;
        cout << "Energy: " << this->getEnergy() << ", Speed: " << this->getSpeed() << endl;
    }

private:
    __device__ int pickNewLocation(World* world, int * available, int options_left) {

        for (int i = 0; i < options_left; i++)
        {
            if ((*(world->getBoard() + available[i])).getContainsFood())
            {
                // move to the first location found that contains food
                return available[i];
            }
        }
        // https://stackoverflow.com/questions/12614164/generating-random-numbers-with-uniform-distribution-using-thrust
        // create a default_random_engine object to act as our source of randomness
        thrust::default_random_engine rng;
        // create a uniform_int_distribution to produce ints from [0,options_left]
        thrust::uniform_int_distribution<int> dist(0,options_left-1);
        rng.discard(clock64());
        // dist(rng) returns rand int in distribution
        int x = dist(rng);

        return available[x];
    }
    __device__ void removeIndex(int index, int *& arr, int & options_left)
    {
        int it = 0;
        int * temp = nullptr;

        options_left -= 1;

        temp = new int(options_left);
        for(int j = 0; j < (options_left+1); j++)
        {
            if (j != index)
            {
                temp[it] = arr[j];
                it++;
            }
        }
        delete [] arr;
        arr = temp;
        temp = nullptr;
    }
    __host__ void mutateAnimal(void)
    {
        int thirty_percent = 0;
        int current_speed = this->getSpeed();
        int new_val = 0;


        // Get thirty percent of speed
        thirty_percent = current_speed * 0.3;
        // Ensure it is at least 1
        if (thirty_percent == 0) thirty_percent = 1;

        // To allow for a negative change [-thirty percent, thirty percent],
        // we generate a value from 0 to 2*thirty percent then subtract thirty percent
        new_val = rand() % ((thirty_percent * 2) + 1);
        new_val -= thirty_percent;
        if ((current_speed + new_val) < 1)
        {
            this->setSpeed(1);
        }
        else
            {
            this->setSpeed(current_speed + new_val);
        }

        // We inversely change energy by the amount speed changed * const factor.
        if ((this->getEnergy() - (new_val * SPEED_TO_ENERGY_FACTOR)) < 1)
        {
            this->setEnergy(1);
        }
        else
            {
            this->setEnergy(this->getEnergy() - (new_val * SPEED_TO_ENERGY_FACTOR));
        }
    }

    int food;          // Keeps track of how many pieces of food an animal has
    int speed;         // Determines how many blocks an animal can move per unit of energy
    int energy;        // Determines how many moves an animal can make.
    int location;      // Memory index of world of current animal position.
    bool hasFood;
};

int getNumberInput(string message)
{
    cout << message;
    string input = "";
    getline(cin, input);
    return stoi(input);
}

void setAnimalStartingLocation(thrust::host_vector<Animal> & animals, int house_dim)
{
    int row = 0;
    int counter = 0;
    int temp_location = 0;
    for(int i = 0; i < animals.size(); i++)
    {
        if (i < house_dim)
        {
            animals[i].setLocation(i);
        }
        else if(i == house_dim)
        {
            animals[i].setLocation(i);
            row++;
            counter++;
        }
        else if(row == (house_dim - 1))
        {
            if(counter == house_dim)
            {
                cout << "Too many animals to put on board.\n";
                cout << "Only " << i << " animals placed.\n";
                break;
            }

            temp_location = (house_dim * row) + counter;
            animals[i].setLocation(temp_location);
            counter++;
        }
        else
        {
            temp_location = (house_dim * row) + (counter * (house_dim - 1));
            animals[i].setLocation(temp_location);
            if(counter == 1)
            {
                row++;
                counter--;
            }
            else
            {
                counter++;
            }
        }
    }
}

void clearWorldSpaceAnimalPresent(World * world)
{
    for(int i = 0; i < world->getHouseDim(); i++)
    {
        (world->getBoard() + i)->setContainsAnimal(false);
    }
}

void setWorldSpaceAnimalPresent(thrust::host_vector<Animal> & animals, World * world)
{
    // Ensure everything is clear to begin
    clearWorldSpaceAnimalPresent(world);

    for(int i = 0; i < animals.size(); i++)
    {
        (world->getBoard() + animals[i].getLocation())->setContainsAnimal(true);
    }
}

__global__ void KernelRunSim(Animal * animals_vec_d, World * world_d, int * num_animals)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    // Only let thread run if there is an animal for it to control.
    if(index >= *num_animals)
    {
        return;
    }

    for(int i = 0; i < animals_vec_d[index].getEnergy(); i++)
    {
        for(int j = 0; j < animals_vec_d[index].getSpeed(); j++)
        {
            (animals_vec_d[index]).move(world_d);

            // if max food acquired, animal will stop moving.
            if(animals_vec_d[index].getFood() == MAX_FOOD)
            {
                i = animals_vec_d[index].getEnergy();
                break;
            }
        }
    }

    return;
}

// TODO: Make it so this doesn't need to be done.
__global__ void SetSpace(World * world_d, Space * b)
{
    world_d->setBoard(b);
    return;
}

void outputResults(vector<Animal> animals) {
    double averageSpeed = 0.0;
    double averageEnergy = 0.0;
    int maxEnergy = 0;
    int minEnergy = 10000000;
    int maxSpeed = 0;
    int minSpeed = 10000000;

    for (int i = 0; i < animals.size(); i++) {
        animals.at(i).printAnimal();
        cout << endl;

        averageEnergy += animals.at(i).getEnergy();
        averageSpeed += animals.at(i).getSpeed();

        if (animals.at(i).getEnergy() > maxEnergy) {
            maxEnergy = animals.at(i).getEnergy();
        }
        else if (animals.at(i).getEnergy() < minEnergy) {
            minEnergy = animals.at(i).getEnergy();
        }

        if (animals.at(i).getSpeed() > maxSpeed) {
            maxSpeed = animals.at(i).getSpeed();
        }
        else if (animals.at(i).getSpeed() < minSpeed) {
            minSpeed = animals.at(i).getSpeed();
        }
    }

    averageEnergy = averageEnergy / (double)animals.size();
    averageSpeed = averageSpeed / (double)animals.size();

    cout << endl;
    cout << "Final number of animals: " << animals.size() << endl;
    cout << "Average energy: " << averageEnergy << endl;
    cout << "Average speed: " << averageSpeed << endl;
    cout << "Maximum energy: " << maxEnergy << endl;
    cout << "Maximum speed: " << maxSpeed << endl;
}

int test()
{
	cout << "Welcome to the Evolution Simulator. Please enter your parameters to begin the simulation..." << endl;
	int rounds = getNumberInput("Enter the number of rounds: ");
	int dim = getNumberInput("Enter the dimension of the world board: ");
	int num_animals_h = getNumberInput("Enter the number of animals: ");
	while (num_animals_h > (dim * 4 + 4)) {
		num_animals_h = getNumberInput("The number of animals must be less than " + to_string((dim * 4 + 4)) + ". Enter the number of animals: ");
	}
    int start_energy = getNumberInput("Enter the starting energy of animals: ");
    int start_speed = getNumberInput("Enter the starting speed of animals: ");
	int food = getNumberInput("Enter the number of spaces that have food: ");
	while (food > (dim*dim*3/4)) {
		food = getNumberInput("The number of spaces that have food must be less than " + to_string(dim*dim / 2) + ". Enter the number of spaces that have food: ");
	}


    // Random seed
    srand(time(NULL));

	// Vars used later
	vector<Animal> temp_animal_vec;
    int * num_animals_d = nullptr;
    cudaMalloc(&num_animals_d,sizeof(int *));
    cudaMemcpy(num_animals_d,&num_animals_h,sizeof(int *),cudaMemcpyHostToDevice);
    Animal * animals_pointer_d = nullptr;
    thrust::host_vector<World> w_h(1);
    thrust::device_vector<World> w_d;
    World * world_h = nullptr;
    thrust::host_vector<Animal> animals_h(num_animals_h);
    // 1 time init
    for(int i = 0; i < num_animals_h; i++)
    {
        animals_h[i].setSpeed(start_speed);
        animals_h[i].setEnergy(start_energy);
    }
    thrust::device_vector<Animal> animals_d;
    World * world_d = nullptr;
    thrust::device_vector<Space> space_d;
    thrust::host_vector<Space> space_h;
    int grid = 1;
    // Initialize world
    world_h = new World(food, num_animals_h, dim);

    auto start = std::chrono::high_resolution_clock::now();
    for(int roundsDone = 0; roundsDone < rounds; roundsDone++)
    {
        setAnimalStartingLocation(animals_h, world_h->getHouseDim());

        setWorldSpaceAnimalPresent(animals_h, world_h);

        animals_d = animals_h;
        animals_pointer_d = thrust::raw_pointer_cast(animals_d.data());

        world_h->populateFood();

        w_h[0] = *world_h;

        w_d = w_h;

        // https://stackoverflow.com/questions/40682163/cuda-copy-inherited-class-object-to-device
        //Allocate storage for object onto GPU and copy host object to device
        world_d = thrust::raw_pointer_cast(w_d.data());

        space_h = thrust::host_vector<Space>(world_h->getHouseBoardSize());
        for(int i = 0; i < world_h->getHouseBoardSize(); i++)
        {
            space_h[i] = *(world_h->getBoard()+i);
        }

        space_d = space_h;

        // Not the best way, but we set the spaces on the device world.
        SetSpace<<<1,1 >>>(world_d, thrust::raw_pointer_cast(space_d.data()));
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        if((num_animals_h / 512) > 1)
        {
            grid = num_animals_h / 512;
        }

        KernelRunSim<<<grid,512>>>(animals_pointer_d, world_d, num_animals_d);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        animals_h = animals_d;

        temp_animal_vec.clear();
        for(int i = 0; i < num_animals_h; i++)
        {
            if(animals_h[i].getFood() == 1)
            {
                animals_h[i].setFood(0);
                temp_animal_vec.push_back(animals_h[i]);
            }
            else if(animals_h[i].getFood() == 2)
            {
                animals_h[i].setFood(0);
                temp_animal_vec.push_back(animals_h[i]);
                // Push back animals mutated offspring if it got 2 food
                temp_animal_vec.push_back(animals_h[i].produceOffspring());
            }
        }

        // Change number of animals to those that survived and new ones.
        num_animals_h = temp_animal_vec.size();
        // If all animals failed/died, exit sim
        if(num_animals_h == 0)
        {
            break;
        }
        cudaMemcpy(num_animals_d,&num_animals_h,sizeof(int *),cudaMemcpyHostToDevice);


        animals_h = thrust::host_vector<Animal>(num_animals_h);
        for(int i = 0; i < num_animals_h; i++)
        {
            animals_h[i] = temp_animal_vec[i];
        }

        for(int i = 0; i < world_h->getHouseBoardSize(); i++)
        {
            (world_h->getBoard() + i)->setContainsAnimal(false);
            (world_h->getBoard() + i)->setIsHome(false);
            (world_h->getBoard() + i)->setContainsFood(false);
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    outputResults(temp_animal_vec);

    cout << "Time to initialize and run simulation: " << duration.count() << " microseconds";

	return 0;
}

int main(void){return test();}