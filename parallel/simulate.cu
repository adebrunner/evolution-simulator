#include <iostream>
#include <string>

#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>

using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::to_string;
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
        // Random seed
        srand(time(NULL));
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
int const SPEED_TO_ENERGY_FACTOR = 3;
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
        //printf("1\n");
        int options_left = 4;
        // Check surrounding squares to determine which are available
        int * available = new int(options_left);

        available[0] = getLocation() + 1;
        available[1] = getLocation() - 1;
        available[2] = getLocation() - world->getHouseDim();
        available[3] = getLocation() + world->getHouseDim();

        //printf("11\n");
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
        //printf("111\n");
//        for(int n = 0; n < options_left; n++ )
//        {
//            printf("%d\n", available[n]);
//        }
        int newLocation = this->pickNewLocation(world, available, options_left);
        //printf("%d 1.12\n", newLocation);
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
                return;
            }

            newLocation = this->pickNewLocation(world, available, options_left);
        }
        //printf("12\n");
        // Pick a open location and assign an animal to it
        this->setLocation(newLocation);

        // If there was food in the space, the animal picks it up
        if ((*(world->getBoard() + this->getLocation())).getContainsFood()) {
            (*(world->getBoard() + this->getLocation())).setContainsFood(false);
            this->pickupFood();
        }
    }
    __device__ Animal produceOffspring(void)
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

private:
    __device__ int pickNewLocation(World* world, int * available, int options_left) {
        //printf("2\n");
        //printf("3.0: %d\n", options_left);
        //printf("2.0\n");

        for (int i = 0; i < options_left; i++)
        {
            //printf("3.1\n");
            //printf("%d\n", available[i]);
            if ((*(world->getBoard() + available[i])).getContainsFood())
            {
                //printf("MAGIC\n");
                // move to the first location found that contains food
                return available[i];
            }
        }
        //printf("2.1\n");
        // https://stackoverflow.com/questions/12614164/generating-random-numbers-with-uniform-distribution-using-thrust
        // create a minstd_rand object to act as our source of randomness
        thrust::default_random_engine rng;
        // create a uniform_int_distribution to produce ints from [0,options_left]
        thrust::uniform_int_distribution<int> dist(0,options_left-1);
        rng.discard(clock64());
        // dist(rng) returns rand int in distribution
        int x = dist(rng);

       // printf("3: %d\n", x);
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
    __device__ void mutateAnimal(void)
    {
        int thirty_percent = 0;
        int current_speed = this->getSpeed();
        int new_val = 0;

        // https://stackoverflow.com/questions/12614164/generating-random-numbers-with-uniform-distribution-using-thrust
        // create a minstd_rand object to act as our source of randomness
        thrust::default_random_engine rng;

        // Get thirty percent of speed
        thirty_percent = current_speed * 0.3;
        // Ensure it is at least 1
        (thirty_percent == 0) ? 1 : thirty_percent;

        // To allow for a negative change [-thirty percent, thirty percent],
        // we generate a value from 0 to 2*thirty percent then subtract thirty percent
        // create a uniform_int_distribution to produce ints from [-thirty_percent,thirty_percent]
        thrust::uniform_int_distribution<int> dist(-thirty_percent,thirty_percent);
        // dist(rng) returns rand int in distribution
        rng.discard(clock64());
        new_val = dist(rng);
        new_val -= thirty_percent;

        this->setSpeed(current_speed + new_val);

        // We inversely change energy by the amount speed changed * const factor.
        this->setEnergy(this->getEnergy() - (thirty_percent * SPEED_TO_ENERGY_FACTOR));
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
//    ////cout << world_d->getBoard()->getIsHome() << endl;
//    //bool x = world_d->getBoard()->getIsHome();
//    ////printf("%s\n", x ? "true" : "false");
//    ////printf(world_d->getBoard()->getIsHome() + "\n");
//    ////printf("\tCopying data\n");
    for(int i = 0; i < animals_vec_d[index].getEnergy(); i++)
    {
        for(int j = 0; j < animals_vec_d[index].getSpeed(); j++)
        {
            //printf("%d premove loc %d\n", index, animals_vec_d[index].getLocation());
            (animals_vec_d[index]).move(world_d);
            //printf("%d postmove loc %d\n", index, animals_vec_d[index].getLocation());

            // if max food acquired, animal will stop moving.
            if(animals_vec_d[index].getFood() == MAX_FOOD)
            {
               // printf("%d won!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n", index);
                i = animals_vec_d[index].getEnergy();
                break;
            }
        }
//        if((world_d->getBoard()+i)->getContainsFood())
//        {
//            printf("%d\n", i);
//        }

        //world_d->getBoard()->putAnimal();
       // (world_d->getBoard() +  i)->getIsHome();
    }
    printf("%d food amount %d\n", index, animals_vec_d[index].getFood());



    return;
}

// TODO: Make it so this doesn't need to be done.
__global__ void SetSpace(World * world_d, Space * b)
{
    world_d->setBoard(b);
    return;
}

int test()
{
	cout << "Welcome to the Evolution Simulator. Please enter your parameters to begin the simulation..." << endl;
	int rounds = getNumberInput("Enter the number of rounds: ");
	int dim = getNumberInput("Enter the dimension of the world board: ");
	int num_animals = getNumberInput("Enter the number of animals: ");
    int start_energy = getNumberInput("Enter the starting energy of animals: ");
    int start_speed = getNumberInput("Enter the starting speed of animals: ");
	while (num_animals > (dim * 4 + 4)) {
		num_animals = getNumberInput("The number of animals must be less than " + to_string((dim * 4 + 4)) + ". Enter the number of animals: ");
	}
	int food = getNumberInput("Enter the number of spaces that have food: ");
	while (food > (dim*dim/4)) {
		food = getNumberInput("The number of spaces that have food must be less than " + to_string(dim*dim / 4) + ". Enter the number of spaces that have food: ");
	}

    // Initialize world
    World * world_h = new World(food, num_animals, dim);

	thrust::host_vector<Animal> animals_h(num_animals);

	for(int i = 0; i < num_animals; i++)
    {
	    animals_h[i].setSpeed(start_speed);
        animals_h[i].setEnergy(start_energy);
    }

	setAnimalStartingLocation(animals_h, world_h->getHouseDim());

	setWorldSpaceAnimalPresent(animals_h, world_h);

    thrust::device_vector<Animal> animals_d = animals_h;
    Animal * animals_pointer_d = thrust::raw_pointer_cast(animals_d.data());

	world_h->populateFood();

//	for(int i = 0; i < world_h->getHouseBoardSize(); i++)
//    {
//	    if((world_h->getBoard()+i)->getContainsFood())
//        {
//	        cout << i << endl;
//        }
//    }
    //cout <<"YAY\n";
	thrust::host_vector<World> w_h(1);
	w_h[0] = *world_h;
    //cout <<"Yw\n";
    //cout << w_h[0].getHouseDim() << "\n";
	thrust::device_vector<World> w_d = w_h;
    //cout <<"Yq\n";
	// https://stackoverflow.com/questions/40682163/cuda-copy-inherited-class-object-to-device
    //Allocate storage for object onto GPU and copy host object to device
    World * world_d = thrust::raw_pointer_cast(w_d.data());

    thrust::host_vector<Space> space_h(world_h->getHouseBoardSize());
    for(int i = 0; i < world_h->getHouseBoardSize(); i++)
    {
        space_h[i] = *(world_h->getBoard()+i);
    }

    thrust::device_vector<Space> space_d = space_h;
    //world_d->setBoard(thrust::raw_pointer_cast(space_d.data()));
//    cudaMalloc((void **)&world_d,sizeof(World));
//    cudaMemcpy(world_d,&world_h,sizeof(World),cudaMemcpyHostToDevice);
//    //cout <<"Yfff\n";
//Copy dynamically allocated Space objects to GPU
//    Space ** d_par;
//    int length = world_h->getHouseBoardSize();
//    d_par = new Space*[length];
//    for(int i = 0; i < length; ++i) {
//        cudaMalloc(&d_par[i],sizeof(Space));
//        cudaMemcpy(d_par[i],(world_h->getBoard() + i),sizeof(Space),cudaMemcpyHostToDevice);
//    }
//
//    // Not the best way, but we set the spaces on the device world.
    SetSpace<<<1,1 >>>(world_d, thrust::raw_pointer_cast(space_d.data()));
//    gpuErrchk( cudaPeekAtLastError() );
//    gpuErrchk( cudaDeviceSynchronize() );

    int grid = 1;
    if((num_animals / 512) > 1)
    {
        grid = num_animals / 512;
    }

    int * num_animals_d;
    cudaMalloc(&num_animals_d,sizeof(int *));
    cudaMemcpy(num_animals_d,&num_animals,sizeof(int *),cudaMemcpyHostToDevice);

    //cout << world_h->getHouseDim() <<" SEE\n";

    for(int i = 0; i < num_animals; i++)
    {
        cout << i << " " << animals_h[i].getFood() << endl;
    }

	KernelRunSim<<<grid,512>>>(animals_pointer_d, world_d, num_animals_d);
    //printf("\t312\n");
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    animals_h = animals_d;

    for(int i = 0; i < num_animals; i++)
    {
        cout << i << " " << animals_h[i].getFood() << endl;
    }

	return 0;
}



int main(void){return test();}