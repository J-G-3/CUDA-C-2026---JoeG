// Name: Joe Gonzalez
// GPU random walk.
// nvcc P_GPURandomWalk.cu -o temp -lcurand

/*
 What to do:
 This code runs a random walk for 10,000 steps on the CPU.

 1. Use cuRAND to run 2,000 random walks of 10,000 steps simultaneously on the GPU, each with a different seed.
    Print the final positions of random walks 5, 100, 789, and 1622 as a spot check to get a warm
    fuzzy feeling that your code is producing different random walks for each thread.

 2. Use cudaMallocManaged(&variable, amount_of_memory_needed);
    This allocates unified memory, which is automatically managed between the CPU and GPU.
    You lose some control over placement, but it saves you from having to manually copy data
    to and from the GPU.
*/

/*
 Purpose:
 To learn how to use cuRAND and unified memory.
*/

/*
 Note:
 The maximum signed int value is 2,147,483,647, so the maximum unsigned int value is 4,294,967,295.

 RAND_MAX is guaranteed to be at least 32,767. When I checked it on my laptop (10/6/2025), it was 2,147,483,647.
 rand() returns a value in [0, RAND_MAX]. It actually generates a list of pseudo-random numbers that depends on the seed.
 This list eventually repeats (this is called its period). The period is usually 2³¹ = 2,147,483,648,
 but it may vary by implementation.

 Because RAND_MAX is odd on this machine and 0 is included, there is no exact middle integer.
 Casting to float as in (float)RAND_MAX / 2.0 divides the range evenly.
 Using integer division (RAND_MAX / 2) would bias results slightly toward the positive side by one value out of 2,147,483,647.

 I know this is splitting hares (sorry, rabbits), but I'm just trying to be as accurate as possible.
 You might do this faster with a clever integer approach, but I’m using floats here for clarity.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h> // cudaMalloc managed, etc. cuda funcs
#include <curand_kernel.h> // for random # generation on GPU

// Defines
#define NUMBER_OF_WALKS 2000 // our number of walks
#define THREADS_PER_BLOCK 256

// Globals
int NumberOfRandomSteps = 10000; // number of steps per walk
float MidPoint = (float)RAND_MAX/2.0f;

// Function prototypes
int getRandomDirection();
__global__ void GPURandomWalk(int *PositionX, int *PositionY, int NumberOfSteps, unsigned long long BaseSeed);
int main(int, char**);

int getRandomDirection()
{
	int randomNumber = rand();

	if(randomNumber < MidPoint) return(-1); // get a rand number, if its below midpoint go -1 otherwise go +1
	else return(1);
}

__global__ void GPURandomWalk(int *PositionX, int *PositionY, int NumberOfSteps, unsigned long long BaseSeed) // long long guarantees 64 bit for cuRAND which expects a 64 bit seed
{
// each thread is doing a walk

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < NUMBER_OF_WALKS) // until we finish 2000 walks
	{
		curandState State; // random number generator for the thread and has state(container for the random number generator)
		curand_init(BaseSeed + tid, 0, 0, &State); //curand(seed, sequence, offset, &state)
												   // sequence is a substream within the seed
												   // curand and will initialize and update State; state holds a fully initialized random generator

		int positionX = 0; // each thread starts at 0,0
		int positionY = 0;

		for(int i = 0; i < NumberOfSteps; i++) // loop for 10,000 steps 
		{
				if (curand(&State) & 1) // curand(&state) give me a random number using this threads random num generator
										// compare last bit to 1 and if true move right one
				{
    				positionX += 1;
				}
				else
				{
   				 	positionX -= 1;
				}

				if (curand(&State) & 1)
				{
    				positionY += 1;
				}
				else
				{
    				positionY -= 1;
				}	



		
		}

		PositionX[tid] = positionX;
		PositionY[tid] = positionY;
	}
}

int main(int argc, char** argv)
{
	srand(time(NULL));

	printf(" RAND_MAX for this implementation is = %d \n", RAND_MAX);

	// CPU version kept here for comparison
	int positionX = 0;
	int positionY = 0;
	for(int i = 0; i < NumberOfRandomSteps; i++)
	{
		positionX += getRandomDirection(); // coin toss based on whether random number is greater than the median
		positionY += getRandomDirection();
	}

	printf("\n CPU final position = (%d,%d) \n", positionX, positionY); // recording the final position of x & y for CPU

	// Unified memory
	int *PositionXGPU; // CPU read from and GPU write to arrays without memcpy 
	int *PositionYGPU;
	cudaMallocManaged(&PositionXGPU, NUMBER_OF_WALKS * sizeof(int)); // store allocated mem address at PositionXGPU with size of NOW*4bytes
	cudaMallocManaged(&PositionYGPU, NUMBER_OF_WALKS * sizeof(int)); // 

	int NumberOfBlocks = (NUMBER_OF_WALKS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; // calculates number of blocks we need
	unsigned long long BaseSeed = (unsigned long long)time(NULL); // uses current time to generate a random seed


    // calling kernal
	GPURandomWalk<<<NumberOfBlocks, THREADS_PER_BLOCK>>>(PositionXGPU, PositionYGPU, NumberOfRandomSteps, BaseSeed);
	cudaDeviceSynchronize();

	printf("\n Spot check GPU random walks:\n");
	printf(" Walk 5 final position    = (%d,%d)\n", PositionXGPU[5],    PositionYGPU[5]);
	printf(" Walk 100 final position  = (%d,%d)\n", PositionXGPU[100],  PositionYGPU[100]);
	printf(" Walk 789 final position  = (%d,%d)\n", PositionXGPU[789],  PositionYGPU[789]);
	printf(" Walk 1622 final position = (%d,%d)\n", PositionXGPU[1622], PositionYGPU[1622]);

	cudaFree(PositionXGPU);
	cudaFree(PositionYGPU);

	return 0;
}
