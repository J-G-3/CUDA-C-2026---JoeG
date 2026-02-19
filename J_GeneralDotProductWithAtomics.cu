// Name: Joe Gonzalez
// Robust Vector Dot product 
// nvcc J_GeneralDotProductWithAtomics.cu -o temp
/*
 What to do:
 This code computes the dot product of vectors of any length using shared memory to
 reduce the number of global memory accesses. However, since blocks can’t synchronize
 with each other, the final reduction must be handled on the CPU.

 To simplify the GPU-side logic, we’ll add some “pregame” setup and use atomic adds.

 1. Make sure the number of threads per block is a power of 2. This avoids messy edge
    cases during the reduction step. If it’s not a power of 2, print an error message
    and exit. (Without this, you'd have to check if the reduction is even or not,
    add the last element to the first, adjust the loop, etc.)

 2. Calculate the correct number of blocks needed to process the entire vector.
    Then check device properties to ensure the grid and block sizes are within hardware limits.
    Just because it works on your fancy GPU doesn’t mean it will work on your client’s older one.
    If the block or grid size exceeds the device’s capabilities, report the issue and exit gracefully.

 3. It’s inefficient to check inside your kernel if a thread is working past the end of the vector
    on every iteration. Instead, figure out how many extra elements are needed to fill out the grid,
    and pad the vector with zeros. Zero-padding doesn’t affect the dot product (0 * anything = 0).
    Use `cudaMemset` to explicitly zero out your device memory — don’t rely on "getting lucky"
    like you might have in previous assignments.

 4. In previous assignments, we had to do the final reduction on the CPU because we couldn't sync blocks.
    Now, use **atomic adds** to sum partial results directly on the GPU and avoid CPU post-processing.
    Then, copy the final result back to the CPU using `cudaMemcpy`.

    Note: Atomic operations on floats are only supported on GPUs with compute capability 3.0 or higher.
    Use device properties to check this before running the kernel.
    While you’re at it, if multiple GPUs are available, select the best one based on compute capability.

 5. Add any additional bells and whistles to make your code more robust and user-proof.
    Think of edge cases or bad input your client might provide and handle it cleanly.
*/

/*
 Purpose:
 To learn how to use atomic adds to avoid jumping out of the kernel for block synchronization.
 This is also your opportunity to make the code "foolproof" — handling edge cases gracefully.

 At this point, you should understand all the CUDA basics.
 From now on, we’ll focus on refining that knowledge and adding advanced features.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>   // NEW: needed for cudaDeviceProp, cudaGetDeviceCount, etc.

// Defines
#define N 1000000 // Length of the vector
#define BLOCK_SIZE 256 // Threads in a block; power of 2 and 8 warps 

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
float DotCPU, DotGPU;
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 1.01f;

// NEW: padded memory array to equal number of threads
int N_PADDED;

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void innitialize();
void dotProductCPU(float*, float*, float*, int);
__global__ void dotProductGPU(float*, float*, float*, int);
bool  check(float, float, float);
long elaspedTime(struct timeval, struct timeval);
void cleanUp(); // (prototype existed; original used CleanUp() function name)

// This check to see if an error happened in your CUDA code. It tell you what it thinks went wrong,
// and what file and line it occured on.
void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

// function to see if a number is a power of 2
int isPowerOf2(int x)
{
	if (x <= 0)
    {
        return 0;
    }
    while (x > 1){
        if (x%2 != 0)
        {
            return 0;
        }

        x/=2;

    }
    
    return 1; // once while loop broken with no remainders then is PWR of 2 

}

// This will be the layout of the parallel space we will be using.
void setUpDevices()
{
	// The "Pregame" setup
	int deviceCount = 0; // setting up an int for device count 
	cudaGetDeviceCount(&deviceCount);                                       // directly updating deviceCount "&" address location
	cudaErrorCheck(__FILE__, __LINE__);                                      // check that there are "devices", did we use an adress, 

	if(deviceCount <= 0) // if no devices in the system
	{
		printf("\n ERROR: No CUDA devices found.\n");
		exit(0);
	}

	printf("\n CUDA devices found: %d\n", deviceCount); // print number of devices

	int bestDevice = 0;                                                     // setting ann int for our device with the best compute Capability
	cudaDeviceProp bestProp;                                                // creating a struct to hold cuda device properties 
                                                                            // right now its just a struct with empty garbage values

	cudaGetDeviceProperties(&bestProp, 0);                                  // now we fill the previously defined struct with the properties of device zero


    // Using a for loop to list our devices and their specs (name, compute capability(minor.major))


	for(int d = 0; d < deviceCount; d++)  // d=0; for d+1 until = device count

	{
		cudaDeviceProp p; // another empty struct
		cudaGetDeviceProperties(&p, d); // fill in with device properties of the current d idx

		printf(" Device %d: %s, Compute Capability = %d.%d\n", d, p.name, p.major, p.minor); // accessing the defined p struct to print necessary items



        // okay now we previously defined "bestProp"; now we compare the struct of bestProp to the current p struct (current device loaded specs)
        // compare using compute capability values

		if(p.major > bestProp.major || (p.major == bestProp.major && p.minor > bestProp.minor))
		{
			bestDevice = d; // the bestDevice now equals current device index
			bestProp = p; // bestProp is loaded with the struct of the best device specs
		}
	}

	cudaSetDevice(bestDevice); // saying "hey this is the id of the best GPU now use it"
	cudaErrorCheck(__FILE__, __LINE__);

	printf("\n Selected device %d: %s, Compute Capability = %d.%d\n",bestDevice, bestProp.name, bestProp.major, bestProp.minor); // tell customer "we are using this device and heres the compute capability "

	// 1) Check if defined block size is a power of 2 and print error if not
	if(isPowerOf2(BLOCK_SIZE) == 0)
	{
		printf("\n ERROR: BLOCK_SIZE must be a power of 2. BLOCK_SIZE = %d\n", BLOCK_SIZE);
		exit(0);
	}

	// 4) Atomic add requires 3.0+ for compute capability; return error if that is not satisfied
	if(bestProp.major < 3)
	{
		printf("\n ERROR: atomicAdd(float) requires Compute Capability 3.0+\n");
		printf(" This GPU is %d.%d\n", bestProp.major, bestProp.minor);
		exit(0);
	}

// defining block and grid size

	BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;

	GridSize.x = (N - 1)/BlockSize.x + 1; // This gives us the correct number of blocks.
	GridSize.y = 1;
	GridSize.z = 1;

	// NEW: padded length so every launched thread has safe memory
	N_PADDED = (int)(GridSize.x * BlockSize.x); // total # of threads

	// 2) Checking grid and block size limits

    // using our bestProp struct to see if we defined more items than our GPU has
	if(BLOCK_SIZE > bestProp.maxThreadsPerBlock)
	{
		printf("\n ERROR: BLOCK_SIZE %d exceeds maxThreadsPerBlock %d on this GPU\n",
			   BLOCK_SIZE, bestProp.maxThreadsPerBlock);
		exit(0);
	}

	if((int)GridSize.x > bestProp.maxGridSize[0]) // [0] because x dimension
	{
		printf("\n ERROR: GridSize.x %u exceeds maxGridSize[0] %d on this GPU\n",
			   GridSize.x, bestProp.maxGridSize[0]);
		exit(0);
	}
/*
	size_t sharedNeeded = BLOCK_SIZE * sizeof(float); // getting our byte size of our shared array
	if(sharedNeeded > (size_t)bestProp.sharedMemPerBlock) //
	{
		printf("\n ERROR: Shared memory needed %zu exceeds sharedMemPerBlock %zu\n",
			   sharedNeeded, (size_t)bestProp.sharedMemPerBlock);
		exit(0);
	}
*/

// print Ns and sizes
	printf("\n Launch Setup:\n");
	printf(" BLOCK_SIZE  = %d threads\n", BLOCK_SIZE); 
	printf(" GridSize.x  = %u blocks\n", GridSize.x);
	printf(" N          = %d\n", N);    // user input vector
	printf(" N_PADDED   = %d\n\n", N_PADDED); // padded vector (zeros)
}


// Allocating the memory we will be using.
void allocateMemory()
{
	// Host "CPU" memory.
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));  // kept to resemble the former

	// Device "GPU" Memory
	// allocate padded input sizes so kernel reads are always safe
	cudaMalloc(&A_GPU, N_PADDED*sizeof(float)); // equal to all threads and not going into illegal memory with just n
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU, N_PADDED*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);

	// output is now ONE float accumulator for atomic adds
	cudaMalloc(&C_GPU, sizeof(float)); // now just equal to one float BECAUSE ATOMIC ADD
	cudaErrorCheck(__FILE__, __LINE__);
}

// Loading values into the vectors that we will doting.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{
		A_CPU[i] = (float)i;
		B_CPU[i] = (float)(3*i);
	}
}

// Adding vectors a and b on the CPU then stores result in vector c.
void dotProductCPU(float *a, float *b, float *C_CPU, int n)
{
	for(int id = 0; id < n; id++)
	{
		C_CPU[id] = a[id] * b[id];
	}

	for(int id = 1; id < n; id++)
	{
		C_CPU[0] += C_CPU[id];
	}
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void dotProductGPU(float *a, float *b, float *c, int n)
{
	int threadIndex = threadIdx.x;
	int vectorIndex = threadIdx.x + blockDim.x*blockIdx.x;

	__shared__ float c_sh[BLOCK_SIZE];

	c_sh[threadIndex] = a[vectorIndex] * b[vectorIndex];
	__syncthreads();

	int fold = blockDim.x;
	while(1 < fold)
	{
		fold = fold/2;

		if(threadIndex < fold)
		{
			c_sh[threadIndex] = c_sh[threadIndex] + c_sh[threadIndex + fold];
		}
		__syncthreads();
	}

	// atomic add is gonna add the first element of every block and store it in c(a one float defined solution variable)
	// atomicAdd(global_sum, block_sum
    if(threadIndex == 0)
	{
		atomicAdd(c, c_sh[0]);
	}
}

// Checking to see if anything went wrong in the vector addition.
bool check(float cpuAnswer, float gpuAnswer, float tolerence)
{
	double percentError;
	
	percentError = abs((gpuAnswer - cpuAnswer)/(cpuAnswer))*100.0;
	printf("\n\n percent error = %lf\n", percentError);
	
	if(percentError < Tolerance) 
	{
		return(true);
	}
	else 
	{
		return(false);
	}
}

// Calculating elasped time.
long elaspedTime(struct timeval start, struct timeval end)
{
	// tv_sec = number of seconds past the Unix epoch 01/01/1970
	// tv_usec = number of microseconds past the current second.

	long startTime = start.tv_sec * 1000000L + start.tv_usec; // In microseconds.
	long endTime = end.tv_sec * 1000000L + end.tv_usec; // In microseconds

	// Returning the total time elasped in microseconds
	return endTime - startTime;
}

// Cleaning up memory after we are finished.
void CleanUp()
{
	// Freeing host "CPU" memory.
	free(A_CPU);
	free(B_CPU);
	free(C_CPU);

	cudaFree(A_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
}

int main()
{
	timeval start, end;
	long timeCPU, timeGPU;

	// Setting up the GPU
	setUpDevices();

	// Allocating the memory you will need.
	allocateMemory();

	// Putting values in the vectors.
	innitialize();

	// Adding on the CPU
	gettimeofday(&start, NULL);
	dotProductCPU(A_CPU, B_CPU, C_CPU, N);
	DotCPU = C_CPU[0];
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);

	// Adding on the GPU
	gettimeofday(&start, NULL);

	
	// Zero-padding & zero accumulator (C answer variable) using cudaMemset

	cudaMemset(A_GPU, 0, N_PADDED*sizeof(float)); // mem region now 0s
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemset(B_GPU, 0, N_PADDED*sizeof(float)); // mem region now 0s
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemset(C_GPU, 0, sizeof(float));          // accumulator must start at 0
	cudaErrorCheck(__FILE__, __LINE__);

	// Copy Memory from CPU to GPU
	// (copy only real N values; rest stays zero)(already padded)
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);

	dotProductGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU, C_GPU, N);
	cudaErrorCheck(__FILE__, __LINE__);

	// Copy final result from GPU to CPU (ONE float now)
	cudaMemcpy(&DotGPU, C_GPU, sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);

	// Making sure the GPU and CPU wiat until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);

	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);

	// Checking to see if all went correctly.
	if(check(DotCPU, DotGPU, Tolerance) == false)
	{
		printf("\n\n Something went wrong in the GPU dot product.\n");
	}
	else
	{
		printf("\n\n You did a dot product correctly on the GPU");
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
		printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
	}

	// Your done so cleanup your room.
	CleanUp();

	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");

	return(0);
}



