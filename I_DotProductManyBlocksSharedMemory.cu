// Name: Joe Gonzalez
// Vector Dot product on many block and useing shared memory
// nvcc I_DotProductManyBlocksSharedMemory.cu -o temp
/*
 What to do:
 This code computes the dot product of vectors smaller than the block size.

 Your tasks:
 - Extend the code to launch as many blocks as needed based on a fixed thread count and the vector length.
 - Use **shared memory** within each block to speed up the computation.
 - Pad the input with zeros to fill the last block, if necessary.
 - Perform the final reduction (summing partial results) on the **CPU**.
 - Set the thread count (block size) to 256.
 - Test your code by setting N to different values.
*/

/*
 Purpose:
 To understand that blocks do **not** synchronize with each other during a kernel call.
 In other words, you can't detect when **all blocks** are finished from inside the kernel.
 You can work around this by exiting the kernel, which ensures all blocks have completed.
 Also to learn how to use shared memory to speed up your code.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define N 1000 // Length of the vector

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
float DotCPU, DotGPU;
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void innitialize();
void dotProductCPU(float*, float*, int);
__global__ void dotProductGPU(float*, float*, float*, int);
bool  check(float, float, float);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();

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

// This will be the layout of the parallel space we will be using.
void setUpDevices()
{
	BlockSize.x = 256; // Very nice size as it is a multiple of 32 (warp) = 8 warps out of this
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x =  (N+BlockSize.x -1 ) / BlockSize.x;
	GridSize.y = 1;
	GridSize.z = 1;
}

// Allocating the memory we will be using.
void allocateMemory()
{	

	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	
	// Device "GPU" Memory
	cudaMalloc(&A_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,GridSize.x * sizeof(float)); // change to the number of blocks because we are returning "block" amount of sums 
	cudaErrorCheck(__FILE__, __LINE__);


}

// Loading values into the vectors that we will add.
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
	__shared__ float shared[256]; // create an array of 256 floats 
								  // this is for every block; every block has an array of 256 floats that every thread in that block can access
	int globalId = blockIdx.x * blockDim.x + threadIdx.x; // 256 * Block# + thread (just for indexing the whole vector using different blocks for different parts of the operation)
	int tid = threadIdx.x;

	// Multiply
	if (globalId < n) // if within the global index range we multiply; if not give it a zero
	{
		shared[tid] = a[globalId] * b[globalId]; // multiplying useful threads
	}
	else
	{
		shared[tid] = 0.0f; // so we pad using logic so that all unused threads are now set to zero
	}
	
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride /= 2) // stride = 256/2; for stride > 0; and after each run we set stride = stride/2
															   // the folding technique reduced to a for loop 
	{
		if (tid < stride) // for current thread index < stride 
		{
			shared[tid] += shared[tid + stride]; // updating the shared memory array; 256 divides nicely by 2, (block sizee)
												 // shared = shared[threadidx.x + stride]; jumping elements per stride

		}
	__syncthreads();

	}

	if (tid == 0) // for the first block (since it holds our sum)
	{
		c[blockIdx.x] = shared[0]; // updating the global memory array to hold our partial sums 
	}







	/*int id = threadIdx.x;
	
	c[id] = a[id] * b[id]; // perform the multiplication of all elements to there corresponding indexes c[i] = (a1*b1, a2*b2, a3*b3....)
	__syncthreads();       // sync all threads to stop until done multiplying
		
	int fold = blockDim.x; // block dim.x = 256
	while(1 < fold)        // while 1 < fold ==  (1<256)
	{
		if(fold%2 != 0)    // while dividing the fold/2 has a remainder that does not equal zero
		{
			if(id == 0 && (fold - 1) < n)  // if first element && fold-1 (255 because 0- 255) < n (some big number because we want to use as much blocks as possible)
			{
				c[0] = c[0] + c[fold - 1]; // the first element is equal to the last c[255] + c[0]; so we add and replace c[0] to not lose the last odd element (if it were an odd number of elements just explaining in terms of 256 because thats the current n)
			}
			fold = fold - 1; // 256-1 = 255, minus one to make it even because we just used the 255th element in the if statement and no longer need it
		}
		fold = fold/2; // fold now equals have of the total minus one or an already even number (if the "if" statment was not satisfied)so there is a clean division
		if(id < fold && (id + fold) < n) // if current index less than fold (255) && (current index + fold (255)) < n
		{
			c[id] = c[id] + c[id + fold]; // then add c[current index] + c[current index + halved fold (255/2)]; we now update the idx for the vector to make sure we are adding within bounds
		}
		__syncthreads(); // sync so that every thread on standby until they are all done adding 
	}
	*/
}

// Checking to see if anything went wrong in the vector addition.
bool check(float cpuAnswer, float gpuAnswer, float tolerence)						// do GPU and CPU results match ?
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
	
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

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
	//float localC_CPU, localC_GPU;
	
	// Setting up the GPU
	setUpDevices();
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	innitialize();
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	dotProductCPU(A_CPU, B_CPU, C_CPU, N);
	DotCPU = C_CPU[0];						// CPU dot product
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	/*if(BlockSize.x < N)
	{
		printf("\n\n Your vector size is larger than the block size.");
		printf("\n Because we are only using one block this will not work.");
		printf("\n Good Bye.\n\n");
		exit(0);
	}
	*/



	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	dotProductGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU, C_GPU, N);							// GPU dot product
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Adding the results of kernal, C_GPU adding all the elements of the partial sums
	// Copy mem from GPU to CPU
	cudaMemcpy(C_CPU, C_GPU, GridSize.x * sizeof(float), cudaMemcpyDeviceToHost); // same as we initialized not wasting memory, only returned block amount of floats
	cudaErrorCheck(__FILE__, __LINE__);

	// CPU addition
	DotGPU = 0.0f;
	for (int i = 0; i<GridSize.x; i++)
	{
		DotGPU += C_CPU[i]; // adding all partial sums on the CPU
	}
	
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