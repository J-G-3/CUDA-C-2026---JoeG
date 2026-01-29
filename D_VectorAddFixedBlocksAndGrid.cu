// Name: Joe Gonzalez
// Vector addition on the GPU of any size with fixed block and grid size also adding pragma unroll for speed up.
// nvcc D_VectorAddFixedBlocksAndGrid.cu -o temp
/*
 What to do:
 This code works well for adding vectors with fixed-size blocks. 
 Given the size of the vector it needs to add, it takes a set block size, determines how 
 many blocks are needed, and creates a grid large enough to complete the task. Cool, cool!
 
 But—and this is a big but—this can get you into trouble because there is a limited number 
 of blocks you can use. Though large, it is still finite. Therefore, we need to write the 
 code in such a way that we don't have to worry about this limit. Additionally, some block 
 and grid sizes work better than others, which we will explore when we look at the 
 streaming multiprocessors.
 
 Extend this code so that, given a block size and a grid size, it can handle any vector addition. 
 Start by hard-coding the block size to 256 and the grid size to 64. Then, experiment with different 
 block and grid sizes to see if you can achieve any speedup. Set the vector size to a very large value 
 for time testing.

 You’ve probably already noticed that the GPU doesn’t significantly outperform the CPU. This is because 
 we’re not asking the GPU to do much work, and the overhead of setting up the GPU eliminates much of the 
 potential speedup. 
 
 To address this, modify the computation so that:
 c = sqrt(cos(a)*cos(a) + a*a + sin(a)*sin(a) - 1.0) + sqrt(cos(b)*cos(b) + b*b + sin(b)*sin(b) - 1.0)
 Hopefully, this is just a convoluted and computationally expensive way to calculate a + b.
 If the compiler doesn't recognize the simplification and optimize away all the unnecessary work, 
 this should create enough computational workload for the GPU to outperform the CPU.

 Write the loop as a for loop rather than a while loop. This will allow you to also use #pragma unroll 
 to explore whether it provides any speedup. Make sure to include an if (id < n) condition in your code 
 to ensure safety. Finally, be prepared to discuss the impact of #pragma unroll and whether it helped 
 improve performance.
*/

/*
 Purpose:
 1. To learn how to stride through a vector of any size and add it on the GPU.
 2. To learn how to use #pragma unroll
*/

// Include files
#include <sys/time.h> // for time 
#include <stdio.h>
#include <math.h>

// Defines
#define N 11111503 // Length of the vector
#define UNROLL 4 // pragma constant, choose a small number like 2,4,8; too high of a number can result in too many registers being used; after runs 4 was the fastest time 
// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void innitialize();
void addVectorsCPU(float*, float*, float*, int);
__global__ void addVectorsGPU(float, float, float, int);
bool  check(float*, int);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();

// This check to see if an error happened in your CUDA code. It tell you what it thinks went wrong,
// and what file and line it occured on.
void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error; //error variable if equal to 0 then =cudaSuccess
	error = cudaGetLastError();

	if(error != cudaSuccess) //if error variable not= 0 then..
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line); //returning error, file, line#
	}
}

// This will be the layout of the parallel space we will be using.
void setUpDevices()
{
	BlockSize.x = 256; //THREAD COUNT
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = 64; //(N - 1)/BlockSize.x + 1; // This gives us the correct number of blocks. (CEILING DIVISION  )
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
	cudaMalloc(&C_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
}

// Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(2*i);
	}
}

// Adding vectors a and b on the CPU then stores result in vector c.
void addVectorsCPU(float *a, float *b, float *c, int n)
{
	for(int id = 0; id < n; id++)
	{ 
        
        c[id] = sqrt(cos(a[id])*cos(a[id]) + a[id]*a[id] + sin(a[id])*sin(a[id]) - 1.0) + sqrt(cos(b[id])*cos(b[id]) + b[id]*b[id] + sin(b[id])*sin(b[id]) - 1.0);
		// writing a more difficult way to use c[id] = a[id] + b[id]; in order to get the GPU to work harder
	}
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void addVectorsGPU(float *a, float *b, float *c, int n)
{
    // for the brain:
    // bloclidx = current block we on
    // blockDimx = 256 
    // threadidx = current thread we on
    // gridDimx = 64


	int tid = blockIdx.x*blockDim.x + threadIdx.x; // thread index to update vector element 
	int stride = blockDim.x * gridDim.x; // stride varible (the jumps that we are making) (jumping to block and thread location to update vector elements)
    // more on stride: each thread will process elements spaced 16384 apart (256*64)



// Using pragma Unroll gave us 15800-15900 us avg sample size: 11111503
// Using pragma Unroll 2 gave 15500-15600 us sample size: 11111503
    for (int base = tid; base < n; base += stride * UNROLL) //UNROLL IS A COMPILE TIME CONSTANT
    // stride*UNROLL says "hey we are running this chunk (4 elements) in our inner for loop, now skip these 4 elements because we don't want to overlap information"
    {
        // #pragma unroll   // compiler directive, says "when compiling the next loop, fully unroll it if possible", only unrolls the next for loop, place before a loop with small/defined bounds
           #pragma unroll 2 // partially unrolls, 2 copies of the body code per loop pass
        // #pragma unroll 4 // fully unroll same thing as the first but we are literally telling it to unroll as many times as k
        // #pragma unroll 1 // tells the compiler to not unroll "generate one copy of the body per iteration" 


        for(int k = 0; k < UNROLL; k++)
        {
            int id = base + k * stride;
            if(id < n)
            {
               c[id] = sqrt(cos(a[id])*cos(a[id]) + a[id]*a[id] + sin(a[id])*sin(a[id]) - 1.0) + sqrt(cos(b[id])*cos(b[id]) + b[id]*b[id] + sin(b[id])*sin(b[id]) - 1.0);
            }
        }      
    }
// pragma unroll higher time due to higher instruction count? 


// without using the unroll gives an average of 15200 us sample size: 11111503
   /* for(int base = tid; base < n; base += stride * 4) 
    {
      for(int k = 0; k < 4; k++)
        {
            int id = base + k * stride;
            if(id < n)
            {
               c[id] = sqrt(cos(a[id])*cos(a[id]) + a[id]*a[id] + sin(a[id])*sin(a[id]) - 1.0) + sqrt(cos(b[id])*cos(b[id]) + b[id]*b[id] + sin(b[id])*sin(b[id]) - 1.0);
            }
        }
    }
    */

    //previous if statement
	/*if(id < n) // Making sure we are not working on memory we do not own.
		// c[id] = sqrt(cos(a[id])*cos(a[id]) + a[id]*a[id] + sin(a[id])*sin(a[id]) - 1.0) + sqrt(cos(b[id])*cos(b[id]) + b[id]*b[id] + sin(b[id])*sin(b[id]) - 1.0);
        // c[id] = a[id] + b[id];
	//}*/

}

// Checking to see if anything went wrong in the vector addition.
bool check(float *c, int n, float tolerance) // Returning true or false value  based on addition accuracy
{
	int id;
	double myAnswer;
	double trueAnswer;
	double percentError;
	double m = n-1; // Needed the -1 because we start at 0.
	
	myAnswer = 0.0;
	for(id = 0; id < n; id++)
	{ 
		myAnswer += c[id];
	}
	
	trueAnswer = 3.0*(m*(m+1))/2.0;
	
	percentError = fabs((myAnswer - trueAnswer)/trueAnswer)*100.0; //changed to fabs for doubles 
	
	if(percentError < tolerance) //changed Tolerance to tolerance so that we use our function variable instead of the global one, would work either way but then why would we have tolerance
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
void cleanUp()
{
	// Freeing host "CPU" memory.
	free(A_CPU); 
	free(B_CPU); 
	free(C_CPU);
	
	// Free divice "GPU" memory. (adding error checks afater every GPU related line to see if there is freeing error)
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
	addVectorsCPU(A_CPU, B_CPU ,C_CPU, N);
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	// Zeroing out the C_CPU vector just to be safe because right now it has the correct answer in it.
	for(int id = 0; id < N; id++)
	{ 
		C_CPU[id] = 0.0;
	}
	
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	addVectorsGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU ,C_GPU, N);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Copy Memory from GPU to CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Making sure the GPU and CPU wiat until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	
	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(C_CPU, N, Tolerance) == false)
	{
		printf("\n\n Something went wrong in the GPU vector addition\n");
	}
	else
	{
		printf("\n\n You added the two vectors correctly on the GPU");
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
		printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
	}
	
	// Your done so cleanup your room.	
	cleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}