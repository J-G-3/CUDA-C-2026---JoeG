// Name: Joe Gonzalez
// Setting up a stream
// nvcc N_Streams.cu -o temp

/*
 What to do:
 Read about CUDA streams. Look at all the ???s in the code and remove the ???s that need to be removed so the code will run.
*/

/*
 Purpose:
 To learn how to setup and work with CUDA streams.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define DATA_CHUNKS (1024*1024) // 1 million elements per chunk used to split data set into smaller more manageable chunks 
                                // why do we need chunks?
                                // because data sets may be too large to fit in the GPU's mem all at once so we split it into smaller 
                                // chunks to be processed and then sent back to the CPU
#define ENTIRE_DATA_SET (20*DATA_CHUNKS) // dealing with 20 chunks; 20 million elements
#define MAX_RANDOM_NUMBER 1000 // max amount the rand can be 
#define BLOCK_SIZE 256 // pretty number 

//Globals
dim3 BlockSize; //This variable will hold the Dimensions of your block
dim3 GridSize; //This variable will hold the Dimensions of your grid
float *NumbersOnGPU, *PageableNumbersOnCPU, *PageLockedNumbersOnCPU;
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
cudaEvent_t StartEvent, StopEvent; // event definitions 
// ??? Notice that we have to define a stream
// cudaStream_t Stream0; // cudaStream_t a handle used to represent a stream in CUDA;  a way to manage and control asynchronous operations on the GPU


cudaStream_t streams[20]; // creating 20 stream variabls





// a stream is a sequence of operations that are executed on the GPU (one after the other);
// When you use streams, CUDA allows operations in different streams to execute concurrently







//Function prototypes
void cudaErrorCheck(const char*, int);
void setUpCudaDevices();
void allocateMemory();
void loadData();
void cleanUp();
__global__ void trigAdditionGPU(float *, float *, float *, int );

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

//This will be the layout of the parallel space we will be using.
void setUpCudaDevices()
{
	cudaEventCreate(&StartEvent); // events created to record time 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventCreate(&StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaDeviceProp prop; // cuda device prop used to hold all device data
	int whichDevice;
	
	cudaGetDevice(&whichDevice); // which device the address of the best device and its properties
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaGetDeviceProperties(&prop, whichDevice); // retrieve properties from whichdevice
	cudaErrorCheck(__FILE__, __LINE__);

    // device overlap means that the GPU supports overlap between memory transfers and kernal execution;
    // It can perform async mem operations while simultaneously running kernals
    // if device overlap == 0 then the CPU and GPU have ot wait for mem operations(like data transfer) to complete before the computation 

	if(prop.deviceOverlap != 1) 
	{
		printf("\n GPU will not handle overlaps so no speedup from streams");
		printf("\n Good bye.");
		exit(0);
	}
	
	// ??? Notice that we have to create the stream
	//cudaStreamCreate(&Stream0); // & passing the address of stream0  to modify its contents
	//cudaErrorCheck(__FILE__, __LINE__);
	
         for (int i = 0; i<20; ++i)
    {
    cudaStreamCreate(&streams[i]); // creating a stream for every defined stream variable (streams)

    }



	BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	if(DATA_CHUNKS%BLOCK_SIZE != 0)
	{
		printf("\n Data chunks do not divide evenly by block size, sooo this program will not work.");
		printf("\n Good bye.");
		exit(0);
	}
	GridSize.x = DATA_CHUNKS/BLOCK_SIZE;
	GridSize.y = 1;
	GridSize.z = 1;	
}

//Sets a side memory on the GPU and CPU for our use.
void allocateMemory()
{	
	//Allocate Device (GPU) Memory
	cudaMalloc(&A_GPU,ENTIRE_DATA_SET*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,ENTIRE_DATA_SET*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,ENTIRE_DATA_SET*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	//??? Notice that we are using host page locked memory
	//Allocate page locked Host (CPU) Memory (page lock for faster communication GPU and CPU)
	cudaHostAlloc(&A_CPU, ENTIRE_DATA_SET*sizeof(float), cudaHostAllocDefault);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaHostAlloc(&B_CPU, ENTIRE_DATA_SET*sizeof(float), cudaHostAllocDefault);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaHostAlloc(&C_CPU, ENTIRE_DATA_SET*sizeof(float), cudaHostAllocDefault);
	cudaErrorCheck(__FILE__, __LINE__);
}

void loadData()
{
	time_t t;
	srand((unsigned) time(&t)); // seeding rand with time 
	
	for(int i = 0; i < ENTIRE_DATA_SET; i++) // making a big data set
	{		
		A_CPU[i] = MAX_RANDOM_NUMBER*rand()/RAND_MAX; // random number from 0-1000 
		B_CPU[i] = MAX_RANDOM_NUMBER*rand()/RAND_MAX;	
	}
}

//Cleaning up memory after we are finished.
void cleanUp()
{
	cudaFree(A_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	
	// ??? Notice that we have to free this memory with cudaFreeHost
    // freeing mem on the host with cuda because it had pinned memory on it (pagelocked memory)
    // using pinned memory for faster communication between the CPU and GPU; bypassing the need for making CPU into GPU data
    // now the GPU can just directly read from the CPU thanks to shared mem
	cudaFreeHost(A_CPU);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFreeHost(B_CPU);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFreeHost(C_CPU);
	cudaErrorCheck(__FILE__, __LINE__);
	
    // killing the events to prevent mem leaks 
	cudaEventDestroy(StartEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventDestroy(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// ??? Notice that we have to kill the stream. 
    // kill stream == no mem leaks
	//cudaStreamDestroy(Stream0);
    for (int i = 0; i < 20; i++)
    {
        cudaStreamDestroy(streams[i]);
        cudaErrorCheck(__FILE__, __LINE__);

    }
	
}

__global__ void trigAdditionGPU(float *a, float *b, float *c, int n)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x; // global index
	
	if(id < n)
	{
		c[id] = sin(a[id]) + cos(b[id]);
	}
}

int main()
{
	float timeEvent;
	
	setUpCudaDevices(); //setup
	allocateMemory(); // allocate
	loadData(); // load random values
	
	cudaEventRecord(StartEvent, 0); // start event to start recording
	cudaErrorCheck(__FILE__, __LINE__);
	


    // this for loop ensures that we get through all of our data chunks
	/*for(int i = 0; i < ENTIRE_DATA_SET; i += DATA_CHUNKS)
	{
	cudaMemcpyAsync(A_GPU, &A_CPU[i], DATA_CHUNKS*sizeof(float), cudaMemcpyHostToDevice, Stream0);
    cudaMemcpyAsync(B_GPU, &B_CPU[i], DATA_CHUNKS*sizeof(float), cudaMemcpyHostToDevice, Stream0);

    trigAdditionGPU<<<GridSize, BlockSize, 0, Stream0>>>(A_GPU, B_GPU, C_GPU, DATA_CHUNKS); // look how we summon the kernal after starting an async memory transfer in the same stream
                                                                                            // if deviceOverlap == 0 the memory transfer and kernal execution could not happen concurrently; they would have to execute sequentially, but instead we can start the memory transfer and then have 
                                                                                            // the kernal execute in a concurrent stream to have both the data transferred and operations on the GPU execting

    cudaMemcpyAsync(&C_CPU[i], C_GPU, DATA_CHUNKS*sizeof(float), cudaMemcpyDeviceToHost, Stream0);
	}
	*/

    // looping to process all 20 chunks at once using the 20 streams 
    for (int i = 0; i<20; i++)
    {
        // have to use & with host data and but not with device data because B_GPU is already has the address of the starting point to that array
        // copy data from the CPU to GPU for chunk i; 
        // refresher: B_GPU is the pointer to starting adress of mem where we offset from start by adding chunks
        // &B_CPU an array that we offset by chunks to edit the right elements
        cudaMemcpyAsync(B_GPU + i * DATA_CHUNKS, &B_CPU[i * DATA_CHUNKS], DATA_CHUNKS * sizeof(float), cudaMemcpyHostToDevice, streams[i]);

        // SAME WITH acpu to aGPU
        cudaMemcpyAsync(A_GPU + i * DATA_CHUNKS, &A_CPU[i * DATA_CHUNKS], DATA_CHUNKS * sizeof(float), cudaMemcpyHostToDevice, streams[i]);


        // 0 before the streams represents shared mem size
        // identify the stream with streams (incremented i), A-C_GPU all starting addresses of arrays that are incremented by i chunks 
        trigAdditionGPU<<<GridSize, BLOCK_SIZE, 0, streams[i]>>> (A_GPU + i * DATA_CHUNKS, B_GPU + i*DATA_CHUNKS, C_GPU+i*DATA_CHUNKS, DATA_CHUNKS);

        // copy from GPU back to CPU
        cudaMemcpyAsync(&C_CPU[i * DATA_CHUNKS], C_GPU + i * DATA_CHUNKS, DATA_CHUNKS * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);


    }

    for (int i = 0; i < 20; i++)
    {
        cudaStreamSynchronize(streams[i]);
    }



	// ??? Notice that we have to make the CPU wait until the GPU has finished stream0
	//cudaStreamSynchronize(Stream0); 
	
	cudaEventRecord(StopEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);

	// Make the CPU wiat until this event finishes so the timing will be correct.
	cudaEventSynchronize(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent); // time event = elapsed time
	cudaErrorCheck(__FILE__, __LINE__);
	printf("\n Time on GPU = %3.1f milliseconds", timeEvent);
	
	
	printf("\n");
	//You're done so cleanup your mess.
	cleanUp();	
	
	return(0);
}
