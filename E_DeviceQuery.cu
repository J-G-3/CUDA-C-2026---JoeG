// Name: Joe Gonzalez
// Device query
// nvcc E_DeviceQuery.cu -o temp
/*
 What to do:
 This code prints out useful information about the GPU(s) in your machine, 
 but there is much more data available in the cudaDeviceProp structure.

 Extend this code so that it prints out all the information about the GPU(s) in your system. 
 Also, and this is the fun part, be prepared to explain what each piece of information means. 
*/

/*
 Purpose:
 To learn how to find out what is on the GPU(s) in your machine and if you even have a GPU.
*/

// Include files
#include <stdio.h>
#include <cuda_runtime.h>
// Defines

// Global variables

// Function prototypes
void cudaErrorCheck(const char*, int);

void cudaErrorCheck(const char *file, int line) //error check that we use 
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}








int main()
{
	cudaDeviceProp prop; //desclaring a struct that will hold cuda device properties 
    //structs are a group of variables, can hold more than one data type


	int count;
	cudaGetDeviceCount(&count); //count = integer itself, &count = address in memory where count lives
	cudaErrorCheck(__FILE__, __LINE__); //checking if error happened prev line..
	printf(" You have %d GPUs in this machine\n", count);
	
	for (int i=0; i < count; i++) // do this for as many devices there are in the system
    {
    	cudaGetDeviceProperties(&prop, i); //PROP IS A STRUCT we are looking at its location, i is the device index, selects the GPU with Index i
		cudaErrorCheck(__FILE__, __LINE__);
        //cudaGetDeviceProperties, returns a value for error check, assigns information starting at props memory location and then uses i to see which device its on, i = device index


    //------------------------------------------------------------GENERAL INFORMATION----------------------------------------------------------------------------------------------------------
    	printf(" \n\n---General Information for device %d ---\n", i);
		printf("Name: %s\n", prop.name); // prop = the struct, name = a field inside of the struct; so we call the name field 
		printf("Compute capability: %d.%d\n", prop.major, prop.minor); // major version of GPUs compute capability; minor version of GPUs compute capability, together they define what features the GPU supports
                                                                       // MAJOR = 7 MINOR = 5; COMPUTE CAPABILITY = 7.5
                                                                       // Compute capability is like the feature level for CUDA (what the GPU supports); threads per block, shared mem limits, etc.
        printf("Clock rate: %d\n", prop.clockRate); // represents the base clock freq; stored as Khz;  
                                                    // how fast each core can execute instructions, core clock freq
    
		printf("Device copy overlap: ");
	    if (prop.deviceOverlap) printf("Enabled\n"); // if device overlap is nonzero value then print enabled, if zero print Disabled
		else printf("Disabled\n");                   // whether device can do mem-copies from host - Device at the same time as running kernals

		printf("Kernel execution timeout : ");
        if (prop.kernelExecTimeoutEnabled) printf("Enabled\n"); // if GPU has a timer that will execute kernals if they take too long (watchdog timer)
		else printf("Disabled\n");

        //Added for compute mode
        // A CUDA context is an environment on GPU that holds everything that CUDA needs (workspace or sandbox)
        printf("Compute Mode:");
        if (prop.computeMode == cudaComputeModeDefault)  // multiple programs/processes can run CUDA kernels on the GPU simultaneously
            printf("Default (multiple contexts allowed) \n)");
        else if (prop.computeMode == cudaComputeModeExclusive) // prevents multiple processes from sharing the GPU
            printf("Exclusive (only one context at a time) \n)");
        else if (prop.computeMode == cudaComputeModeProhibited) // GPU exists but cannot be used for CUDA
            printf("Prohibited (no cuda allowed) \n)");
        else if (prop.computeMode == cudaComputeModeExclusiveProcess) // one process at a time but that process can run multiple contexts
            printf("Exclusive Process (one process at a time) \n)");
        else // uknown or unsupported value
             printf("Unknown \n)");

        //Integrated iGPU
        printf("integrated (iGPU): %s\n", prop.integrated ? "Yes" : "No"); // is the GPU an integrated GPU on the motherboard
        //ECC enabled? (error correcting code)
        printf("ECC enabled: %s\n", prop.ECCEnabled ? "Yes" : "No"); // detects and corrects single bit errors in GPU memory; prevents data corruption
                                                                     // LVL: single bit mem errors in the VRAM; good for long running or high precision workloads
                                                                     // without it code runs faster but errors go unfixed, faster mem access, no time spent checking errors, rare mem bit flips can corrupt results

        printf("PCI Domain/Bus/Device: %d / %d / %d\n", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID); // Peripheral Component Interconnect Express (PCIe); unique PCI address
                                                                                                            // useful for picking which GPU to run on

        printf("Concurrent kernels: %s\n", prop.concurrentKernels ? "Yes" : "No"); // yes: more than one kernal in different CUDA streams may run and overlap on the GPU at a time
                                                                                   // no: kernals run one at a time, wait for one to finish then execute

        printf("Async engine count (copy engines): %d\n", prop.asyncEngineCount); // Async engine count; engines that transfer data between CPU and GPU without blocking kernal execution; higher the count the more memory transfers that can happen
        printf("Can map host memory: %s\n", prop.canMapHostMemory ? "Yes" : "No"); // indicates whether the GPU can directly access the host memory
                                                                                   // Yes, need less cudaMemcpys; No, GPU cannot access host memory directly 

        printf("Unified addressing: %s\n", prop.unifiedAddressing ? "Yes" : "No"); // does the GPU support Unified virtual addressing (UVA)
                                                                                   // Yes: pointers returned from cudaMalloc or cudaHostMalloc can be passed across host and device
                                                                                   // No: host and device have seperate address spaces

    //-------------------------------------------------------------------------------MEMORY INFORMATION------------------------------------------------------------------------------------------
		printf(" \n\n---Memory Information for device %d ---\n", i); // mem dissection time!
		printf("Total global mem: %ld\n", prop.totalGlobalMem); // global mem (VRAM); the size of the GPU's VRAM in bytes
		printf("Total constant Mem: %ld\n", prop.totalConstMem); // constant mem (small but fast to access, part of GPU ram meant for unchanging variables that can be used across threads);total amount constant memory on GPU in bytes read only during kernal execution
		printf("Max mem pitch: %ld\n", prop.memPitch); // maximum pitch (width in bytes) allowed for 2D allocations in the GPU, number of bytes between the start of the first row to the first of the next
                                                       // movie theatre example: seat distance from 1st seat in the 1st row from the 1st seat in the second row
                                                
		printf("Texture Alignment: %ld\n", prop.textureAlignment); // specifies the allignment requirement in bytes for 1D/2D textures in GPU memory
                                                                   // so think of this as where you would start when allocating textures, each texture has a slot of bytes that it needs to be allocated to
                                                                   // lets say each slot is 512 bytes. Each texture must start at the 1st byte of the 512 in order to avoid misaligment. Misalignment results
                                                                   // in slower mem fetches as 2 more slots have to be used to refer to one texture or errors may occur even 
                                                                   // this 512 bytes is what is being returned: how big are our texture slots in bytes
        //More Memory Info
        printf("L2 cache size: %d\n", prop.l2CacheSize); // size of the L2 cache in bytes
                                                         // L2 Cache: small, fast mem layer between the GPUs main mem and the SMs
        printf("Memory clock rate (kHz): %d\n", prop.memoryClockRate); // clock speed of GPUs memory (global mem/VRAM) in Khz; how fast data can be read from or written to VRAM 
        printf("Memory bus width (bits): %d\n", prop.memoryBusWidth); // width of the memory bus in bits connecting the GPU to its VRAM in bits 
        printf("Texture pitch alignment: %ld\n", prop.texturePitchAlignment); // alignment requirement in bytes for the row pitch of 2D data
                                                                              // each row of a 2D texture in GPU memory must start at an address that is a multiple of # bytes (32 for example if 32 is returned as the value)
                                                                              // lets us know where to start and avoid misalignment 


        //MP info
		printf(" \n\n---MP Information for device %d ---\n", i); 
		printf("Multiprocessor count : %d\n", prop.multiProcessorCount); // Number of SMs on the GPU; SMs have a set of (Cores(ALUs)) that execute threads
		printf("Shared mem per block: %ld\n", prop.sharedMemPerBlock); // amount of shared memory per thread block (the block that contains those threads) in bytes
		printf("Registers per block: %d\n", prop.regsPerBlock); // total number of registers that each block can use for all threads in that block
		printf("Threads in warp: %d\n", prop.warpSize); // warp is a group of threads executed together, most modern have warp sizes of 32, 32 threads executed simultaneously in a single instruction step
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock); // max number of threads you can have in a block
		printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]); //max thread dimensions that you can set 
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]); // max grid dimensions that you can set
		printf("\n");
        //Added MP info
        printf("Shared mem per SM: %ld\n", prop.sharedMemPerMultiprocessor); // gives the max amount of shared mem a SM can use; shared mem is fast low latency mem within a block
        printf("Registers per SM: %d\n", prop.regsPerMultiprocessor); // max # of registers available per SM, 
        printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);// max threads per SM






        #if CUDART_VERSION >= 6000 // CUDART_VERSION being used (Cuda Runtime version check)
        printf("Managed memory supported: %s\n", prop.managedMemory ? "Yes" : "No"); // Yes: Cuda runtime can automatically move data between device and host as needed
        printf("Concurrent managed access: %s\n", prop.concurrentManagedAccess ? "Yes" : "No"); // yes: host and device can access managed memory without having to wait for the other to finish
        #endif

        #if CUDART_VERSION >= 5000 // if version greater than
        printf("Stream priorities supported: %s\n", prop.streamPrioritiesSupported ? "Yes" : "No"); // yes:assign different priorities to streams, fine grain control, streams will execute based on priority order that you can assign
        #endif                                                                                      // stream is a way to organize GPU tasks, a stream itself is a sequence of operations, like kernal executions or like memory copies

        #if CUDART_VERSION >= 9000
        printf("Cooperative launch: %s\n", prop.cooperativeLaunch ? "Yes" : "No"); // allows threads across multiple blocks or grids to coordinate their execution
        printf("Cooperative multi-device launch: %s\n", prop.cooperativeMultiDeviceLaunch ? "Yes" : "No"); // can have kernals running on multiple GPUs that synchronize and cooperate
        #endif


	}	
	return(0);
}