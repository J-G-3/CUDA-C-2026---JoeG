// Name:Joe Gonzalez
// Vector addition on two GPUs.
// nvcc Q_VectorAdd2GPUS.cu -o temp

/*
 What to do:
 This code adds two vectors of any length on a GPU.
 Rewriting the Code to Run on Two GPUs:

 1. Check GPU Availability:
    Ensure that you have at least two GPUs available. If not, report the issue and exit the program.

 2. Handle Odd-Length Vector:
    If the vector length is odd, ensure that you select a half N value that does not exclude the last element of the vector.

 3. Send First Half to GPU 0:
    Send the first half of the vector to the first GPU, and perform the operation of adding a to b.

 4. Send Second Half to GPU 1:
    Send the second half of the vector to the second GPU, and again perform the operation of adding a to b.

 5. Return Results to the CPU:
    Once both GPUs have completed their computations, transfer the results back to the CPU and verify that the results are correct.

 6. Do NOT use "unified memory" I want you to copy the memory to each GPU so you can learn how to do it on a simple problem.
*/

/*
 Purpose:
 To learn how to use multiple GPUs.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Defines
#define N 11503 // Length of the vector

// Global variables
float *A_CPU, *B_CPU, *C_CPU; // CPU pointers

// GPU 0 pointers
float *A_GPU0, *B_GPU0, *C_GPU0;

// GPU 1 pointers
float *A_GPU1, *B_GPU1, *C_GPU1;

dim3 BlockSize; // This variable will hold the dimensions of your blocks
float Tolerance = 0.01f;

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory(int, int);
void innitialize();
void addVectorsCPU(float*, float*, float*, int);
__global__ void addVectorsGPU(float*, float*, float*, int);
bool check(float*, int, float);
long elaspedTime(struct timeval, struct timeval);
void CleanUp();

void cudaErrorCheck(const char *file, int line)
{
    cudaError_t error;
    error = cudaGetLastError();

    if(error != cudaSuccess)
    {
        printf("\nCUDA ERROR: message = %s, File = %s, Line = %d\n",
               cudaGetErrorString(error), file, line);
        exit(0);
    }
}

void setUpDevices()
{
    BlockSize.x = 256;
    BlockSize.y = 1;
    BlockSize.z = 1;
}

void allocateMemory(int N0, int N1)
{
    // Host memory
    A_CPU = (float*)malloc(N * sizeof(float));
    B_CPU = (float*)malloc(N * sizeof(float));
    C_CPU = (float*)malloc(N * sizeof(float));

    // GPU 0 memory
    cudaSetDevice(0);
    cudaMalloc((void**)&A_GPU0, N0 * sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc((void**)&B_GPU0, N0 * sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc((void**)&C_GPU0, N0 * sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);

    // GPU 1 memory
    cudaSetDevice(1);
    cudaMalloc((void**)&A_GPU1, N1 * sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc((void**)&B_GPU1, N1 * sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc((void**)&C_GPU1, N1 * sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);
}

void innitialize()
{
    for(int i = 0; i < N; i++)
    {
        A_CPU[i] = (float)i;
        B_CPU[i] = (float)(2 * i);
    }
}

void addVectorsCPU(float *a, float *b, float *c, int n)
{
    for(int id = 0; id < n; id++)
    {
        c[id] = a[id] + b[id];
    }
}

__global__ void addVectorsGPU(float *a, float *b, float *c, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < n)
    {
        c[id] = a[id] + b[id];
    }
}

bool check(float *c, int n, float tolerence)
{
    int id;
    double myAnswer;
    double trueAnswer;
    double percentError;
    double m = n - 1;

    myAnswer = 0.0;
    for(id = 0; id < n; id++)
    {
        myAnswer += c[id];
    }

    trueAnswer = 3.0 * (m * (m + 1)) / 2.0;

    percentError = fabs((myAnswer - trueAnswer) / trueAnswer) * 100.0;

    if(percentError < tolerence)
    {
        return true;
    }
    else
    {
        return false;
    }
}

long elaspedTime(struct timeval start, struct timeval end)
{
    long startTime = start.tv_sec * 1000000 + start.tv_usec;
    long endTime   = end.tv_sec * 1000000 + end.tv_usec;

    return endTime - startTime;
}

void CleanUp()
{
    free(A_CPU);
    free(B_CPU);
    free(C_CPU);

    cudaSetDevice(0);
    cudaFree(A_GPU0);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaFree(B_GPU0);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaFree(C_GPU0);
    cudaErrorCheck(__FILE__, __LINE__);

    cudaSetDevice(1);
    cudaFree(A_GPU1);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaFree(B_GPU1);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaFree(C_GPU1);
    cudaErrorCheck(__FILE__, __LINE__);
}

int main()
{
    timeval start, end;
    long timeCPU, timeGPU;

    int deviceCount;
    int halfN, N0, N1;
    dim3 GridSize0, GridSize1;

    cudaGetDeviceCount(&deviceCount);
    if(deviceCount < 2)
    {
        printf("\nNeed at least 2 GPUs to run this program.\n\n");
        return 0;
    }

    halfN = N / 2;
    N0 = halfN;
    N1 = N - halfN;   // Handles odd N automatically

    setUpDevices();
    allocateMemory(N0, N1);
    innitialize();

    // CPU addition
    gettimeofday(&start, NULL);
    addVectorsCPU(A_CPU, B_CPU, C_CPU, N);
    gettimeofday(&end, NULL);
    timeCPU = elaspedTime(start, end);

    // Zero out C_CPU before GPU run
    for(int id = 0; id < N; id++)
    {
        C_CPU[id] = 0.0f;
    }

    GridSize0.x = (N0 - 1) / BlockSize.x + 1;
    GridSize0.y = 1;
    GridSize0.z = 1;

    GridSize1.x = (N1 - 1) / BlockSize.x + 1;
    GridSize1.y = 1;
    GridSize1.z = 1;

    // GPU addition
    gettimeofday(&start, NULL);

    // GPU 0: first half
    cudaSetDevice(0);
    cudaMemcpy(A_GPU0, A_CPU, N0 * sizeof(float), cudaMemcpyHostToDevice);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMemcpy(B_GPU0, B_CPU, N0 * sizeof(float), cudaMemcpyHostToDevice);
    cudaErrorCheck(__FILE__, __LINE__);

    addVectorsGPU<<<GridSize0, BlockSize>>>(A_GPU0, B_GPU0, C_GPU0, N0);
    cudaErrorCheck(__FILE__, __LINE__);

    // GPU 1: second half
    cudaSetDevice(1);
    cudaMemcpy(A_GPU1, A_CPU + N0, N1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMemcpy(B_GPU1, B_CPU + N0, N1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaErrorCheck(__FILE__, __LINE__);

    addVectorsGPU<<<GridSize1, BlockSize>>>(A_GPU1, B_GPU1, C_GPU1, N1);
    cudaErrorCheck(__FILE__, __LINE__);

    // Copy results back from GPU 0
    cudaSetDevice(0);
    cudaMemcpy(C_CPU, C_GPU0, N0 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaErrorCheck(__FILE__, __LINE__);

    // Copy results back from GPU 1
    cudaSetDevice(1);
    cudaMemcpy(C_CPU + N0, C_GPU1, N1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaErrorCheck(__FILE__, __LINE__);

    // Synchronize both GPUs
    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaErrorCheck(__FILE__, __LINE__);

    cudaSetDevice(1);
    cudaDeviceSynchronize();
    cudaErrorCheck(__FILE__, __LINE__);

    gettimeofday(&end, NULL);
    timeGPU = elaspedTime(start, end);

    if(check(C_CPU, N, Tolerance) == false)
    {
        printf("\n\nSomething went wrong in the GPU vector addition\n");
    }
    else
    {
        printf("\n\nYou added the two vectors correctly on the GPU");
        printf("\nThe time it took on the CPU was %ld microseconds", timeCPU);
        printf("\nThe time it took on the GPU was %ld microseconds", timeGPU);
    }

    CleanUp();

    printf("\n\n");
    return 0;
}
