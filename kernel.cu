// Name:Joe Gonzalez
// Vector addition on the CPU, with timer and error checking
// To compile: nvcc A_VectorAddCPU.cu -o temp
/*
 What to do:
 1. Understand every line of the code and be able to explain it in class.
 2. I have intentionally broken the code in several places—find and fix them.
 3. Compile, run, and experiment with the code.
 4. Also, play with the Pointerstest.cu code in the supplementary code folder to understand how pointers work.
*/

/*
 Purpose:
 To fully understand how vector addition works on the CPU, so you can compare
 it to GPU-based vector addition in the next assignment.
*/

// Include files
#include <time.h> //changed to time.h for windows (sys/time.h used on linux?), clock_t, clock(), CLOCKS_PER_SEC
#include <stdlib.h> // Added to use malloc(), free()
#include <math.h> // needed for fabs, asb value floating
#include <stdio.h>

// Defines
#define N 1000 // Length of the vector N = 1000

// Global variables
float *A_CPU, *B_CPU, *C_CPU;
float Tolerance = 0.0001f; //float 

// Function prototypes, telling sys to be ready for these
void allocateMemory();
void innitialize();
void addVectorsCPU(float*, float*, float*, int);
bool check(float*, int, float);
long elaspedTime(clock_t start, clock_t end); //replaced timeval with clock_t since I removed <sys/time.h>
void cleanUp();

//Reserving the spots for arrays
void allocateMemory()
{
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N * sizeof(float));
	B_CPU = (float*)malloc(N * sizeof(float)); //added B_CPU, was never initialized 
	C_CPU = (float*)malloc(N * sizeof(float));


}

//Loading values into the vectors that we will add.
void innitialize()
{
	for (int i = 0; i < N; i++)
	{
		A_CPU[i] = (float)i;
		B_CPU[i] = (float)(2 * i);
	}
}

//Adding vectors a and b then stores result in vector c.
void addVectorsCPU(float* a, float* b, float* c, int n)
{
	for (int id = 0; id < n; id++)
	{
		c[id] = a[id] + b[id]; //changed from multiplication to addition
	}
}

// Checking to see if anything went wrong in the vector addition.
bool check(float* c, int n, float tolerance) //Tolerance mispelled
{
	int id;
	double myAnswer;
	double trueAnswer;
	double percentError;
	double m = n - 1; // Needed the -1 because we start at 0.

	myAnswer = 0.0;
	
	//Making sure the addition is right
	for (id = 0; id < n; id++)
	{
		myAnswer += c[id]; //summing all the elements to compare to true sum
	}
	//A[i]+B[i] = 3i ... 3*((N-1)N)/2
	trueAnswer = 3.0 * (m * (m + 1)) / 2.0;

	percentError = fabs((myAnswer - trueAnswer) / trueAnswer) * 100.0; //changed abs to fabs (abs only works for integers, fabs works for doubles)

	if (percentError <= tolerance) //changed tolerance to lowercase and changed < to <=
	{
		return(true);
	}
	else
	{
		return(false);
	}
}

// Calculating elasped time.
long elaspedTime(clock_t start, clock_t end) // replaced struct timeval with clock_t
{
	// tv_sec = number of seconds past the Unix epoch 01/01/1970
	// tv_usec = number of microseconds past the current second.

	//long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	//long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

	// Returning the total time elasped in microseconds
	//return (endTime - startTime);

	return (long)((double)(end - start) * 1000000.0 / CLOCKS_PER_SEC);
	// calc as double then long

}

//Cleaning up memory after we are finished.
void cleanUp() // needing to make the cleanUp lowercase to match definition
{
	// Freeing host "CPU" memory.
	free(A_CPU);
	free(B_CPU);
	free(C_CPU);



}

int main()
{
	//struct timeval start, end;// Removed so I could define without timeval

	// Allocating the memory you will need.
	allocateMemory();

	// Putting values in the vectors.
	innitialize();


	// ORIGINAL gettimeofday
	// Starting the timer.	
	//gettimeofday(&start, NULL);

	// Add the two vectors.
	//addVectorsCPU(A_CPU, B_CPU, C_CPU, N);

	// Stopping the timer.
	//gettimeofday(&end, NULL);

	//Replacing gettimeofday(start)....gettimeofday(end)
	clock_t start, end;
		start = clock();
		 addVectorsCPU(A_CPU, B_CPU, C_CPU, N);
		end = clock();
	for (int i = 0; i < 5; i++)
		printf("C[%d] = %f\n", i, C_CPU[i]);




	// Checking to see if all went correctly.
	if (check(C_CPU, N, Tolerance) == false)
	{
		printf("\n\n Something went wrong in the vector addition\n");
	}
	else
	{
		printf("\n\n You added the two vectors correctly on the CPU");
		printf("\n The time it took was %ld microseconds", elaspedTime(start, end));
	}

	// Your done so cleanup your room.	
	cleanUp(); //changed CleanUp to cleanUp

	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");

	
	return(0);

}
