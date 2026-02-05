// Name:Joe Gonzalez
// Simple Julia CPU.
// nvcc F_JuliaCPUtoGPU.cu -o temp -lglut -lGL
// glut and GL are openGL libraries.
/*
 What to do:
 This code displays a simple Julia fractal using the CPU.
 Rewrite the code so that it uses the GPU to create the fractal.
 Keep the window at 1024 by 1024.
*/

/*
 Purpose:
 To apply your new GPU skills to do  something cool!
*/

// GL code, open graphics library

// Include files
#include <stdio.h>
#include <GL/glut.h>

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
#define A  -0.824	//Real part of C
#define B  -0.1711	//Imaginary part of C

// Global variables
unsigned int WindowWidth = 1024; // x
unsigned int WindowHeight = 1024; // y (resolution = 1024 x 1024 = 1,048,576 pixels)
dim3 BlockSize;
dim3 GridSize;

// the greater these mins and maxes are, the more you zoom out (actual output window)
float XMin = -2.0; // -2 to 2 
float XMax = 2.0;
float YMin = -2.0; // -2 to 2
float YMax = 2.0;

// Function prototypes
void cudaErrorCheck(const char*, int);
__device__ float escapeOrNotColor(float, float);

void setUpDevices()
{
	BlockSize.x = 16; // 16 by 16 because we dont want all the workers (32 by 32) because we may only be able to run one block at a time 
	BlockSize.y = 16; // also GPU likes to work in groups of 32 (warps), 16 x 16 = 256; and 256/32 = 8 warps
	// having a number that is not a multiple of 32 would cause partial warps and resources to be wasted on those warps
	// we want the best block use ratio 29/32 is much better than 10/32, the GPU works in groups of 32 so 16x16 is the sweet spot for this specific task
	BlockSize.z = 1;  // final thought: too big of blocks can cause stalling; think waiting for two massive orders they get a lot done but now we have to wait for those orders, 16x16 many orders and while 2 are cooking we can start the next, needs less resources does more 	

	GridSize.x = ((WindowWidth + BlockSize.x - 1) / BlockSize.x); // ceiling division to get grid size
	GridSize.y = ((WindowHeight + BlockSize.y - 1) / BlockSize.y); // 1024+16-1   /   16     = 64
	GridSize.z = 1;
}

// Just the error check
void cudaErrorCheck(const char* file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

__device__ float escapeOrNotColor(float x, float y) // added __device__ to be capatible with the kernal
													// only GPU can call it now but now float * can be used in GPU specific tasks
{
	float mag, tempX; // current magnitude and last x
	int count;

	int maxCount = MAXITERATIONS; // max amount of iterations (200)
	float maxMag = MAXMAG; // greater than this you left the set range (10)

	count = 0; // iteration count
	mag = sqrt(x * x + y * y);
	while (mag < maxMag && count < maxCount) // so while we are within iteration (200) and Magnitude (10) range we will execute the following calculations 
	{
		tempX = x; //We will be changing the x but we need its old value to find y.
		x = x * x - y * y + A; // x = x^2 + y^2 + A (real part)
		y = (2.0 * tempX * y) + B; // using the unupdated x value to update y (imaginary part)
		mag = sqrt(x * x + y * y); // recalculate distance from origin
		count++;
	}
	if (count < maxCount) // leaves bounds return black
	{
		return(0.0); // black
	}
	else // hasnt escaped return red
	{
		return(1.0); // red
	}
}


// Defining the Kernal

__global__ void julia_kernel(float* pixels, int width, int height, float xMin, float xMax, float yMin, float yMax)
{
	// blockIdx.x = 0-63 in this case(block we're on); blockDim.x = 16 (amount of threads in the x); threadIdx.x = 0-15
	// using block idx blockDim and threadIdx to define the position that we are executing the tasks in
	int x_idx = blockIdx.x * blockDim.x + threadIdx.x; // x global pixel index
	int y_idx = blockIdx.y * blockDim.y + threadIdx.y; // y global pixel index

	if (x_idx < width && y_idx < height) // dont use extra threads
	{
		// stepSize = min-max / # of pixels
		float stepSizeX = (xMax - xMin) / ((float)width); //   4/1024 = .0039
		float stepSizey = (yMax - yMin) / ((float)height);

		float x = xMin + x_idx * stepSizeX; // -2 + (0-1023)* 4/1024
		float y = yMin + y_idx * stepSizey;

		int k = (y_idx * width + x_idx) * 3;
		pixels[k] = escapeOrNotColor(x, y); // Red
		pixels[k + 1] = 0.0f;                // Green off
		pixels[k + 2] = 0.0f;                // Blue off

		// Summary x and y are the complex point for the pixel, escapeOrNotColor = what color should this pixel be 
		// pixel0: [R G B]
		// pixel1: [R G B]
		// ...
		// Pixels do not store any color its just an array of numbers that gets converted to color later on
	}
}





// Major Changes and Calling the kernal

void display(void)
{
	// float *pixels; // original only dealing with CPU 
	float* pixels_host;
	float* pixels_device;

	size_t size = WindowWidth * WindowHeight * 3 * sizeof(float); // size_t is a size dedicated variable data type; size of an objects in bytes variable type


	// Allocating Device and Host Memory
	pixels_host = (float*)malloc(size); // Pixels allocated for the host
	cudaMalloc((void**)&pixels_device, size); // memory allocation for the GPU
	cudaErrorCheck(__FILE__, __LINE__); // nothing went wrong with the allocation ?

	// Defining Grid and Block size
	setUpDevices();

	// Calling the Kernal
	julia_kernel << <GridSize, BlockSize >> > (pixels_device, WindowWidth, WindowHeight, XMin, XMax, YMin, YMax);
	cudaErrorCheck(__FILE__, __LINE__);

	// get to the same place (GPU and CPU)
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);

	// copying back to the CPU
	cudaMemcpy(pixels_host, pixels_device, size, cudaMemcpyDeviceToHost);

	// not using Async so that CPU can wait for the GPU and avoid potential errors (using synchronous)


	// Rendering
	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixels_host); // tells OpenGL draw a rectangle of pixels directly to the window using the data stored in pixels_host
																			// GL_RGB: each pixel has 3 color components: red green blue
																			// GL_FLOAT: each color component is a float
	glFlush(); // execute all drawing commands immediately 

	// cleanup
	free(pixels_host);
	cudaFree(pixels_device);
	cudaErrorCheck(__FILE__, __LINE__);




	/*-------------------------------------------- Previous Code Utilizing the CPU --------------------------------------




		float x, y, stepSizeX, stepSizeY;
		int k;

		//We need the 3 because each pixel has a red, green, and blue value.
		pixels = (float *)malloc(WindowWidth*WindowHeight*3*sizeof(float));
		//cudaMalloc


		stepSizeX = (XMax - XMin)/((float)WindowWidth);
		stepSizeY = (YMax - YMin)/((float)WindowHeight);

		k=0;
		y = YMin;
		while(y < YMax)
		{
			x = XMin;
			while(x < XMax)
			{
				pixels[k] = escapeOrNotColor(x,y);	//Red on or off returned from color
				pixels[k+1] = 0.0; 	//Green off
				pixels[k+2] = 0.0;	//Blue off
				k=k+3;			//Skip to next pixel (3 float jump)
				x += stepSizeX;
			}
			y += stepSizeY;
		}

		//Putting pixels on the screen.
		glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixels);
		glFlush();
	-------------------------------------------------------------------------------------------------------------------------*/
}
int main(int argc, char** argv) // argc and argv are command line arguments, in this code used for glut to communicate within itself to execute tasks properly 
{
	glutInit(&argc, argv); // initializes GLUT library, must be called before any GLUT function
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE); // GLUT_RGB is the color mode, GLUT_SINGLE = single buffering(drawing goes directly on screen)
	glutInitWindowSize(WindowWidth, WindowHeight); // sets initial size of the window 1024x1024
	glutCreateWindow("Fractals--Man--Fractals"); //creates the actual window on screen, the window title 
	glutDisplayFunc(display); // GLUT uses the display function, GLUT controls when rendering happens 
	glutMainLoop(); // never returns 
	return 0;
}
