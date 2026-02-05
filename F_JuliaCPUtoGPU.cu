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
float XMax =  2.0;
float YMin = -2.0; // -2 to 2
float YMax =  2.0;

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
	
	GridSize.x = ((WindowWidth + BlockSize.x - 1) / BlockSize.x); // ceiling division
	GridSize.y = ((WindowHeight + BlockSize.y -1) / BlockSize.y);
	GridSize.z = 1;
}

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

__device__ float escapeOrNotColor (float x, float y) // added __device__ to be capatible with the kernal
                                                     // only GPU can call it now but now float * can be used in GPU specific tasks
{
	float mag,tempX;
	int count;
	
	int maxCount = MAXITERATIONS; // max amount of iterations
	float maxMag = MAXMAG; // greater than this you left the set range (10)
	
	count = 0;
	mag = sqrt(x*x + y*y);
	while (mag < maxMag && count < maxCount) // so while we are within iteration (200) and Magnitude (10) range we will execute the following calculations 
	{	
		tempX = x; //We will be changing the x but we need its old value to find y.
		x = x*x - y*y + A; // x = x^2 + y^2 + A
		y = (2.0 * tempX * y) + B; // using the unupdated x value to update y
		mag = sqrt(x*x + y*y);
		count++;
	}
	if(count < maxCount) // leaves bounds within max iterations then return 0
	{
		return(0.0);
	}
	else // greater than max iterations then its staying home
	{
		return(1.0);
	}
}


// Defining the Kernal

__global__ void julia_kernel(float *pixels, int width, int height, float xMin, float xMax, float yMin, float yMax )
{

// using block idx blockDim and threadIdx to define the position that we are executing the tasks in
int x_idx = blockIdx.x * blockDim.x + threadIdx.x; // x index
int y_idx = blockIdx.y * blockDim.y + threadIdx.y; // y index

if (x_idx < width && y_idx < height)
{
    float stepSizeX = (xMax - xMin)/((float)width); 
    float stepSizey = (yMax - yMin)/((float)height);

    float x = xMin + x_idx * stepSizeX;
    float y = yMin + y_idx * stepSizey;

    int k = (y_idx * width + x_idx) *3;
    pixels[k] = escapeOrNotColor(x,y); // Red
    pixels[k+1] = 0.0f;                // Green off
    pixels[k+2] = 0.0f;                // Blue off


}
}





// Major Changes and Calling the kernal

void display(void) 
{ 
	// float *pixels; // original only dealing with CPU 
    float *pixels_host;
    float *pixels_device;

    size_t size = WindowWidth * WindowHeight * 3 * sizeof(float); // size_t is a size dedicated variable data type; size of an objects in bytes variable type


    // Allocating Device and Host Memory
    pixels_host = (float *)malloc(size); // Pixels allocated for the host
    cudaMalloc((void**)&pixels_device, size); // memory allocation for the GPU
    cudaErrorCheck(__FILE__, __LINE__); // nothing went wrong with the allocation ?

    // Defining Grid and Block size
    setUpDevices();
    
    // Calling the Kernal
    julia_kernel<<<GridSize, BlockSize>>>(pixels_device, WindowWidth, WindowHeight, XMin, XMax, YMin, YMax);
	cudaErrorCheck(__FILE__, __LINE__);
    
    // get to the same place (GPU and CPU)
    cudaDeviceSynchronize();
    cudaErrorCheck(__FILE__, __LINE__);

    // copying back to the CPU
    cudaMemcpy(pixels_host, pixels_device, size, cudaMemcpyDeviceToHost);

    // not using Async so that CPU can wait for the GPU and avoid potential errors (using synchronous)


    // Rendering
    glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixels_host);
    glFlush();

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
int main(int argc, char** argv)
{ 
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--Man--Fractals");
   	glutDisplayFunc(display);
   	glutMainLoop();
    return 0;
}
