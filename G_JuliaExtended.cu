// Name:
// Not simple Julia Set on the GPU
// nvcc G_JuliaExtended.cu -o temp -lglut -lGL

/*
 What to do:
 This code displays a simple Julia set fractal using the GPU.
 However, it currently only runs on a 1024x1024 window.

 Your tasks:
 - Modify the code so it works on any given window size.
 - Add color to the fractal — be creative! You will be judged on your artistic flair.
   (Don't cut off your ear or anything, but try to make Vincent wish he'd had a GPU.)
*/

/*
 Purpose:
 To have some fun with your new GPU skills!
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>
#include <math.h>
// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
#define A  -0.84	//Real part of C
#define B  -0.1711	//Imaginary part of C

// Global variables
unsigned int WindowWidth = 1024; // x 
unsigned int WindowHeight = 1024; // y (resolution is 1024x1024 = 1,048,576 pixels)
dim3 BlockSize;
dim3 GridSize;


float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

// Function prototypes
void cudaErrorCheck(const char*, int);
__global__ void colorPixels(float, float, float, float, float);

void setUpDevices(){

BlockSize.x = 16; // remember we work in warps (groups of 32 threads)
BlockSize.y = 16; // 16 x 16 = 256; 256/32 = 8 warps; (16x16 better than giant block)
BlockSize.z = 1;

GridSize.x = ((WindowWidth + BlockSize.x -1)/BlockSize.x);
GridSize.y = ((WindowHeight + BlockSize.y -1)/BlockSize.y);
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



__global__ void colorPixels(float *pixels,int width, int height, float xMin, float yMin, float dx, float dy) 
{
	// dx and dy are the step sizes (stride)
    
    
    float mag,tempX;
	int count;
	
	int maxCount = MAXITERATIONS; // escapes max iterations they don't escape and will stop 
	float maxMag = MAXMAG;        // MAXMAG; outside of bounds will halt as well
	

   /*
	//Getting the offset into the pixel buffer. 
	//We need the 3 because each pixel has a red, green, and blue value.
	id = 3*(threadIdx.x + blockDim.x*blockIdx.x);
	
	//Asigning each thread its x and y value of its pixel.
	x = xMin + dx*threadIdx.x;
	y = yMin + dy*blockIdx.x;
	*/
    float n=100;

	int x_idx = blockIdx.x * blockDim.x + threadIdx.x; // x global pixel index
	int y_idx = blockIdx.y * blockDim.y + threadIdx.y; // y global pixel index

    if (x_idx >= width || y_idx >= height) // dont use extra threads
	{
    return;
    }
    int id = 3 * (x_idx + width * y_idx); // 3 elements
    
    
    float x = xMin + x_idx * dx; // -2 + (0-1023) * S
    float y = yMin + y_idx *dy;



	count = 0;
	mag = sqrt(x*x + y*y);
	while (mag < maxMag && count < maxCount) 
	{
		//We will be changing the x but we need its old value to find y.	
		tempX = x;                 // temp value to update the y
		x = (x*x - y*y + A);         // real part 
		y = (2.0 * tempX * y) + B; // imaginary part
		mag = (x*x + y*y);     // recalculating dist. from origin
		count++;

	
    }
	//Setting the red value
	if(count < maxCount) //It excaped
	{        
       float t =   float(count)/ float(maxCount);            // outside of range execute
		pixels[id]     = 0.2f*t; // RED
		pixels[id + 1] = sin(1.0f)-.76f*t; // GREEN
		pixels[id + 2] = .75f; // BLUE
	}

	else//It Stuck around
	{         
         
        float t = float(count) / float(maxCount);   // Inside of range execute
		pixels[id]     = 0.32f*t; // RED
		pixels[id + 1] = cos(1.0f*3.2f)+t; // GREEN
		pixels[id + 2] =0.0f; // BLUE
    }
	//Setting the green
	//pixels[id+1] = 0.0;
	//Setting the blue 
	//pixels[id+2] = 0.0;
    
}

void display(void) 
{ 

	/// dim3 blockSize, gridSize;
	float *pixelsCPU, *pixelsGPU; 
	float stepSizeX, stepSizeY;
	

	//We need the 3 because each pixel has a red, green, and blue value.
	pixelsCPU = (float *)malloc(WindowWidth*WindowHeight*3*sizeof(float)); // same thing as f_julia just didnt use size variable
	cudaMalloc(&pixelsGPU,WindowWidth*WindowHeight*3*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	stepSizeX = (XMax - XMin)/((float)WindowWidth);
	stepSizeY = (YMax - YMin)/((float)WindowHeight);
	
    setUpDevices();
	//Threads in a block
	/*if(WindowWidth > 1024)
	{
	 	printf("The window width is too large to run with this program\n");
	 	printf("The window width width must be less than 1024.\n");
	 	printf("Good Bye and have a nice day!\n");
	 	exit(0);
	}
	*/


    /*
    blockSize.x = 1024; //WindowWidth;
	blockSize.y = 1;
	blockSize.z = 1;
	
	//Blocks in a grid
	gridSize.x = WindowHeight;
	gridSize.y = 1;
	gridSize.z = 1;
	*/

	colorPixels<<<GridSize, BlockSize>>>(pixelsGPU, WindowWidth, WindowHeight, XMin, YMin, stepSizeX, stepSizeY);
	cudaDeviceSynchronize();
    cudaErrorCheck(__FILE__, __LINE__);
	
	//Copying the pixels that we just colored back to the CPU.
	cudaMemcpyAsync(pixelsCPU, pixelsGPU, WindowWidth*WindowHeight*3*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Putting pixels on the screen.
	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixelsCPU); // working with RGB and Floats
	glFlush(); 

    // Kernal returns pixels[] into pixels_GPU, using GPU memory and then uses memcpy to copy GPU pixels vector dire
    // into CPU to be used in GLDraw pixels


}

int main(int argc, char** argv)
{ 
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--Man--Fractals");
   	glutDisplayFunc(display);
   	glutMainLoop();
}