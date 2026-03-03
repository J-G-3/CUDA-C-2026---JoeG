// Name: Joe Gonzalez
// Ray tracing
// nvcc K_RayTracerWithConstantMemory.cu -o temp -lglut -lGL -lm

/*
 What to do:
 This program creates a random set of spheres and uses ray tracing to render an image of them 
 to be displayed on the screen. In the scene, positive X is to the right, positive Y is up, and 
 positive Z comes out of the screen toward the viewer.

 All the spheres are located within a 2x2x2 cube, and you observe them through a 2x2 viewing window.
 
 Your mission, should you choose to accept it:
 1. The spheres created on the CPU do not change, so transfer them to the GPU and store them in constant memory.
 2. Use CUDA events to time your code execution.
*/

/*
 Purpose:
 To learn how to use constant memory and CUDA events.
*/

// Include files
#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>





// Defines
#define WINDOWWIDTH 1024
#define WINDOWHEIGHT 1024
#define XMIN -1.0f
#define XMAX 1.0f
#define YMIN -1.0f
#define YMAX 1.0f
#define ZMIN -1.0f
#define ZMAX 1.0f
#define NUMSPHERES 100
#define MAXRADIUS 0.2 // The biggest radius a sphere can have.



// Local structures
struct sphereStruct 
{
	float r,b,g; // Sphere color
	float radius;
	float x,y,z; // Sphere center
};

__constant__ sphereStruct SpheresConst[NUMSPHERES]; // this memory is constant and will not change during the execution of the kernal; 
                                                    // CONSTANT MEM IS FASTER FOR READ ONLY DATA
                                                    // HOW THIS LINE WORKS:
                                                    // SpheresConst is an array of sphereStruct objects, sized to  hold 100 spheres
                                                    // creates Spheres const (a 100 element array); each element is a sphereStruct (a struct that holds sphere qualities)


// Globals variables
static int Window; // static variables retain their edited values over different functions 
unsigned int WindowWidth = WINDOWWIDTH; // stay positive cannot represent a negative number 
unsigned int WindowHeight = WINDOWHEIGHT;
dim3 BlockSize, GridSize;
float *PixelsCPU, *PixelsGPU; 
sphereStruct *SpheresCPU;

// Function prototypes
void cudaErrorCheck(const char *, int);
void display();

// void idle();
void KeyPressed(unsigned char , int , int );
__device__ float hit(float , float , float *, sphereStruct);
__global__ void makeSpheresBitMap(float *);
void makeRandomSpheres();
void makeBitMap();
void paintScreen();
void setup();

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

void display()
{
	makeBitMap();	
}

void KeyPressed(unsigned char key, int x, int y)
{	
	if(key == 'q')
	{
		glutDestroyWindow(Window);
		
		// Free host memory.
		free(PixelsCPU); 
		free(SpheresCPU); 
	
		// Free divice memory.
		cudaFree(PixelsGPU); 
		cudaErrorCheck(__FILE__, __LINE__);
		// cudaFree(SpheresGPU); 
		// cudaErrorCheck(__FILE__, __LINE__);
		
		printf("\nGood Bye\n");
		exit(0);
	}
}

__device__ float hit(float pixelx, float pixely, float *dimingValue, sphereStruct sphere)
{
	float dx = pixelx - sphere.x;  //Distance from ray to sphere center in x direction (horizontal distance from the rays origin)
	float dy = pixely - sphere.y;  //Distance from ray to sphere center in y direction (vertical distance from the rays origin)
	float r2 = sphere.radius*sphere.radius;
	if(dx*dx + dy*dy < r2) // if the ray hits the sphere, when dx^2 + dy^2 are greater than r^2 then the ray hits outside the sphere(a miss!)
	{
		float dz = sqrtf(r2 - dx*dx - dy*dy); // Distance from ray to edge of sphere
		*dimingValue = dz/sphere.radius; // n is value between 0 and 1 used for darkening points near edge.
                                         // smaller the dimming value the darker it will be 
		return dz + sphere.z; //  Return the distance to be scaled by; depth of the point relative to the camera
	}
	return (ZMIN- 1.0); //If the ray doesn't hit anything return a number 1 unit behind the box.
}

__global__ void makeSpheresBitMap(float *pixels)
{
	float stepSizeX = (XMAX - XMIN)/((float)WINDOWWIDTH - 1); // 2/(1024-1)
	float stepSizeY = (YMAX - YMIN)/((float)WINDOWHEIGHT - 1);
	
	// Asigning each thread a pixel
	float pixelx = XMIN + threadIdx.x*stepSizeX; // -1 + threadidx.x * 2/(1024-1); so we get a thread for every step in the x direction
	float pixely = YMIN + blockIdx.x*stepSizeY; // same for y direction 
	
	// Finding this pixels location in memory
	int id = 3*(threadIdx.x + blockIdx.x*blockDim.x); // *3 because working with RGB
	
	//initialize rgb values for each pixel to zero (black)
	float pixelr = 0.0f;
	float pixelg = 0.0f;
	float pixelb = 0.0f;
	float hitValue;
	float dimingValue;
	float maxHit = ZMIN -1.0f; // Initializing it to be 1 unit behind the box.
	for(int i = 0; i < NUMSPHERES; i++)
	{
		hitValue = hit(pixelx, pixely, &dimingValue, SpheresConst[i]);              // IMPORTANT: we use &dimmingValue; the & lets us directly change the dimming value from the hit function so everytime we use the func a new dim value is calc and returned
		// do we hit any spheres? If so, how close are we to the center? (i.e. n)
		if(maxHit < hitValue)
		{
			// Setting the RGB value of the sphere but also diming it as it gets close to the side of the sphere.
			pixelr = SpheresConst[i].r * dimingValue; 
			pixelg = SpheresConst[i].g * dimingValue;	
			pixelb = SpheresConst[i].b * dimingValue; 	
			maxHit = hitValue; // reset maxHit value to be the current closest sphere
		}
	}
	// location of color values
	pixels[id] = pixelr; 
	pixels[id+1] = pixelg;
	pixels[id+2] = pixelb;
}

void makeRandomSpheres()
{	
	float rangeX = XMAX - XMIN; // 2
	float rangeY = YMAX - YMIN; // 2
	float rangeZ = ZMAX - ZMIN; // 2
	
	for(int i = 0; i < NUMSPHERES; i++) 
	{
		SpheresCPU[i].x = (rangeX*(float)rand()/RAND_MAX) + XMIN; // RAND_MAX is just some large constant; ensure we get number from zero and 1 
		SpheresCPU[i].y = (rangeY*(float)rand()/RAND_MAX) + YMIN;
		SpheresCPU[i].z = (rangeZ*(float)rand()/RAND_MAX) + ZMIN;
		SpheresCPU[i].r = (float)rand()/RAND_MAX; // random number/large constant
		SpheresCPU[i].g = (float)rand()/RAND_MAX;
		SpheresCPU[i].b = (float)rand()/RAND_MAX;
		SpheresCPU[i].radius = MAXRADIUS*(float)rand()/RAND_MAX; // (.2 * randomNumber)/random large integer
	}
}	



void makeBitMap()
{	
    //NEW: Creating Events
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start); // create an even object
    cudaEventCreate(&stop);

    //Record Start event
    cudaEventRecord(start, 0); // records the start time of the event, 0 is the stream, 0 means default stream

    // copy spheres to constant memory 
    cudaMemcpyToSymbol(SpheresConst, SpheresCPU, NUMSPHERES * sizeof(sphereStruct)); // used to copy data from host memory to constant memory
                                                                                     // cudaMemcpyToSymbol(the array stored in constant memory, array we want copied to GPUs constant memory, size of data being copied)
    cudaErrorCheck(__FILE__, __LINE__);

    // launch the kernal
    makeSpheresBitMap<<<GridSize, BlockSize>>>(PixelsGPU);
    cudaErrorCheck(__FILE__, __LINE__);

    // copy the pixels back
    cudaMemcpyAsync(PixelsCPU, PixelsGPU, WINDOWWIDTH * WINDOWHEIGHT * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaErrorCheck(__FILE__, __LINE__); 

    // Record stop event 
    cudaEventRecord(stop,0);

    // wait for stop event to complete
    cudaEventSynchronize(stop);

    // calculate elapsed time 
    cudaEventElapsedTime(&elapsedTime, start, stop);              // call cuda event record twice with two different variables to get two different time values then plug into cudaeventelapsedtime to get the difference (total time)
    printf("CUDA kernal execution time:%.3f ms \n", elapsedTime);


    // Destroy events (avoid memory leaks as they persist after being declared)
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
 



    /*
	cudaMemcpy(SpheresGPU, SpheresCPU, NUMSPHERES*sizeof(sphereStruct), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	makeSphersBitMap<<<GridSize, BlockSize>>>(PixelsGPU, SpheresGPU);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaMemcpyAsync(PixelsCPU, PixelsGPU, WINDOWWIDTH*WINDOWHEIGHT*3*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	*/
	paintScreen();
}

void paintScreen()
{
	//Putting pixels on the screen.
	glDrawPixels(WINDOWWIDTH, WINDOWHEIGHT, GL_RGB, GL_FLOAT, PixelsCPU); // glDrawPixels (width, height, RGB format, color values are floats, PixelsCPU is the array that holds the color data)
	glFlush();
}

void setup() // sets up devices BLOCKS are the y axis and threads are the x axis
{
	//Allocating memory for the scene that will be displayed to the screen.
	//We need the 3 because each pixel has a red, green, and blue value.
	PixelsCPU = (float *)malloc(WINDOWWIDTH*WINDOWHEIGHT*3*sizeof(float));
	cudaMalloc(&PixelsGPU,WINDOWWIDTH*WINDOWHEIGHT*3*sizeof(float)); 
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Allocating memory for the spheres that will create the scene.
	//This is what you will be changing out for constant memory.
	SpheresCPU= (sphereStruct*)malloc(NUMSPHERES*sizeof(sphereStruct));
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Threads in a block
	if(WINDOWWIDTH > 1024) //To keep the code simple we make sure the scene width fits in a block.
	{
	 	printf("The window width is too large to run with this program\n");
	 	printf("The window width must be less than 1024.\n");
	 	printf("Good Bye and have a nice day!\n");
	 	exit(0);
	}
	BlockSize.x = WINDOWWIDTH;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	//Blocks in a grid
	GridSize.x = WINDOWHEIGHT;
	GridSize.y = 1;
	GridSize.z = 1;
	
	// Seeding the random number generator.
	time_t t;
	srand((unsigned) time(&t)); // seeding our rand values with t to get random values every time 
}

int main(int argc, char** argv)
{ 
	setup();              // setup environment 
	makeRandomSpheres();  // create the random spheres with set RGB, XYZ, and radius 
   	glutInit(&argc, argv); // initializes GLUT command line arguments
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE); // display mode: RGB color mode for rendering | single buffer - screen is updated immediately 
   	glutInitWindowSize(WINDOWWIDTH, WINDOWHEIGHT); // initial size of window created
	Window = glutCreateWindow("Random Spheres"); // creates a window known as variable "window" and "Random Spheres" is set title of window
	glutKeyboardFunc(KeyPressed); // registers keyboard callback function; GLUT will call it whenever key event occurs
   	glutDisplayFunc(display); // registers display callback func; "display" function called when window needs to be drawn/ redrawn
   	glutMainLoop(); // starts GLUT main loop that will not stop until killed or key pressed
}
