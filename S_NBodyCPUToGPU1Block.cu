// Name: Joe Gonzalez
// Creating a GPU nBody simulation from an nBody CPU simulation. 
// nvcc S_NBodyCPUToGPU1Block.cu -o temp -lglut -lm -lGLU -lGL

/*
 What to do:
 This is some lean nBody code that runs on the CPU. Rewrite it, keeping the same general format, 
 but offload the compute-intensive parts of the code to the GPU for acceleration.
 Note: The code takes two arguments as inputs:
 1. The number of bodies to simulate, (We will keep the number of bodies under 1024 for this HW so it can be run on one block.)
 2. Whether to draw sub-arrangements of the bodies during the simulation (1), or only the first and last arrangements (0).
*/

/*
 Purpose:
 To learn how to move an Nbody CPU simulation to an Nbody GPU simulation..
*/

// Include files
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h> // needed for cudamalloc, etc. (cuda related functions )

// Defines
#define PI 3.14159265359
#define DRAW_RATE 10

// This is to create a Lennard-Jones type function G/(r^p) - H(r^q). (p < q) p has to be less than q.
// In this code we will keep it a p = 2 and q = 4 problem. The diameter of a body is found using the general
// case so it will be more robust but in the code leaving it as a set 2, 4 problem make the coding much easier.
#define G 10.0
#define H 10.0
#define LJP 2.0
#define LJQ 4.0

#define DT 0.0001
#define RUN_TIME 1.0

// Globals
int N, DrawFlag;
float3 *P, *V, *F;
float  *M;

float3 *d_P, *d_V, *d_F;
float  *d_M;

float GlobeRadius, Diameter, Radius;
float Damp;

// Function prototypes
void keyPressed(unsigned char, int, int);
long elaspedTime(struct timeval, struct timeval);
void drawPicture();
void timer();
void setup();
void nBody();
int main(int, char**);


// One kernel to calculate forces, velocities, and positions all in one


__global__ void nBodyKernel(float3 *P, float3 *V, float3 *F, float *M,
                             int N, float dt, float damp, float runTime)
{
    int i = threadIdx.x; // one block so global idx just within that one block
    if(i >= N) return;

    __shared__ float time; // time will be shared because it must be known across all blocks

    if(threadIdx.x == 0)
    {
        time = 0.0f;
    }

    __syncthreads();

    float dx, dy, dz, d, d2, force_mag;

    while(time < runTime) // while current time is less than 1.0
    {
        // zero this thread's force
        F[i].x = 0.0f;
        F[i].y = 0.0f;
        F[i].z = 0.0f;

        __syncthreads();

        // compute forces against all other bodies
        for(int j = 0; j < N; j++)
        {
            if(i == j) continue; // we do not car about the force upon ourselve

            dx = P[j].x - P[i].x; // them - me i is reference (position difference)
            dy = P[j].y - P[i].y;
            dz = P[j].z - P[i].z;

            d2 = dx*dx + dy*dy + dz*dz; // distance^2, 
            d  = sqrtf(d2); // distance

            force_mag = (G * M[i] * M[j]) / d2- (H * M[i] * M[j]) / (d2 * d2); // Gravitational force magnitude using lennard jones constants

            F[i].x += force_mag * dx / d; // calculating forces in specific directions (xyz)
            F[i].y += force_mag * dy / d; // only using addition because we calculate all the forces individually; 
            F[i].z += force_mag * dz / d;
        }

        __syncthreads(); // wait for all threads to finish respective calculations

        // update velocities and positions
        if(time == 0.0f)
        {
            V[i].x += (F[i].x / M[i]) * 0.5f * dt; // updating velocity based on acceleration and time
            V[i].y += (F[i].y / M[i]) * 0.5f * dt;
            V[i].z += (F[i].z / M[i]) * 0.5f * dt;
        }
        else
        {
            V[i].x += ((F[i].x - damp * V[i].x) / M[i]) * dt; // use damp(air resistance)
            V[i].y += ((F[i].y - damp * V[i].y) / M[i]) * dt;
            V[i].z += ((F[i].z - damp * V[i].z) / M[i]) * dt;
        }

        P[i].x += V[i].x * dt; // update position based on time and velocity
        P[i].y += V[i].y * dt;
        P[i].z += V[i].z * dt;

        __syncthreads(); // wait til all threads are done

        if(threadIdx.x == 0){
            time += dt; // 
        }
        __syncthreads();
    }
}



// CPU functions — unchanged from original

void keyPressed(unsigned char key, int x, int y)
{
    if(key == 's') timer(); // step in time
    if(key == 'q') exit(0); // kill program
}

long elaspedTime(struct timeval start, struct timeval end)
{
    long startTime = start.tv_sec * 1000000 + start.tv_usec;
    long endTime   = end.tv_sec   * 1000000 + end.tv_usec;
    return endTime - startTime;
}

void drawPicture()
{
    glClear(GL_COLOR_BUFFER_BIT); // clear color and depth
    glClear(GL_DEPTH_BUFFER_BIT);

    glColor3d(1.0, 1.0, 0.5); // rgb
    for(int i = 0; i < N; i++) // update position and set sphere quality 
    {
        glPushMatrix();
        glTranslatef(P[i].x, P[i].y, P[i].z);
        glutSolidSphere(Radius, 20, 20);
        glPopMatrix();
    }

    glutSwapBuffers(); // swap to buffer with updated data
}

void timer() // time a single time step
{
    struct timeval start, end;
    long computeTime;

    drawPicture();
    gettimeofday(&start, NULL);
        nBody();
    gettimeofday(&end, NULL);
    drawPicture();

    computeTime = elaspedTime(start, end);
    printf("\n The compute time was %ld microseconds.\n\n", computeTime);
}


// setup — same logic as CPU; adds GPU alloc and upload

void setup()
{
    float randomAngle1, randomAngle2, randomRadius; // 
    float d, dx, dy, dz;
    int   test;

    Damp = 0.5f;

    // allocating CPU 
    M = (float*)  malloc(N * sizeof(float)); // allocating mem for 
    P = (float3*) malloc(N * sizeof(float3));
    V = (float3*) malloc(N * sizeof(float3));
    F = (float3*) malloc(N * sizeof(float3));

    // allocating GPU
    cudaMalloc((void**)&d_M, N * sizeof(float));
    cudaMalloc((void**)&d_P, N * sizeof(float3));
    cudaMalloc((void**)&d_V, N * sizeof(float3));
    cudaMalloc((void**)&d_F, N * sizeof(float3));

    Diameter = pow(H / G, 1.0 / (LJQ - LJP)); // diameter calc from lennard jones formula
    Radius   = Diameter / 2.0f;
    // G and H are strength constants 
    // r is the distance between the 2 bodies 
    // LJP and LJQ are p and q exponents (p = 2, q = 4)

    // pow(base, exponent) --> (H/G)^(1/(q-p))


	// Using the radius of a body and a 68% packing ratio to find the radius of a global sphere that should hold all the bodies.
	// Then we double this radius just so we can get all the bodies setup with no problems. 
    float totalVolume = float(N) * (4.0f/3.0f) * PI * Radius*Radius*Radius; // just collective volume of all spheres
    totalVolume /= 0.68f; // do this because of packing ratio i.e. making sure we have enough space for all spheres 
    float totalRadius = pow(3.0f * totalVolume / (4.0f * PI), 1.0f/3.0f); // use algebra from volume formula to get radius

    // Radius of the sphere that can be the max for all spheres
    GlobeRadius = 2.0f * totalRadius;

    for(int i = 0; i < N; i++)
    {
        test = 0;
        while(test == 0)
        {
            randomAngle1 = ((float)rand()/(float)RAND_MAX) * 2.0f * PI; // Rand angles used in pos calcu
            randomAngle2 = ((float)rand()/(float)RAND_MAX) * PI;

            randomRadius = ((float)rand()/(float)RAND_MAX) * GlobeRadius;// random radiuses (ratio of global)

            P[i].x = randomRadius * cos(randomAngle1) * sin(randomAngle2);
            P[i].y = randomRadius * sin(randomAngle1) * sin(randomAngle2);
            P[i].z = randomRadius * cos(randomAngle2);

            test = 1; // set exit condition
            for(int j = 0; j < i; j++)
            {
                dx = P[i].x - P[j].x;
                dy = P[i].y - P[j].y;
                dz = P[i].z - P[j].z;
                d  = sqrt(dx*dx + dy*dy + dz*dz);
                if(d < Diameter) {  // if spheres overlapping restart and get a new pos
                test = 0; 
                break; }
            }
        }

        V[i].x = V[i].y = V[i].z = 0.0f;
        F[i].x = F[i].y = F[i].z = 0.0f;
        M[i] = 1.0f;
    }

    cudaMemcpy(d_M, M, N * sizeof(float),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_P, P, N * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, N * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, F, N * sizeof(float3), cudaMemcpyHostToDevice);

    printf("\n To start timing type s.\n");
}

// nBody — launches the one kernel and waits

void nBody()
{
    nBodyKernel<<<1, N>>>(d_P, d_V, d_F, d_M, N, DT, Damp, RUN_TIME);

    cudaDeviceSynchronize();

    cudaMemcpy(P, d_P, N * sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(V, d_V, N * sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(F, d_F, N * sizeof(float3), cudaMemcpyDeviceToHost);
}

// main — unchanged from original

int main(int argc, char** argv)
{
    if(argc < 3)
    {
        printf("\n You need to enter the number of bodies (an int)");
        printf("\n and if you want to draw the bodies as they move (1 draw, 0 don't draw),");
        printf("\n on the command line.\n");
        exit(0);
    }
    else
    {
        N        = atoi(argv[1]);
        DrawFlag = atoi(argv[2]);
    }

    setup();

    int XWindowSize = 1000;
    int YWindowSize = 1000;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
    glutInitWindowSize(XWindowSize, YWindowSize);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("nBody Test");

    GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
    GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
    GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
    GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
    GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
    GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
    GLfloat mat_shininess[]  = {10.0};

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glShadeModel(GL_SMOOTH);
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);
    glLightfv(GL_LIGHT0, GL_AMBIENT,  light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
    glMaterialfv(GL_FRONT, GL_SPECULAR,  mat_specular);
    glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_DEPTH_TEST);

    glutKeyboardFunc(keyPressed);
    glutDisplayFunc(drawPicture);

    float3 eye  = {0.0f, 0.0f, 2.0f * GlobeRadius};
    float  near = 0.2f;
    float  far  = 5.0f * GlobeRadius;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-0.2, 0.2, -0.2, 0.2, near, far);
    glMatrixMode(GL_MODELVIEW);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    gluLookAt(eye.x, eye.y, eye.z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    glutMainLoop();
    return 0;
}