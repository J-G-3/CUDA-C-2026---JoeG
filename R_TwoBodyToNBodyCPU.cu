// Name: Joe Gonzalez
// Two body problem
// nvcc R_TwoBodyToNBodyCPU.cu -o temp -lglut -lGLU -lGL
// To stop hit "control c" in the window you launched it from.

/*
 What to do:
 This is some crude code that moves two bodies around in a box, attracted by gravity and 
 repelled when they hit each other. Take this from a two-body problem to an N-body problem, where 
 NUMBER_OF_SPHERES is a #define that you can change. Also clean it up a bit so it is more user friendly.
*/

/*
 Purpose:
 To learn about Nbody code.
*/

// Include files
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Defines
#define XWindowSize 1000
#define YWindowSize 1000
#define STOP_TIME 10000.0
#define DT        0.0001
#define GRAVITY 0.1 
#define MASS 10.0  	
#define DIAMETER 1.0
#define SPHERE_PUSH_BACK_STRENGTH 50.0
#define PUSH_BACK_REDUCTION 0.1
#define DAMP 0.01
#define DRAW 100
#define LENGTH_OF_BOX 6.0
#define MAX_VELOCITY 5.0

// Globals
const float XMax = (LENGTH_OF_BOX/2.0);
const float YMax = (LENGTH_OF_BOX/2.0);
const float ZMax = (LENGTH_OF_BOX/2.0);
const float XMin = -(LENGTH_OF_BOX/2.0);
const float YMin = -(LENGTH_OF_BOX/2.0);
const float ZMin = -(LENGTH_OF_BOX/2.0);

/*
float px1, py1, pz1, vx1, vy1, vz1, fx1, fy1, fz1, mass1; 
float px2, py2, pz2, vx2, vy2, vz2, fx2, fy2, fz2, mass2;
*/

#define NUMBER_OF_SPHERES 3

float px[NUMBER_OF_SPHERES]; // now we diverged from the hard coded p values for the two spheres
float py[NUMBER_OF_SPHERES]; // p: positiion (xyz directions)
float pz[NUMBER_OF_SPHERES];

float vx[NUMBER_OF_SPHERES];
float vy[NUMBER_OF_SPHERES]; // Velocity 
float vz[NUMBER_OF_SPHERES];

float fx[NUMBER_OF_SPHERES];
float fy[NUMBER_OF_SPHERES]; // force
float fz[NUMBER_OF_SPHERES];

float mass[NUMBER_OF_SPHERES]; 



// Function prototypes
void set_initail_conditions();
void Drawwirebox();
void draw_picture();
void keep_in_box();
void get_forces();
void move_bodies(float);
void nbody();
void Display(void);
void reshape(int, int);
int main(int, char**);

void set_initail_conditions()
{ 
    time_t t;
    srand((unsigned) time(&t)); // seeding rand with time 
    int overlap; // replaced yeah buddy with overlap
    float dx, dy, dz, separation; // creating for all spheres (distance) relative to each other

    for (int i = 0; i < NUMBER_OF_SPHERES; i++) {
        px[i] = (LENGTH_OF_BOX - DIAMETER) * rand() / RAND_MAX - (LENGTH_OF_BOX - DIAMETER) / 2.0; // creating random positions 
        py[i] = (LENGTH_OF_BOX - DIAMETER) * rand() / RAND_MAX - (LENGTH_OF_BOX - DIAMETER) / 2.0; // [-box/2, box/2]; [-3,3] 6 units in total 
        pz[i] = (LENGTH_OF_BOX - DIAMETER) * rand() / RAND_MAX - (LENGTH_OF_BOX - DIAMETER) / 2.0; // subtract the length by the diameter in order to get a viable position that is within bounds
																								   // LENGTH - DIAM = Viable position or Bounds
        overlap = 1; // setting condition for while loop to execute
        while (overlap) { 																			// cool shortcut while(overlap) == while overlap is a non zero value
            overlap = 0;
            // Check for overlaps with other bodies
            for (int j = 0; j < i; j++) {
                dx = px[i] - px[j]; // distances from x y z values
                dy = py[i] - py[j];
                dz = pz[i] - pz[j];
                separation = sqrt(dx * dx + dy * dy + dz * dz); // seperation scalar distance from bodies center
                if (separation < DIAMETER) {
                    overlap = 1;  // Overlap detected, regenerate position for body i
                    px[i] = (LENGTH_OF_BOX - DIAMETER) * rand() / RAND_MAX - (LENGTH_OF_BOX - DIAMETER) / 2.0; // create new body if a body inside another 
                    py[i] = (LENGTH_OF_BOX - DIAMETER) * rand() / RAND_MAX - (LENGTH_OF_BOX - DIAMETER) / 2.0;
                    pz[i] = (LENGTH_OF_BOX - DIAMETER) * rand() / RAND_MAX - (LENGTH_OF_BOX - DIAMETER) / 2.0;
                    break;
                }
            }
        }

        vx[i] = 2.0 * MAX_VELOCITY * rand() / RAND_MAX - MAX_VELOCITY; // create random vel for each body; mult by 2 to get [-Max,Max]
        vy[i] = 2.0 * MAX_VELOCITY * rand() / RAND_MAX - MAX_VELOCITY;
        vz[i] = 2.0 * MAX_VELOCITY * rand() / RAND_MAX - MAX_VELOCITY;
        mass[i] = 1.0;  // All bodies with mass 1.0 for simplicity
    }



    // Only for two Spheres
	/*time_t t;
	srand((unsigned) time(&t)); // seeding random number with time
	int yeahBuddy;
	float dx, dy, dz, seperation;
	
	px1 = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0; // picks a random position within the box 
	py1 = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0; // length = 6 units; Diameter = 1 unit
	pz1 = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0; // positions range from [-box/2, box/2]
	
	yeahBuddy = 0; // setting yeahbuddy and then having a while loop occur during = 0 condition 
	while(yeahBuddy == 0)
	{
		px2 = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0; // same for sphere 2 
		py2 = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0; // 
		pz2 = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
		
		dx = px2 - px1; // subtracting distance elements
		dy = py2 - py1; // how far apart are they
		dz = pz2 - pz1;
		seperation = sqrt(dx*dx + dy*dy + dz*dz); // distance scalar from the two spheres
		yeahBuddy = 1; // set yeah buddy to 1 to get out the loop
		if(seperation < DIAMETER) yeahBuddy = 0; // checking if they overlap, have to redo the location if they 
	}
	
	vx1 = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY; // assigning random velocity 
	vy1 = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
	vz1 = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
	
	vx2 = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
	vy2 = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
	vz2 = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
	
	mass1 = 1.0; // mass set to one for simplicity 
	mass2 = 1.0;
    */
}

void Drawwirebox()
{		
	glColor3f (5.0,1.0,1.0); // (r, g, b) sets the color for following drawing commands 
	glBegin(GL_LINE_STRIP);  // tells GL to start drawing a series of connected lines ( a "strip" of lines); 
		glVertex3f(XMax,YMax,ZMax); // right side of box??
		glVertex3f(XMax,YMax,ZMin);	// continuous line 
		glVertex3f(XMax,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMax);
		glVertex3f(XMax,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax); 
		
		glVertex3f(XMin,YMax,ZMax); // left side of box ??
		glVertex3f(XMin,YMax,ZMin);	
		glVertex3f(XMin,YMin,ZMin); 
		glVertex3f(XMin,YMin,ZMax); 
		glVertex3f(XMin,YMax,ZMax);	
	glEnd();
	
	glBegin(GL_LINES); // filling in final edges that were open and not covered by initial.
		glVertex3f(XMin,YMin,ZMax); 
		glVertex3f(XMax,YMin,ZMax);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMin);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMax,ZMin);
		glVertex3f(XMax,YMax,ZMin);		
	glEnd();
	
}

void draw_picture()
{
	
	float radius = DIAMETER/2.0;
	glClear(GL_COLOR_BUFFER_BIT); // used to clear the color buffer bit (the color that is visible on the screen)
	glClear(GL_DEPTH_BUFFER_BIT); // resets depth buffer which
								  // Depth buffer default = to 1(the farthest possible depth) resets depth info of entire frame
	
	Drawwirebox(); // draw the cube 
	
	for (int i = 0; i < NUMBER_OF_SPHERES; i++) {
        glColor3d((float)i / NUMBER_OF_SPHERES, 1.0 - (float)i / NUMBER_OF_SPHERES, 0.0);  // Different colors for each body (r, g ,b)
        glPushMatrix(); // saves current state of matrix onto stack
        glTranslatef(px[i], py[i], pz[i]);// moves sphere to a location defined by coordinates
        glutSolidSphere(radius, 20, 20); // (radius, horizontal slices, vertical slices ) more slices the more smooth and more computation it will take 
        glPopMatrix(); // this makes sure changes only happen to current sphere and no following one; restores changes 
    }

	/*glColor3d(1.0,0.5,1.0);
	glPushMatrix();
	glTranslatef(px1, py1, pz1);
	glutSolidSphere(radius,20,20);
	glPopMatrix();
	
	glColor3d(0.0,0.5,0.0);
	glPushMatrix();
	glTranslatef(px2, py2, pz2);
	glutSolidSphere(radius,20,20);
	glPopMatrix();*/
	
	glutSwapBuffers(); // display the updated scene to the user
	
}

void keep_in_box(int i)
{
	 float halfBoxLength = (LENGTH_OF_BOX - DIAMETER) / 2.0;

    if (px[i] > halfBoxLength) // px not greater than approx 3 makes sense these are our bounds [-3,3] not really equal to 3 but whatever
							   // px[i] > 2, 5/2 
    {
        px[i] = 2.0 * halfBoxLength - px[i]; // 10-px
        vx[i] = -vx[i];
    }
    else if (px[i] < -halfBoxLength) // make sure px not less than approx -3
    {
        px[i] = -2.0 * halfBoxLength - px[i];
        vx[i] = -vx[i];
    }

    if (py[i] > halfBoxLength)
    {
        py[i] = 2.0 * halfBoxLength - py[i];
        vy[i] = -vy[i];
    }
    else if (py[i] < -halfBoxLength)
    {
        py[i] = -2.0 * halfBoxLength - py[i];
        vy[i] = -vy[i];
    }

    if (pz[i] > halfBoxLength)
    {
        pz[i] = 2.0 * halfBoxLength - pz[i];
        vz[i] = -vz[i];
    }
    else if (pz[i] < -halfBoxLength)
    {
        pz[i] = -2.0 * halfBoxLength - pz[i];
        vz[i] = -vz[i];
    }



	/*float halfBoxLength = (LENGTH_OF_BOX - DIAMETER)/2.0;
	
	if(px1 > halfBoxLength) // half of the viable box length (LENGTH - DIAM)
	{
		px1 = 2.0*halfBoxLength - px1; // reflect position 
		vx1 = - vx1; // reverse velocity to simulate a bounce
	}
	else if(px1 < -halfBoxLength)
	{
		px1 = -2.0*halfBoxLength - px1;
		vx1 = - vx1;
	}
	
	if(py1 > halfBoxLength)
	{
		py1 = 2.0*halfBoxLength - py1;
		vy1 = - vy1;
	}
	else if(py1 < -halfBoxLength)
	{
		py1 = -2.0*halfBoxLength - py1;
		vy1 = - vy1;
	}
			
	if(pz1 > halfBoxLength)
	{
		pz1 = 2.0*halfBoxLength - pz1;
		vz1 = - vz1;
	}
	else if(pz1 < -halfBoxLength)
	{
		pz1 = -2.0*halfBoxLength - pz1;
		vz1 = - vz1;
	}
	
	if(px2 > halfBoxLength)
	{
		px2 = 2.0*halfBoxLength - px2;
		vx2 = - vx2;
	}
	else if(px2 < -halfBoxLength)
	{
		px2 = -2.0*halfBoxLength - px2;
		vx2 = - vx2;
	}
	
	if(py2 > halfBoxLength)
	{
		py2 = 2.0*halfBoxLength - py2;
		vy2 = - vy2;
	}
	else if(py2 < -halfBoxLength)
	{
		py2 = -2.0*halfBoxLength - py2;
		vy2 = - vy2;
	}
			
	if(pz2 > halfBoxLength)
	{
		pz2 = 2.0*halfBoxLength - pz2;
		vz2 = - vz2;
	}
	else if(pz2 < -halfBoxLength)
	{
		pz2 = -2.0*halfBoxLength - pz2;
		vz2 = - vz2;
	}*/
}

void get_forces()
{
    
	float dx, dy, dz, r, r2, forceMag; // distances and force magnitude

    // Reset forces to zero
    for (int i = 0; i < NUMBER_OF_SPHERES; i++) {
        fx[i] = fy[i] = fz[i] = 0.0; // setting all forces to zero for all bodies
    }

    // Calculate forces between all pairs of bodies
    for (int i = 0; i < NUMBER_OF_SPHERES; i++) {
        for (int j = i + 1; j < NUMBER_OF_SPHERES; j++) {
            dx = px[j] - px[i]; // me minus them (get relative force from someone unto me by my position minus theres and using that in the force calculation)
            dy = py[j] - py[i]; // i is the reference and j is them
            dz = pz[j] - pz[i];

            r2 = dx * dx + dy * dy + dz * dz; // dx^2 + dy^2 + dz^2 = r^2
            r = sqrt(r2); // distance value between the two bodies

            forceMag = (mass[i] * mass[j] * GRAVITY) / r2; // gravity of force on one body to another
														   // fmag is the mag of gravitation force between two bodies
            // Update forces on body i and body j
            fx[i] += forceMag * dx / r; // multiply by the unit vector (dx/r)
			 							// this will give you force components along each axis
            fy[i] += forceMag * dy / r;
            fz[i] += forceMag * dz / r;

            fx[j] -= forceMag * dx / r; // opp direction (newtons third law equal and oppisite reaction)
            fy[j] -= forceMag * dy / r;
            fz[j] -= forceMag * dz / r;
        }
    }
	
	
	/*for (int i = 0; i < NUMBER_OF_SPHERES; i++)
    {
        fx[i] = 0.0; // set all forces to zero
        fy[i] = 0.0;
        fz[i] = 0.0;
    }

    for (int i = 0; i < NUMBER_OF_SPHERES; i++)
    {
        for (int j = i + 1; j < NUMBER_OF_SPHERES; j++)  // Avoid double counting
        {
            float dx = px[j] - px[i]; // calculate the distances
            float dy = py[j] - py[i];
            float dz = pz[j] - pz[i];

            float r2 = dx * dx + dy * dy + dz * dz;
            float r = sqrt(r2);

            float forceMag = mass[i] * mass[j] * GRAVITY / r2;

            // Add force to body i
            fx[i] += forceMag * dx / r;
            fy[i] += forceMag * dy / r;
            fz[i] += forceMag * dz / r;

            // Add force to body j (opposite direction)
            fx[j] -= forceMag * dx / r;
            fy[j] -= forceMag * dy / r;
            fz[j] -= forceMag * dz / r;
        }*/
    }

	/*float dx,dy,dz,r,r2,dvx,dvy,dvz,forceMag,inout;
	
	dx = px2 - px1;
	dy = py2 - py1;
	dz = pz2 - pz1;
				
	r2 = dx*dx + dy*dy + dz*dz;
	r = sqrt(r2);

	forceMag =  mass1*mass2*GRAVITY/r2;
			
	if (r < DIAMETER)
	{
		dvx = vx2 - vx1;
		dvy = vy2 - vy1;
		dvz = vz2 - vz1;
		inout = dx*dvx + dy*dvy + dz*dvz;
		if(inout <= 0.0)
		{
			forceMag +=  SPHERE_PUSH_BACK_STRENGTH*(r - DIAMETER);
		}
		else
		{
			forceMag +=  PUSH_BACK_REDUCTION*SPHERE_PUSH_BACK_STRENGTH*(r - DIAMETER);
		}
	}

	fx1 = forceMag*dx/r;
	fy1 = forceMag*dy/r;
	fz1 = forceMag*dz/r;
	fx2 = -forceMag*dx/r;
	fy2 = -forceMag*dy/r;
	fz2 = -forceMag*dz/r;*/


void move_bodies(float time) // updates the velocity and position of all bodies
{
	for (int i = 0; i < NUMBER_OF_SPHERES; i++) {
        if (time == 0.0) {
            vx[i] += 0.5 * DT * (fx[i] - DAMP * vx[i]) / mass[i]; // DT = .0001 (the change in time); DAMP = .01 (damping factor)
            vy[i] += 0.5 * DT * (fy[i] - DAMP * vy[i]) / mass[i]; // F/m = a; then 
            vz[i] += 0.5 * DT * (fz[i] - DAMP * vz[i]) / mass[i]; // Velocity is incremented by half a time step initially 
        } else {												  // v*DAMP is representing the dampening force that is acting on the body
            vx[i] += DT * (fx[i] - DAMP * vx[i]) / mass[i]; 
            vy[i] += DT * (fy[i] - DAMP * vy[i]) / mass[i];
            vz[i] += DT * (fz[i] - DAMP * vz[i]) / mass[i];
        }

        px[i] += DT * vx[i]; // updating position based on calculated velocity
        py[i] += DT * vy[i]; 
        pz[i] += DT * vz[i];

    	keep_in_box(i);  // Ensure bodies stay inside the box

    }



	/*
	if(time == 0.0)
	{
		vx1 += 0.5*DT*(fx1 - DAMP*vx1)/mass1;
		vy1 += 0.5*DT*(fy1 - DAMP*vy1)/mass1;
		vz1 += 0.5*DT*(fz1 - DAMP*vz1)/mass1;
		
		vx2 += 0.5*DT*(fx2 - DAMP*vx2)/mass2;
		vy2 += 0.5*DT*(fy2 - DAMP*vy2)/mass2;
		vz2 += 0.5*DT*(fz2 - DAMP*vz2)/mass2;
	}
	else
	{
		vx1 += DT*(fx1 - DAMP*vx1)/mass1;
		vy1 += DT*(fy1 - DAMP*vy1)/mass1;
		vz1 += DT*(fz1 - DAMP*vz1)/mass1;
		
		vx2 += DT*(fx2 - DAMP*vx2)/mass2;
		vy2 += DT*(fy2 - DAMP*vy2)/mass2;
		vz2 += DT*(fz2 - DAMP*vz2)/mass2;
	}

	px1 += DT*vx1;
	py1 += DT*vy1;
	pz1 += DT*vz1;
	
	px2 += DT*vx2;
	py2 += DT*vy2;
	pz2 += DT*vz2;
	
	keep_in_box();
	*/
}

void nbody()
{	
	int    tdraw = 0; // how often will the simulation update
	float  time = 0.0; // current sim stime 

	set_initail_conditions(); // set p's and v's
	
	draw_picture(); // create/ color spheres and create box drawing (cube)
	
	while(time < STOP_TIME) // time < 10000
	{
		get_forces(); // calculate and update forces
	
		move_bodies(time); // updates position and velocites based on time
	
		tdraw++; // update tdraw +1
		if(tdraw == DRAW) // if tdraw = 100
		{
			draw_picture(); // update visual with updated position (gltranslate)
			tdraw = 0; // reset tdraw
		}
		
		time += DT; // increment time by DT
	}
	printf("\n DONE \n"); // when stop time exceeded print ...
	while(1);
}

void Display(void)
{
	gluLookAt(0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0); // setting camera pos
	glClear(GL_COLOR_BUFFER_BIT); // clear prev frame color
	glClear(GL_DEPTH_BUFFER_BIT); // clear depth info
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); // polygon drawing mode set to fill(front and back faces drawn as filled )
	glutSwapBuffers(); // swaps front and back buffers use newly drawn frame
	glFlush(); // forces GPU to finish all operations before moving; 
	nbody(); // main nbody call to use main funcs 
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h); // sets area in window to generate scene 0,0 x,y w,h width and height

	glMatrixMode(GL_PROJECTION);// switch to proj matrix how 3D is projected on 2D space

	glLoadIdentity(); // replaces current matrix, clears prev transformations to start fresh

	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, 50.0);// defines shape of frustrum?? left, right, bottom, top, near, far (near and far are where obj still visible )

	glMatrixMode(GL_MODELVIEW); // manipulate objects in camera view ??
}

int main(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB); // (double buffering; drawn off screen buffer, enables depth buffering, color mode set to RGB)
	glutInitWindowSize(XWindowSize,YWindowSize); // setting window size
	glutInitWindowPosition(0,0); // setting position of window on screen
	glutCreateWindow("n Body 3D");


	// defines different lighting options
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0}; // 
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};// affects all objects uniformly 
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};// color of light scattered across all surfaces (white)
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};// defines color for reflections (white)
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};// light that affects entire scene; when no direct light hitting objects they are able to be seen 
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0}; // specular reflection of materials (full specular reflection)
	GLfloat mat_shininess[]  = {10.0};// shininess of material and glossiness of surface higher= shinier

	glClearColor(0.0, 0.0, 0.0, 0.0); // backgroup color
	glShadeModel(GL_SMOOTH); // smooth shading 
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);// affects how light interacts with obj (details)



	glLightfv(GL_LIGHT0, GL_POSITION, light_position); // light setup and positions
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);

	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular); // shininess and reflection for front facing objects 
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	
	glEnable(GL_LIGHTING); // enables lighting
	glEnable(GL_LIGHT0);   // en light src
	glEnable(GL_COLOR_MATERIAL); // color materials
	glEnable(GL_DEPTH_TEST); // depth testing: obscure further objects
	glutDisplayFunc(Display); // registers display using our display func
	glutReshapeFunc(reshape); // regusters reshape as reshape func
	glutMainLoop(); // continue running and looping
	return 0;
}


