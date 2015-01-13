#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <GL/glew.h>
#include <CL/opencl.h>
#include <GL/freeglut.h>
#include <GL/glx.h>

int sample_rate = 100000;

/* gl variables */
GLfloat angle = 0.0f;
int refresh_mills = 1000.0/30.0; // refresh interval in milliseconds
bool full_screen_mode = false;
char title[] = "Game of Life on OpenCL (shared)";  // Windowed mode's title
int window_width  = 512;     // Windowed mode's width
int window_height = 512;     // Windowed mode's height
int window_pos_x   = 50;      // Windowed mode's top-left corner x
int window_pos_y   = 50;      // Windowed mode's top-left corner y
double zoom = 1.0;
double ortho_left = -1.0;
double ortho_right = 1.0;
double ortho_top = 1.0;
double ortho_bottom = -1.0;
clock_t wall_clock = 0;

GLuint frame_buffer_name = 0;
GLuint rendered_texture, rendered_texture_out;

/* gl functions*/
void initGL(int argc, char *argv[]);
void startGL();
void displayTimer(int value);
void generationTimer(int value);
void display();
void mouse(int button, int state, int x, int y);
void specialKeys(int key, int x, int y);
void reshape(GLsizei width, GLsizei height);

/* cl kernel */
const char *kernel_source = "devGolGenerateShr.cl";
char *source_string = NULL;

/* cl variables */
cl_context_properties properties[7];
cl_context context;
cl_command_queue command_quque;
cl_platform_id platform;
cl_device_id device;
cl_program program;
cl_kernel kernel;
cl_mem dev_gol_map_in;
cl_mem dev_gol_map_out;
cl_mem dev_gol_image;
size_t size;

clGetGLContextInfoKHR_fn myGetGLContextInfoKHR;

/* work size info */
size_t elements_size[2];
size_t global_work_size[2];
size_t local_work_size[2];
size_t param_data_bytes;
size_t kernel_length;
size_t build_log_size_ret;

/* debug */
cl_int err;
char *build_log;

/* utility function */
int loadProgramSource(const char *filename, char **p_source_string, size_t *length);
void clean();
void die(cl_int err, const char* str);

/* game variables */
char *gol_map = NULL, *gol_tmap = NULL;
int gol_map_width = 512;
int gol_map_height = 512;
long long int gol_generation = 0;

/* game functions */
void golMapClear();
void golMapDump();
void golMapGenerate();
void golMapRandFill();
void golMapStaticFill();
inline char golCellNext(int x, int y);
inline char golCellGet(int x, int y);
inline void golCellSet(int x, int y, char value);
inline void golCellDraw(int x, int y);

int main(int argc, char *argv[])
{
    initGL(argc, argv);

    elements_size[0] = gol_map_width; //cell slots
    elements_size[1] = gol_map_height; //cell slots
    local_work_size[0] = 32;
    local_work_size[1] = 32;
    global_work_size[0] = (size_t)ceil((double)elements_size[0] / local_work_size[0]) * local_work_size[0];
    global_work_size[1] = (size_t)ceil((double)elements_size[1] / local_work_size[1]) * local_work_size[1];

    printf("global_work_size[0]=%u, local_work_size[0]=%u, elements_size[0]=%u, work_groups_x=%u\n",
            global_work_size[0], local_work_size[0], elements_size[0], global_work_size[0] / local_work_size[0]);
    printf("global_work_size[1]=%u, local_work_size[1]=%u, elements_size[1]=%u, work_groups_y=%u\n",
            global_work_size[1], local_work_size[1], elements_size[1], global_work_size[1] / local_work_size[1]);

    /* allocate host memory */
    gol_map      = (char *)malloc(sizeof(cl_char) * global_work_size[0] * global_work_size[1]);
    gol_tmap     = (char *)malloc(sizeof(cl_char) * global_work_size[0] * global_work_size[1]);
    /* end allocate host memory */

    /* init gol_map */
    golMapRandFill();
    /* end init gol_map */
    err = clGetPlatformIDs(1, &platform, NULL);
    die(err, "clGetPlatformIds");

    /*
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    die(err, "clGetDeviceIDs");
    */

    /*
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    die(err, "clCreateContext");
    */

    properties[0] = CL_GL_CONTEXT_KHR;  properties[1] = (cl_context_properties)glXGetCurrentContext();
    properties[2] = CL_GLX_DISPLAY_KHR; properties[3] = (cl_context_properties)glXGetCurrentDisplay();
    properties[4] = CL_CONTEXT_PLATFORM;  properties[5] = (cl_context_properties)platform;
    properties[6] = 0;

    myGetGLContextInfoKHR = (clGetGLContextInfoKHR_fn)clGetExtensionFunctionAddress("clGetGLContextInfoKHR");

    myGetGLContextInfoKHR(properties, CL_DEVICES_FOR_GL_CONTEXT_KHR, sizeof(cl_device_id), &device, &size);

    context =  clCreateContext(properties, 1, &device, NULL, NULL, &err);
    die(err, "clCreateContext");

    printf("dev=%u, ctx=%u\n", device, context);

    command_quque = clCreateCommandQueue(context, device, 0, &err);
    die(err, "clCreateCommandQueue");

    /* create buffers */
    dev_gol_image = clCreateFromGLTexture2D(context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, rendered_texture, &err);
    die(err, "clCreateBuffer dev_gol_image");

    dev_gol_map_in = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_char) * global_work_size[0] * global_work_size[1] , NULL, &err);
    die(err, "clCreateBuffer dev_gol_map_in");

    dev_gol_map_out = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_char) * global_work_size[0] * global_work_size[1], NULL, &err);
    die(err, "clCreateBuffer dev_gol_map_out");
    /* end create buffers */

    loadProgramSource(kernel_source, &source_string, &kernel_length);

    program = clCreateProgramWithSource(context, 1, (const char **)&source_string, &kernel_length, &err);
    die(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(err != CL_SUCCESS)
    {
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_size_ret);

        build_log = (char *)malloc(build_log_size_ret);

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, build_log_size_ret, build_log, NULL);

        printf("%s\n", build_log);
        exit(1);
    }
    //die(err, "clBuildProgram");
    kernel = clCreateKernel(program, "devGolGenerate", &err);
    die(err, "clCreateKernel devGolGenerate");

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&dev_gol_map_in);
    die(err, "clSetKernelArg 0");
    err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&dev_gol_map_out);
    die(err, "clSetKernelArg 1");
    err  = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&dev_gol_image);
    die(err, "clSetKernelArg 1");
    err  = clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&elements_size[0]);
    die(err, "clSetKernelArg 3");
    err  = clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&elements_size[1]);
    die(err, "clSetKernelArg 4");

    err  = clEnqueueWriteBuffer(command_quque, dev_gol_map_in, CL_FALSE, 0, sizeof(cl_char) * global_work_size[0] * global_work_size[1] , gol_map, 0, NULL, NULL);
    die(err, "clEnqueueWriteBuffer");

    startGL();

    /* output result */
    clean();
    return 0;
}

void die(cl_int err, const char* str)
{
    if(err != CL_SUCCESS)
    {
        fprintf(stderr, "Error code %d: %s\n", err, str);
        clean();
        exit(1);
    }
}

void clean()
{
    if(source_string) free(source_string);
    if(kernel) free(kernel);
    if(program) free(program);
    if(command_quque) free(command_quque);
    if(context) free(context);
    if(device) free(device);
    if(platform) free(platform);
    if(dev_gol_map_in) free(dev_gol_map_in);
    if(dev_gol_map_out) free(dev_gol_map_out);

    free(gol_map);
    free(gol_tmap);
}

int loadProgramSource(const char *filename, char **p_source_string, size_t *length)
{
    FILE *file;
    size_t source_length;

    file = fopen(filename, "rb");
    if(file == 0)
    {
        return 1;
    }

    fseek(file, 0, SEEK_END);
    source_length = ftell(file);
    fseek(file, 0, SEEK_SET);

    *p_source_string = (char *)malloc(source_length + 1);
    if(fread(*p_source_string, source_length, 1, file) != 1)
    {
        fclose(file);
        free(*p_source_string);
        return 1;
    }

    fclose(file);
    *length = source_length;
    (*p_source_string)[source_length] = '\0';

    return 0;
}

void initGL(int argc, char *argv[])
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(window_width, window_height);
    glutInitWindowPosition(window_pos_x, window_pos_y);
    glutCreateWindow(title);

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutSpecialFunc(specialKeys); // Register callback handler for special-key event
    glutMouseFunc(mouse);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    /*
    //framebuffer
    glGenFramebuffers(1, &frame_buffer_name);
    glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_name);
*/
    //texture
    glEnable(GL_TEXTURE_2D);

    glGenTextures(1, &rendered_texture);
    glBindTexture(GL_TEXTURE_2D, rendered_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, gol_map_width, gol_map_height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, 0);

    glFinish();
}

void startGL()
{
    glutTimerFunc(0, displayTimer, 0);
    glutTimerFunc(0, generationTimer, 0);
    glutMainLoop();
}

void displayTimer(int value)
{
    glutPostRedisplay();
    glutTimerFunc(refresh_mills, displayTimer, 0);
}

void generationTimer(int value)
{
    err = clEnqueueAcquireGLObjects(command_quque, 1, &dev_gol_image, 0, 0, 0);
    die(err, "clEnqueueAcquireGLObjects");

    //err  = clEnqueueWriteBuffer(command_quque, dev_gol_map_in, CL_FALSE, 0, sizeof(cl_char) * global_work_size[0] * global_work_size[1] , gol_map, 0, NULL, NULL);
    //die(err, "clEnqueueWriteBuffer");

    clEnqueueNDRangeKernel(command_quque, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    die(err, "clEnqueueNDRangeKernel");

    //err = clEnqueueReadBuffer(command_quque, dev_gol_map_out, CL_TRUE, 0, sizeof(cl_char) * global_work_size[0] * global_work_size[1] , gol_tmap, 0, NULL, NULL);
    //die(err, "clEnqueueReadBuffer");
    clFinish(command_quque);

    err = clEnqueueCopyBuffer(command_quque, dev_gol_map_out, dev_gol_map_in, 0, 0, sizeof(unsigned char) * global_work_size[0] * global_work_size[1], 0, NULL, NULL);
    die(err, "clEnqueueCopyBuffer");

    err = clEnqueueReleaseGLObjects(command_quque, 1,  &dev_gol_image, 0, 0, NULL);
    die(err, "clEnqueueReleaseGLObjects");
    //memcpy(gol_map, gol_tmap, sizeof(cl_char) * global_work_size[0] * global_work_size[1]);

    gol_generation++;

    if(gol_generation % sample_rate == 0){
        clock_t now = clock();
        double fps = (double)sample_rate / ( (now - wall_clock) / CLOCKS_PER_SEC );
        printf("generation : %d\n", gol_generation);
        printf("fps = %lf\n", fps);
        wall_clock = now;
    }

    glutTimerFunc(0, generationTimer, 0);
}

void display()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glOrtho(ortho_left, ortho_right, ortho_bottom, ortho_top, -10, 10);
    glScalef(zoom, zoom, 1.0);

    glEnable(GL_TEXTURE_2D);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, gol_map_width, gol_map_height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, gol_map);
    glBindTexture(GL_TEXTURE_2D, rendered_texture);

    glBegin(GL_QUADS);
        glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, 0.0);
        glTexCoord2f(1.0, 0.0); glVertex3f( 1.0, -1.0, 0.0);
        glTexCoord2f(1.0, 1.0); glVertex3f( 1.0,  1.0, 0.0);
        glTexCoord2f(0.0, 1.0); glVertex3f(-1.0,  1.0, 0.0);
    glEnd();

    glutSwapBuffers();
}

void mouse(int button, int state, int x, int y)
{
    // Wheel reports as button 3(scroll up) and button 4(scroll down)
    if ((button == 3) || (button == 4)) // It's a wheel event
    {
        // Each wheel event reports like a button click, GLUT_DOWN then GLUT_UP
        if (state == GLUT_UP) return; // Disregard redundant GLUT_UP events
        //printf("Scroll %s At %d %d\n", (button == 3) ? "Up" : "Down", x, y);

        if(button == 3)
            zoom += 0.1;
        else
            zoom -= 0.1;

        if(zoom < 0) zoom = 0.1;

        GLfloat aspect = (GLfloat)window_width / (GLfloat)window_height;

        if(window_width >= window_height)
        {
            ortho_left = (GLfloat)x / window_width - 2.0 / zoom;
            ortho_right = (GLfloat)x / window_width + 2.0 / zoom;
            ortho_top = (GLfloat)y / window_height / aspect + 2.0 / zoom;
            ortho_bottom = (GLfloat)y / window_height / aspect - 2.0 / zoom;
        }
        else
        {
            ortho_left = (GLfloat)x / window_width * aspect - 2.0 / zoom;
            ortho_right = (GLfloat)x / window_width * aspect + 2.0 / zoom;
            ortho_top = (GLfloat)y / window_height + 2.0 / zoom;
            ortho_bottom = (GLfloat)y / window_height - 2.0 / zoom;
        }

    }else{  // normal button event
        //printf("Button %s At %d %d\n", (state == GLUT_DOWN) ? "Down" : "Up", x, y);
    }
}

void specialKeys(int key, int x, int y) {
    switch (key) {
        case GLUT_KEY_F1:    // F1: Toggle between full-screen and windowed mode
            full_screen_mode = !full_screen_mode;         // Toggle state
            if (full_screen_mode) {                     // Full-screen mode
                window_pos_x   = glutGet(GLUT_WINDOW_X); // Save parameters for restoring later
                window_pos_y   = glutGet(GLUT_WINDOW_Y);
                window_width  = glutGet(GLUT_WINDOW_WIDTH);
                window_height = glutGet(GLUT_WINDOW_HEIGHT);
                glutFullScreen();                      // Switch into full screen
            } else {                                         // Windowed mode
                glutReshapeWindow(window_width, window_height); // Switch into windowed mode
                glutPositionWindow(window_pos_x, window_pos_y);   // Position top-left corner
            }
            break;
        case GLUT_KEY_HOME:
            ortho_left = ortho_top = 1.0;
            ortho_right = ortho_bottom = -1.0;
            zoom = 1.0;
            glutPostRedisplay();
            break;
        case GLUT_KEY_END:
            glutLeaveMainLoop();
            break;
    }
}

void reshape(GLsizei width, GLsizei height)
{
    // Compute aspect ratio of the new window
    if (height == 0) height = 1;                // To prevent divide by 0
    GLfloat aspect = (GLfloat)width / (GLfloat)height;

    // Set the viewport to cover the new window
    glViewport(0, 0, width, height);

    // Set the aspect ratio of the clipping area to match the viewport
    glMatrixMode(GL_PROJECTION);  // To operate on the Projection matrix
    glLoadIdentity();             // Reset the projection matrix
    if (width >= height) {
        // aspect >= 1, set the height from -1 to 1, with larger width
        gluOrtho2D(-1.0 * aspect, 1.0 * aspect, -1.0, 1.0);
    } else {
        // aspect < 1, set the width to -1 to 1, with larger height
        gluOrtho2D(-1.0, 1.0, -1.0 / aspect, 1.0 / aspect);
    }
}

void golMapClear()
{
    memset(gol_map, 0, gol_map_width * gol_map_height * sizeof(char));
}

void golMapDump()
{
    for(int j=0; j<gol_map_width; ++j)
    {
        for(int i=0; i<gol_map_height; ++i)
        {
            printf("%c", golCellGet(j, i)?'.':' ');
        }
        printf("\n");
    }
}

void golMapGenerate()
{
    for(int j=0; j<gol_map_width; ++j)
    {
        for(int i=0; i<gol_map_height; ++i)
        {
            gol_tmap[i*gol_map_width+j] = golCellNext(j, i);
        }
    }
    memcpy(gol_map, gol_tmap, gol_map_width*gol_map_height*sizeof(char));
}

void golMapRandFill()
{
    unsigned seed = time(0);
    //#pragma omp parallel firstprivate(seed)
    {
        srand(seed);
        //seed = omp_get_thread_num();
        //#pragma omp for collapse(2)
        for(int j=0; j<gol_map_width; ++j)
        {
            for(int i=0; i<gol_map_height; ++i)
            {
                gol_map[i*gol_map_width+j] = (rand() & 0x1);
            }
        }
    }
}

inline char golCellNext(int x, int y)
{
    int cell_count = 0;
    if(golCellGet(x-1, y-1)) ++cell_count;
    if(golCellGet(x-1, y  )) ++cell_count;
    if(golCellGet(x-1, y+1)) ++cell_count;
    if(golCellGet(x  , y-1)) ++cell_count;
    if(golCellGet(x  , y+1)) ++cell_count;
    if(golCellGet(x+1, y-1)) ++cell_count;
    if(golCellGet(x+1, y  )) ++cell_count;
    if(golCellGet(x+1, y+1)) ++cell_count;

    if(golCellGet(x, y) == 1 && (cell_count > 3 || cell_count < 2))
        return 0;
    else if(golCellGet(x, y) == 0 && cell_count == 3)
        return 1;
    else
        return golCellGet(x, y);
}

inline char golCellGet(int x, int y)
{
    if(x < 0 || y < 0 || x > gol_map_width || y > gol_map_height){
        return false;
    }
    else {
        return gol_map[y * gol_map_width + x];
    }
}

inline void golCellSet(int x, int y, char value)
{
    if(x < 0 || y < 0 || x > gol_map_width || y > gol_map_height){
        return;
    }
    else {
        gol_map[y * gol_map_width + x] = value;
    }
}

inline void golCellDraw(int x, int y)
{
    float posX = -1.0 + 2.0 * x / gol_map_width;
    float posY = -1.0 + 2.0 * y / gol_map_height;
    float posX2 = posX + 2.0 / gol_map_width;
    float posY2 = posY + 2.0 / gol_map_height;
    glColor3f(1.0f, 1.0f, 0.0f); // Green
    glRectf(posX, posY, posX2, posY2);
}

