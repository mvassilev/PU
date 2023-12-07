// g++ -g -framework OpenCL mandelbrot_parallel_cl.cpp -o mandelbrot_parallel_cl

#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define PROGRAM_FILE "mandelbrot_parallel_cl.cl"
#define KERNEL_FUNC "solve_mandelbrot"
#define WG_SIZE 256 // Workgroup size

// Проверка за грешки от OpenCL
#define CL_CHECK(_expr)                                                        \
   do                                                                          \
   {                                                                           \
      cl_int _err = _expr;                                                     \
      if (_err == CL_SUCCESS)                                                  \
         break;                                                                \
      fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
      exit(1); /* abort(); */                                                  \
   } while (0)

#define CL_CHECK_ERR(_expr)                                                       \
   ({                                                                             \
      cl_int _err = CL_INVALID_VALUE;                                             \
      typeof(_expr) _ret = _expr;                                                 \
      if (_err != CL_SUCCESS)                                                     \
      {                                                                           \
         fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
         exit(1); /* abort(); */                                                  \
      }                                                                           \
      _ret;                                                                       \
   })

// Computes the Mandelbrot Set to N Iterations
void solve_mandelbrot(std::vector<float> const &real, std::vector<float> const &imag, int iterations, std::vector<int> &result)
{
   for (unsigned int i = 0; i < real.size(); i++)
   {
      float x = real[i]; // Real Component
      float y = imag[i]; // Imaginary Component
      int n = 0;         // Tracks Color Information

      // Compute the Mandelbrot Set
      while ((x * x + y * y <= 2 * 2) && n < iterations)
      {
         float xtemp = x * x - y * y + real[i];
         y = 2 * x * y + imag[i];
         x = xtemp;
         n++;
      }

      // Write Results to Output Arrays
      result[i] = x * x + y * y <= 2 * 2 ? -1 : n;
   }
}

// Draws a Portable Pixel Map (PPM/PBM) of the Given Mandelbrot Set
void ppm_draw(std::string const &filename, std::vector<std::vector<int> > const &grid)
{
   // Declare Local Variables
   std::ofstream fd(filename);
   unsigned int iterations = 50;

   // Write the PPM Header
   fd << "P6"
      << " " << grid[0].size()
      << " " << grid.size()
      << " " << 255
      << std::endl;

   // Determine Pixel Color at Index
   for (auto &i : grid)
      for (auto &j : i)
      {
         unsigned char r, g, b;
         if (j == -1)
         {
            r = 0;
            g = 0;
            b = 0;
         }
         else
         {
            r = (j * 255 / iterations);
            g = r;
            b = 255;
         }

         // Write Colors to File
         fd << r << g << b;
      }
}

// Четене на сорс кода на kernel функцията от външен файл
char *ReadKernelProgram()
{
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   program_handle = fopen(PROGRAM_FILE, "r");
   if (program_handle == NULL)
   {
      perror("Couldn't find the program file");
      exit(1);
   }

   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char *)malloc(program_size + 1);
   program_buffer[program_size] = '\0';
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);

   return program_buffer;
}

cl_device_id create_device()
{
   cl_platform_id platform;
   cl_device_id dev;
   int err;

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if (err < 0)
   {
      perror("Couldn't identify a platform");
      exit(1);
   }

   // Access a device
   // GPU
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if (err == CL_DEVICE_NOT_FOUND)
   {
      // CPU
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
   }
   if (err < 0)
   {
      perror("Couldn't access any devices");
      exit(1);
   }

   return dev;
}

cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename)
{

   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   int err;

   /* Read program file and place content into buffer */
   program_handle = fopen(filename, "r");
   if (program_handle == NULL)
   {
      perror("Couldn't find the program file");
      exit(1);
   }
   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char *)malloc(program_size + 1);
   program_buffer[program_size] = '\0';
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);

   /* Create program from file
   Creates a program from the source code in the add_numbers.cl file.
   Specifically, the code reads the file's content into a char array
   called program_buffer, and then calls clCreateProgramWithSource.
   */
   program = clCreateProgramWithSource(ctx, 1,
                                       (const char **)&program_buffer, &program_size, &err);
   if (err < 0)
   {
      perror("Couldn't create the program");
      exit(1);
   }
   free(program_buffer);

   /* Build program
   The fourth parameter accepts options that configure the compilation.
   These are similar to the flags used by gcc. For example, you can
   define a macro with the option -DMACRO=VALUE and turn off optimization
   with -cl-opt-disable.
   */
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if (err < 0)
   {

      /* Find size of log and print to std output */
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                            0, NULL, &log_size);
      program_log = (char *)malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                            log_size + 1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }

   return program;
}

int main()
{
   /* OpenCL structures */
   cl_device_id device;
   cl_context context;
   cl_program program;
   cl_kernel kernel;
   cl_command_queue queue;
   cl_int i, j, err;
   size_t local_size, global_size;

   /* Data and buffers    */
   // Define Mandelbrot Settings
   int iterations = 2000;
   float x_min = -2;
   float x_max = 2;
   float y_min = -1;
   float y_max = 1;
   float x_step = .002f;
   float y_step = .002f;
   float x, y;

   // Create Linear Vector of Coordinates
   int nreals, nimags;
   // float *reals, *imags; // Host input arrays
   // int *ans;             // Host output array
   nimags = (y_max - y_min) / y_step;
   nreals = (x_max - x_min) / x_step;
   // reals = (float *)malloc(sizeof(float) * nreals * nimags);
   // imags = (float *)malloc(sizeof(float) * nimags * nreals);
   // ans = (int *)malloc(sizeof(int) * nreals);

   // for (i=0,j=x_min; i<nreals*nimags; i++, j+=x_step) {
   //    reals[i]=j;
   // }

   // for (i=0,j=y_min; i<nimags*nreals; i++,j+=y_step) {
   //    imags[i]=j;
   // }
   // long k = 0;
   // for(float y = y_min; y < y_max; y += y_step) {
   //      for(float x = x_min; x < x_max; x += x_step) {
   //          reals[k] = x;
   //          imags[k] = y;
   //          k++;
   //      }
   //  }

   // Create Linear Vector of Coordinates
   std::vector<float> *reals = new std::vector<float>();
   std::vector<float> *imags = new std::vector<float>();
   for (float y = y_min; y < y_max; y += y_step)
   {
      for (float x = x_min; x < x_max; x += x_step)
      {
         reals->push_back(x);
         imags->push_back(y);
      }
   }

   std::vector<int> *ans = new std::vector<int>(reals->size());
   nreals = reals->size();
   nimags = imags->size();

   // Device input and output buffers
   cl_mem dreals, dimags, dans;

   /* Create device and context   */
   device = create_device();
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

   /* Build program */
   program = build_program(context, device, PROGRAM_FILE);

   /* Create a command queue */
   queue = clCreateCommandQueue(context, device, 0, &err);

   /* Create data buffer
   Create the input and output arrays in device memory for our
   calculation. 'd' below stands for 'device'.
   */
   dreals = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * reals->size(), NULL, NULL);
   dimags = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * imags->size(), NULL, NULL);
   dans = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * reals->size(), NULL, NULL);

   // Write our data set into the input array in device memory
   err = clEnqueueWriteBuffer(queue, dreals, CL_TRUE, 0, sizeof(float) * reals->size(), reals, 0, NULL, NULL);
   err |= clEnqueueWriteBuffer(queue, dimags, CL_TRUE, 0, sizeof(float) * imags->size(), imags, 0, NULL, NULL);

   CL_CHECK(err);

   /* Create a kernel */
   kernel = clCreateKernel(program, KERNEL_FUNC, &err);

   /* Create kernel arguments    */
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dreals);
   err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dimags);
   err |= clSetKernelArg(kernel, 2, sizeof(int), &iterations);
   err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &dans);

   CL_CHECK(err);

   // Number of work items in each local work group
   local_size = WG_SIZE;
   // Number of total work items - localSize must be devisor
   global_size = ceil(nreals / (float)local_size) * local_size;
   // size_t global_size[3] = {ARRAY_SIZE, 0, 0}; // for 3D data
   // size_t local_size[3] = {WG_SIZE, 0, 0};

   /* Enqueue kernel    */
   err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

   // CL_CHECK(err);

   /* Wait for the command queue to get serviced before reading
   back results */
   clFinish(queue);

   /* Read the kernel's output    */
   clEnqueueReadBuffer(queue, dans, CL_TRUE, 0, sizeof(int) * nreals, ans, 0, NULL, NULL); // <=====GET OUTPUT

   // Compute Stride to Avoid Floating Point Errors
   unsigned int stride = 0;
   for (float x = x_min; x < x_max; x += x_step)
      stride++;

   // Reshape the Output Array to 2D for Drawing
   std::vector<std::vector<int> > ans_grid;
   for (unsigned int i = 0; i < reals->size() / stride; i++) {
      ans_grid.push_back(std::vector<int>(0 + (i * stride), 0 + (i + 1) * stride));
      printf("i: %d\n", i);
      printf("reals->size() / stride): %lu\n", reals->size() / stride);
   }

   printf("done");

   // Print Statistics
   // std::cout << "CPU Time: " << (float) (cpu_end - cpu_begin) / CLOCKS_PER_SEC << "s" << std::endl;

   // Write the Mandelbrot Set to File
   // ppm_draw("mandelbrot_parallel_cl.ppm", ans_grid);

   /* Deallocate resources */
   clReleaseKernel(kernel);
   clReleaseMemObject(dreals);
   clReleaseMemObject(dimags);
   clReleaseMemObject(dans);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}