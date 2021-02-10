// gcc -x c -g -framework OpenCL grayscale_opencl.c -o grayscale_opencl

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <OpenCL/cl.h>
#include "grayscale.h"

#define CREATOR "ParallelProgrammer"
#define RGB_COMPONENT_COLOR 255

static PPMImage *readPPM(const char *filename) {
     char buff[16];
     PPMImage *img;
     FILE *fp;
     int c, rgb_comp_color;

     // зареждане на PPM файла за четене
     fp = fopen(filename, "rb");
     if (!fp) {
          fprintf(stderr, "Unable to open file '%s'\n", filename);
          exit(1);
     }

     // четене на метаданните от него 
     if (!fgets(buff, sizeof(buff), fp)) {
          perror(filename);
          exit(1);
     }

     // проверка на метаданните
     if (buff[0] != 'P' || buff[1] != '6') {
          fprintf(stderr, "Invalid image format (must be 'P6')\n");
          exit(1);
     }

     // заделяне на памет
     img = (PPMImage *)malloc(sizeof(PPMImage));
     if (!img) {
          fprintf(stderr, "Unable to allocate memory\n");
          exit(1);
     }

     // проверка за коментари вътре в самото изображение
     c = getc(fp);
     while (c == '#') {
          while (getc(fp) != '\n') ;
          c = getc(fp);
     }

     ungetc(c, fp);
     // проверка на данните за размера на изображението
     if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
          fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
          exit(1);
     }

     // проверка на RGB компонента
     if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
          fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
          exit(1);
     }

     // проверка на размерността на RGB компонента
     if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
          fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
          exit(1);
     }

     while (fgetc(fp) != '\n') ;
     // заделяне на памет за информацията във всеки пиксел
     img->data = (PPMPixel*)malloc(img->x * img->y * sizeof(PPMPixel));

     if (!img) {
          fprintf(stderr, "Unable to allocate memory\n");
          exit(1);
     }

     // зареждане на данните за всеки пиксел
     if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
          fprintf(stderr, "Error loading image '%s'\n", filename);
          exit(1);
     }

     fclose(fp);
     return img;
}
void writePPM(const char *filename, PPMImage *img) {
     FILE *fp;
     // отваряне на файл в режим за писане
     fp = fopen(filename, "wb");
     if (!fp) {
          fprintf(stderr, "Unable to open file '%s'\n", filename);
          exit(1);
     }

     // записване на метаданни за типа на PPM изображението
     fprintf(fp, "P6\n");

     // запис на коментари
     fprintf(fp, "# Created by %s\n", CREATOR);

     // запис на размера на изображението
     fprintf(fp, "%d %d\n",img->x,img->y);

     // запис на размерността на RGB компонента
     fprintf(fp, "%d\n", RGB_COMPONENT_COLOR);

     // запис на данните за пикселите от изображението
     fwrite(img->data, 3 * img->x, img->y, fp);
     fclose(fp);
}

void changeColorPPM(PPMImage *img) {
     // размер на изображението
     unsigned int n = img->x * img->y;

     // резултатен вектор
     PPMPixel *host_result;

     // входни данни за устройството
     cl_mem device_pixel_data;
     // изходни данни от устройството
     cl_mem device_result;

     // OpenCL платформа
     cl_platform_id cpPlatform;

     // Масив с устройства
     cl_device_id device_id[2];
     cl_context context;

     // Опашката на централния процесор
     cl_command_queue cpu_queue;

     // Опашката на графичния ускорител
     cl_command_queue gpu_queue;

     cl_program program;
     cl_kernel kernel;

     // определяне на размера на паметта
     size_t bytes = n*sizeof(PPMPixel);

     // Заделяне на памет в хоста
     host_result = (PPMPixel*)malloc(bytes);
 
     size_t globalSize, localSize;
     cl_int err;
     
     // размерност на локална група
     localSize = 64;
     
     // брой локални групи
     globalSize = ceil(n/(float)localSize)*localSize;
     
     // списък на наличните платформи
     err = clGetPlatformIDs(1, &cpPlatform, NULL);
     
     // брой налични устройства
     cl_uint num_devices_returned;

     // задаване на 
     err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id[0], &num_devices_returned);
     err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, 1, &device_id[1], &num_devices_returned);
     
     // създаване на контекст  
     context = clCreateContext(0, 2, device_id, NULL, NULL, &err);
     
     // създаване на опашки 
     gpu_queue = clCreateCommandQueue(context, device_id[0], 0, &err);
     cpu_queue = clCreateCommandQueue(context, device_id[1], 0, &err);
     
     // четене на kernel функцията от външен файл
     FILE *program_handle;
     char *program_buffer, *program_log;
     size_t program_size, log_size;
     program_handle = fopen("grayscale_kernel.cl", "r");
     if (program_handle == NULL) {
          perror("Couldn't find the program file");
          exit(1);
     }

     fseek(program_handle, 0, SEEK_END);
     program_size = ftell(program_handle);
     rewind(program_handle);
     program_buffer = (char*)malloc(program_size + 1);
     program_buffer[program_size] = '\0';
     fread(program_buffer, sizeof(char), program_size, program_handle);
     fclose(program_handle);

     // създаване на програма на базата на прочетената kernel функция
     program = clCreateProgramWithSource(context, 1, (const char **) & program_buffer, NULL, &err);
     
     // компилиране на kernel функцията
     err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
     if (err) {
          char log[10240] = "";
          err = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
          printf("Program build log:\n%s\n", &log);
     }
     
     // създаване на изчисляващата kernel функция в програмата, която ще се изпълни
     kernel = clCreateKernel(program, "grayscale", &err);
     
     // създаване на входните и изходните масиви в устройството за изчисленията
     device_pixel_data = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
     device_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
     
     // запис на данните във входния масив в изчисляващото устройство
     err = clEnqueueWriteBuffer(gpu_queue, device_pixel_data, CL_TRUE, 0, bytes, img->data, 0, NULL, NULL);

     // задаване на аргументите на kernel функцията
     err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_pixel_data);
     err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_result);
     err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &n);
     
     // изпълнение на kernel функцията върху входните данните 
     err = clEnqueueNDRangeKernel(gpu_queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
     
     // изчакаване за завършване на обработката на опашката
     clFinish(gpu_queue);
     
     // четене на резултата от устройството
     clEnqueueReadBuffer(gpu_queue, device_result, CL_TRUE, 0, bytes, host_result, 0, NULL, NULL );
     
     int i;
     for (i = 0; i < n; i++)
          img->data[i] = host_result[i];
 
    // освобождаване на OpenCL resources
    clReleaseMemObject(device_pixel_data);
    clReleaseMemObject(device_result);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(gpu_queue);
    clReleaseContext(context);
 
    // освобождаване на хост ресурси
    free(host_result);
}

void printSystemInfo() {
     char buffer[10240];
     cl_platform_id platforms[100];
     cl_device_id device_id; 
	cl_uint platforms_n = 0;
     clGetPlatformIDs(100, platforms, &platforms_n);
     clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
     printf("  DEVICE_NAME = %s\n", buffer);
     cl_device_id devices[100];
	cl_uint devices_n = 0;
	// CL_CHECK(clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 100, devices, &devices_n));
	clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 100, devices, &devices_n);

	printf("=== %d OpenCL device(s) found on platform:\n", platforms_n);
	for (int i=0; i<devices_n; i++){
		char buffer[10240];
		cl_uint buf_uint;
		cl_ulong buf_ulong;
		printf("  -- %d --\n", i);
		clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
		printf("  DEVICE_NAME = %s\n", buffer);
		clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL);
		printf("  DEVICE_VENDOR = %s\n", buffer);
		clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL);
		printf("  DEVICE_VERSION = %s\n", buffer);
		clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL);
		printf("  DRIVER_VERSION = %s\n", buffer);
		clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL);
		printf("  DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)buf_uint);
		clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL);
		printf("  DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int)buf_uint);
		clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
		printf("  DEVICE_GLOBAL_MEM_SIZE = %llu\n", (unsigned long long)buf_ulong);
	}
}

int main() {
     printSystemInfo();
     PPMImage *image;
     image = readPPM("image.ppm");
     changeColorPPM(image);
     writePPM("grayscale_opencl_result.ppm", image);
}

