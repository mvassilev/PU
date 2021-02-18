// gcc -x c -g -framework OpenCL grayscale_opencl.c -o grayscale_opencl
// gcc -x c -g grayscale_opencl.c -o grayscale_opencl -lOpenCL -lm
// clinfo

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifdef __APPLE__
     #include <OpenCL/cl.h>
#else
     #include <CL/cl.h>
#endif
#include "grayscale.h"

#define CREATOR "ParallelProgrammer"
#define RGB_COMPONENT_COLOR 255

// Проверка за грешки от OpenCL
#define CL_CHECK(_expr)                                                         \
  do {                                                                          \
    cl_int _err = _expr;                                                        \
    if (_err == CL_SUCCESS)                                                     \
      break;                                                                    \
    fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);    \
    exit(1); /* abort(); */                                                     \
  } while (0)

#define CL_CHECK_ERR(_expr)							\
  ({										\
     cl_int _err = CL_INVALID_VALUE;						\
     typeof(_expr) _ret = _expr;						\
     if (_err != CL_SUCCESS) {                                                  \
       fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
       exit(1); /* abort(); */                                                  \
     }                                                                          \
     _ret;                                                                      \
  })

// Четене на изображение от външен файл
static PPMImage *ReadPPM(const char *filename) {
     char buff[16];
     PPMImage *img;
     FILE *fp;
     int c, rgb_comp_color;

     // Зареждане на PPM файла за четене
     fp = fopen(filename, "rb");
     if (!fp) {
          fprintf(stderr, "Unable to open file '%s'\n", filename);
          exit(1);
     }

     // Четене на метаданните от него 
     if (!fgets(buff, sizeof(buff), fp)) {
          perror(filename);
          exit(1);
     }

     // Проверка на метаданните
     if (buff[0] != 'P' || buff[1] != '6') {
          fprintf(stderr, "Invalid image format (must be 'P6')\n");
          exit(1);
     }

     // Заделяне на памет
     img = (PPMImage *)malloc(sizeof(PPMImage));
     if (!img) {
          fprintf(stderr, "Unable to allocate memory\n");
          exit(1);
     }

     // Проверка за коментари вътре в самото изображение
     c = getc(fp);
     while (c == '#') {
          while (getc(fp) != '\n') ;
          c = getc(fp);
     }

     ungetc(c, fp);
     // Проверка на данните за размера на изображението
     if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
          fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
          exit(1);
     }

     // Проверка на RGB компонента
     if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
          fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
          exit(1);
     }

     // Проверка на размерността на RGB компонента
     if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
          fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
          exit(1);
     }

     while (fgetc(fp) != '\n') ;
     // Заделяне на памет за информацията във всеки пиксел
     img->data = (PPMPixel*)malloc(img->x * img->y * sizeof(PPMPixel));

     if (!img) {
          fprintf(stderr, "Unable to allocate memory\n");
          exit(1);
     }

     // Зареждане на данните за всеки пиксел
     if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
          fprintf(stderr, "Error loading image '%s'\n", filename);
          exit(1);
     }

     fclose(fp);
     return img;
}

// Запис на изображение във външен файл
void WritePPM(const char *filename, PPMImage *img) {
     FILE *fp;
     // Отваряне на файл в режим за писане
     fp = fopen(filename, "wb");
     if (!fp) {
          fprintf(stderr, "Unable to open file '%s'\n", filename);
          exit(1);
     }

     // Записване на метаданни за типа на PPM изображението
     fprintf(fp, "P6\n");

     // Запис на коментари
     fprintf(fp, "# Created by %s\n", CREATOR);

     // запис на размера на изображението
     fprintf(fp, "%d %d\n",img->x,img->y);

     // Запис на размерността на RGB компонента
     fprintf(fp, "%d\n", RGB_COMPONENT_COLOR);

     // Запис на данните за пикселите от изображението
     fwrite(img->data, 3 * img->x, img->y, fp);
     fclose(fp);
}

// Четене на сорс кода на kernel функцията от външен файл
char *ReadKernelProgram() {
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

     return program_buffer;
}

// Информация за платформата и устройствата
void PrintSystemInfo(cl_platform_id *platforms, cl_uint platforms_n) {
     char buffer[10240];

     cl_device_id devices[100];

	printf("=== %d OpenCL platform(s) found:\n", platforms_n);
     for (int i = 0; i < platforms_n; i++) {
          cl_uint devices_n = 0;
	     CL_CHECK(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 100, devices, &devices_n));

          for (int j = 0; j < devices_n; j++){
               char buffer[10240];
               cl_uint buf_uint;
               cl_ulong buf_ulong;
               printf("  -- %d --\n", i);
               clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
               printf("  DEVICE_NAME = %s\n", buffer);
               clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL);
               printf("  DEVICE_VENDOR = %s\n", buffer);
               clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL);
               printf("  DEVICE_VERSION = %s\n", buffer);
               clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL);
               printf("  DRIVER_VERSION = %s\n", buffer);
               clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL);
               printf("  DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)buf_uint);
               clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL);
               printf("  DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int)buf_uint);
               clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
               printf("  DEVICE_GLOBAL_MEM_SIZE = %llu\n", (unsigned long long)buf_ulong);
          }
     }
}

void ChangeColorPPM(PPMImage *img) {
     // Входни данни за устройството
     cl_mem device_pixel_data;

     // Изходни данни от устройството
     cl_mem device_result;

     // Брой на платформите
     cl_uint platforms_n = 0;

     // Списък на наличните платформи
     cl_platform_id platforms[100];
     clGetPlatformIDs(100, platforms, &platforms_n);

     // Извод на наличните платформи и устройства
     PrintSystemInfo(platforms, platforms_n);

     // Списък на наличните устройства
     cl_device_id devices[100];
     cl_context context;

     // Опашката на централния процесор
     cl_command_queue cpu_queue;

     // Опашката на графичния ускорител
     cl_command_queue queue;

     cl_program program;
     cl_kernel kernel;

     // Размер на изображението
     unsigned int n = img->x * img->y;

     // Определяне на размера на паметта
     size_t bytes = n*sizeof(PPMPixel);

     // Резултатен вектор
     PPMPixel *host_result;

     // Заделяне на памет в хоста
     host_result = (PPMPixel*)malloc(bytes);
 
     size_t globalSize, localSize;
     cl_int err;
     
     // Размерност на локална група
     localSize = 64;
     
     // Брой локални групи
     globalSize = ceil(n/(float)localSize)*localSize;
     
     // Брой налични устройства
     cl_uint num_devices_returned;

     // Извличане на наличните устройства в devices
     err = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_GPU, 100, devices, &num_devices_returned);
     
     // Създаване на контекст от първото устройство в devices
     context = clCreateContext(0, 1, &devices[0], NULL, NULL, &err);
     
     // Създаване на опашки 
     queue = clCreateCommandQueue(context, devices[0], 0, &err);
     
     // Четене на kernel функцията от външен файл
     char *program_buffer = ReadKernelProgram();

     // Създаване на програма на базата на прочетената kernel функция
     program = clCreateProgramWithSource(context, 1, (const char **) & program_buffer, NULL, &err);
     
     // Компилиране на kernel функцията
     err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
     if (err) {
          char log[10240] = "";
          err = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
          printf("Program build log:\n%s\n", log);
     }
     
     // Създаване на изчисляващата kernel функция в програмата, която ще се изпълни
     kernel = clCreateKernel(program, "grayscale", &err);
     
     // Създаване на входните и изходните масиви в устройството за изчисленията
     device_pixel_data = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
     device_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
     
     // Запис на данните във входния масив в изчисляващото устройство
     err = clEnqueueWriteBuffer(queue, device_pixel_data, CL_TRUE, 0, bytes, img->data, 0, NULL, NULL);

     // Задаване на аргументите на kernel функцията
     err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_pixel_data);
     err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_result);
     err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &n);
     
     // Изпълнение на kernel функцията върху входните данните 
     err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
     
     // Изчакаване за завършване на обработката на опашката
     clFinish(queue);
     
     // Четене на резултата от устройството
     clEnqueueReadBuffer(queue, device_result, CL_TRUE, 0, bytes, host_result, 0, NULL, NULL );
     
     int i;
     for (i = 0; i < n; i++)
          img->data[i] = host_result[i];
 
    // Освобождаване на OpenCL resources
    clReleaseMemObject(device_pixel_data);
    clReleaseMemObject(device_result);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
 
    // Освобождаване на хост ресурси
    free(host_result);
}

int main() {
     PPMImage *image;
     image = ReadPPM("image.ppm");
     ChangeColorPPM(image);
     WritePPM("grayscale_opencl_result.ppm", image);
}

