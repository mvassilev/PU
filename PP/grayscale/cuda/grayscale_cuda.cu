// nvcc grayscale_cuda.cu -o grayscale_cuda

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

typedef struct {
     unsigned char red, green, blue;
} PPMPixel;

typedef struct {
     int x, y;
     PPMPixel *data;
} PPMImage;

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

// kernel функция
__global__ void grayscale(int n, PPMPixel *x, PPMPixel *y) {
     int id = blockIdx.x*blockDim.x + threadIdx.x;
     if (id < n) {
          double f = 1;
          double l = 0.3 * x[id].red + 0.6 * x[id].green + 0.1 * x[id].blue;
          y[id].red = x[id].red + f * (l - x[id].red);
          y[id].green = x[id].green + f * (l - x[id].green);
          y[id].blue = x[id].blue + f * (l - x[id].blue);
     }
}

void changeColorPPM(PPMImage *img) {
     cudaError_t error;
     struct timeval tval_before, tval_after, tval_result;

     int N = img->x * img->y;
     PPMPixel *x, *d_x, *y, *d_y;
     x = (PPMPixel*)malloc(N*sizeof(PPMPixel));
     y = (PPMPixel*)malloc(N*sizeof(PPMPixel));

     cudaMalloc(&d_x, N*sizeof(PPMPixel));
     cudaMalloc(&d_y, N*sizeof(PPMPixel));

     for (int i = 0; i < N; i++) {
          x[i] = img->data[i];
          y[i] = img->data[i];
     }

     gettimeofday(&tval_before, NULL);
     cudaMemcpy(d_x, x, N*sizeof(PPMPixel), cudaMemcpyHostToDevice);
     cudaMemcpy(d_y, y, N*sizeof(PPMPixel), cudaMemcpyHostToDevice);
     gettimeofday(&tval_after, NULL);
     // Измерване колко време отнема копирането на данните в устройството
     timersub(&tval_after, &tval_before, &tval_result);
     printf("%ld.%06ld секунди   за копиране на данните в устройството\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
     
     // dim3 blockSize(4, 16, 1);
     // size_t gridCols = (img->x + blockSize.x - 1) / blockSize.x;
     // size_t gridRows = (img->y + blockSize.y - 1) / blockSize.y;

     // dim3 gridSize(gridCols, gridRows);
     // in order to workL otherwise it fails silently
     // grayscale<<<(N+1)/1, 1>>>(N, d_x, d_y);
     gettimeofday(&tval_before, NULL);
     grayscale<<<(N+383)/384, 384>>>(N, d_x, d_y);
     gettimeofday(&tval_after, NULL);
     // Измерване колко време отнема изпънението на kernel функцията
     timersub(&tval_after, &tval_before, &tval_result);
     printf("%ld.%06ld   секунди за ззпълнението на kernel функцията\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
     printf("error: %s\n", cudaGetErrorString(cudaGetLastError()));

     gettimeofday(&tval_before, NULL);
     cudaMemcpy(y, d_y, N*sizeof(PPMPixel), cudaMemcpyDeviceToHost);
     gettimeofday(&tval_after, NULL);
     // Измерване колко време отнема копирането на данните в хоста
     timersub(&tval_after, &tval_before, &tval_result);
     printf("%ld.%06ld\n   секунди за копиране на данните в хоста", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
     printf("error: %s\n", cudaGetErrorString(cudaGetLastError()));

     float sum = 0.0f;
     for (int i = 0; i < N; i++) {
          // printf("y[i].red: %d\n", y[i].red);
          // printf("y[i].green: %d\n", y[i].green);
          // printf("y[i].blue: %d\n", y[i].blue);
          img->data[i] = y[i];
     }
     printf("error: %s\n", cudaGetErrorString(cudaGetLastError()));

     cudaFree(d_x);
     cudaFree(d_y);
     free(x);
     free(y);
     // PPMPixel *x, *d_x;
     // x = (PPMPixel*)malloc(N*sizeof(PPMPixel));

     // // заделяне на памет в устройството
     // cudaMalloc(&d_x, N*sizeof(PPMPixel)); 

     // // копиране на данните за пикселите в x, която ще се предаде на устройството
     // for (int i = 0; i < N; i++) {
     //      x[i].red = img->data[i].red;
     //      x[i].green = img->data[i].green;
     //      x[i].blue = img->data[i].blue;
     // }

     // // изпращане на данните в устройството
     // cudaMemcpy(d_x, x, N*sizeof(PPMPixel), cudaMemcpyHostToDevice);

     // // изпълнение на изчисленията посредством 256 нишки
     // grayscale<<<(N+255)/256, 256>>>(N, d_x);

     // // копиране на данните от устройството обратно в хоста
     // cudaMemcpy(x, d_x, N*sizeof(PPMPixel), cudaMemcpyDeviceToHost);

     // for (int i = 0; i < N; i++) {
     //      img->data[i].red = x[i].red;
     //      img->data[i].green = x[i].green;
     //      img->data[i].blue = x[i].blue;
     // }

     // // освобождаване на данни
     // cudaFree(d_x);
     // free(x);
}

int main(void) {
  PPMImage *image;
  image = readPPM("image.ppm");
  changeColorPPM(image);
  writePPM("grayscale_cuda_result.ppm", image);
}
