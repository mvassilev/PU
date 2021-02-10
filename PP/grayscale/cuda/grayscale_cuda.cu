// nvcc grayscale_cuda.cu -o grayscale_cuda

#include <stdio.h>
#include <stdlib.h>

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
__global__ void grayscale(int n, PPMPixel *c) {
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  if (id < n) {
    double f = 1;
    double l = 0.3 * c[id].red + 0.6 * c[id].green + 0.1 * c[id].blue;
    c[id].red = c[id].red + f * (l - c[id].red);
    c[id].green = c[id].green + f * (l - c[id].green);
    c[id].blue = c[id].blue + f * (l - c[id].blue);
  }
}

void changeColorPPM(PPMImage *img) {
     int N = img->x * img->y;
     PPMPixel *x, *d_x;
     x = (PPMPixel*)malloc(N*sizeof(PPMPixel));

     // заделяне на памет в устройството
     cudaMalloc(&d_x, N*sizeof(PPMPixel)); 

     // копиране на данните за пикселите в x, която ще се предаде на устройството
     for (int i = 0; i < N; i++) {
          x[i] = img->data[i];
     }

     // изпращане на данните в устройството
     cudaMemcpy(d_x, x, N*sizeof(PPMPixel), cudaMemcpyHostToDevice);

     // изпълнение на изчисленията посредством 256 нишки
     grayscale<<<(N+255)/256, 256>>>(N, d_x);

     // копиране на данните от устройството обратно в хоста
     cudaMemcpy(x, d_x, N*sizeof(PPMPixel), cudaMemcpyDeviceToHost);

     // освобождаване на данни
     cudaFree(d_x);
     free(x);
}

int main(void) {
  PPMImage *image;
  image = readPPM("image.ppm");
  changeColorPPM(image);
  writePPM("grayscale_cuda_result.ppm", image);
}
