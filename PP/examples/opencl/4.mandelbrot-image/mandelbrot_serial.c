// gcc mandelbrot_serial.c -o mandelbrot_serial

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Computes the Mandelbrot Set to N Iterations
void solve_mandelbrot(float real[], 
                      float imag[],
                      int iterations, int nreals,
                      int *result)
{
   int i, n;
   float x,y, xtemp;

    for(i = 0; i < nreals; i++)
    {
        x = real[i]; // Real Component
        y = imag[i]; // Imaginary Component
        n = 0;       // Tracks Color Information

        // Compute the Mandelbrot Set
        while ((x * x + y * y <= 2 * 2) && n < iterations)
        {
            xtemp = x * x - y * y + real[i];
            y = 2 * x * y + imag[i];
            x = xtemp;
            n++;
        }

        // Write Results to Output Arrays
        result[i] = x * x + y * y <= 2 * 2 ? -1 : n;
    }
}



int main() {

   // Define Mandelbrot Settings
   int iterations = 2000;
   float x_min  = -2;
   float x_max  =  2;
   float y_min  = -1.5f;
   float y_max  =  1.5f;
   float x_step = .002f;
   float y_step = .002f;

   // Create Linear Vector of Coordinates
   int nreals,nimags;
   int i;
   float *reals,*imags; // Host input arrays
   int *ans; // Host output array
   nimags=(y_max-y_min)/y_step;
   nreals=(x_max-x_min)/x_step;
   reals = (float *)malloc(sizeof(float)*nreals); 
   imags = (float *)malloc(sizeof(float)*nimags); 
   ans = (int *)malloc(sizeof(int)*nreals); 

   for (i=0; i<nreals; i++) {
      reals[i]=reals[i]+i*x_step;
   }

   for (i=0; i<nimags; i++) {
      imags[i]=imags[i]+i*y_step;
   } 

   struct timeval tval_before, tval_after, tval_result;
   gettimeofday(&tval_before, NULL);
   solve_mandelbrot(reals, imags, iterations, nreals, ans);
   gettimeofday(&tval_after, NULL);
   timersub(&tval_after, &tval_before, &tval_result);
   printf("%ld.%06ld     секунди за изпълнение\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);

   for (int i = 0; i < nreals ; i++) {
        printf("%3d, " , ans[i]);
   }

   return 0;
}