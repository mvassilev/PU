#include <stdio.h>
#include <omp.h>

static long num_steps = 100000000;
double step;

int main() {
    int i, j, total_num_threads, num_threads_allocated;
    double x, pi, sum = 0.0;
    double start_time, run_time;

    step = 1.0 / (double)num_steps;
    printf("Num threads available: %d\n", omp_get_max_threads());
    for (i = 1; i <= 4; i++) {
        sum = 0.0;
        omp_set_num_threads(i);
        start_time = omp_get_wtime();
#pragma omp parallel
        {
            num_threads_allocated = omp_get_num_threads();
#pragma omp single
            printf("Num threads allocated for this run: %d\n", num_threads_allocated);

#pragma omp for reduction(+ : sum)
            for (j = 1; j <= num_steps; j++) {
                x = (j - 0.5) * step;
                sum = sum + 4.0 / (1.0 + x * x);
            }
        }

        pi = step * sum;
        run_time = omp_get_wtime() - start_time;
        printf("pi is %f in %f seconds using %d threads\n\n", pi, run_time, num_threads_allocated);
    }
}