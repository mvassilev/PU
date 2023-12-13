#include <stdio.h>
#include <omp.h>

int main() {
    int max_threads = omp_get_max_threads();

    printf("max threads: %d\n", max_threads);
    omp_set_num_threads(max_threads);

#pragma omp parallel
    {
        int id = omp_get_thread_num();
        printf("Hello World from thread = %d with %d threads\n", id, omp_get_num_threads());
    }

    printf("all done, with hopefully %d threads\n", max_threads);
}