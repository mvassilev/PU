// gcc array.c -o array

#include <stdlib.h>
#include <stdio.h>

int main () {            
    int limit = 1024;
    int *A = (int *)malloc(sizeof(int) * limit);
    int *B = (int *)malloc(sizeof(int) * limit);
    for (int i = 0; i < limit ; i++){
        A[i] = i;
    }

    for (int i = 0; i < limit ; i++) {
        B[i] = A[i] * A[i];
    }

    for (int i = 0; i < limit ; i++) {
        printf("%3d, " , B[i]);
    }

    printf("\n");
}