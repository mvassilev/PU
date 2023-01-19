// // Author: Wes Kendall
// // Copyright 2011 www.mpitutorial.com
// // This code is provided freely with the tutorials on mpitutorial.com. Feel
// // free to modify it for your own use. Any distribution of the code must
// // either provide a link to www.mpitutorial.com or keep this header intact.
// //
// // An intro MPI hello world program that uses MPI_Init, MPI_Comm_size,
// // MPI_Comm_rank, MPI_Finalize, and MPI_Get_processor_name.
// //
// #include <mpi.h>
// #include <stdio.h>

// int main(int argc, char** argv) {
//   // Initialize the MPI environment. The two arguments to MPI Init are not
//   // currently used by MPI implementations, but are there in case future
//   // implementations might need the arguments.
//   MPI_Init(NULL, NULL);

//   // Get the number of processes
//   int world_size;
//   MPI_Comm_size(MPI_COMM_WORLD, &world_size);

//   // Get the rank of the process
//   int world_rank;
//   MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

//   // Get the name of the processor
//   char processor_name[MPI_MAX_PROCESSOR_NAME];
//   int name_len;
//   MPI_Get_processor_name(processor_name, &name_len);

//   // Print off a hello world message
//   printf("---->Hello world from processor %s, rank %d out of %d processors\n",
//          processor_name, world_rank, world_size);

//   // Finalize the MPI environment. No more MPI calls can be made after this
//   MPI_Finalize();
// }

#include "mpi.h"
#include <math.h>

int main(argc,argv)
int argc;
char *argv[];
{
    int done = 0, n, myid, numprocs, i;
    double PI25DT = 3.141592653589793238462643;
    double mypi, pi, h, sum, x;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    while (!done)
    {
	if (myid == 0) {
	    printf("Enter the number of intervals: (0 quits) ");
	    scanf("%d",&n);
	}
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (n == 0) break;
  
	h   = 1.0 / (double) n;
	sum = 0.0;
	for (i = myid + 1; i <= n; i += numprocs) {
	    x = h * ((double)i - 0.5);
	    sum += 4.0 / (1.0 + x*x);
	}
	mypi = h * sum;
    
	MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0,
		   MPI_COMM_WORLD);
    
	if (myid == 0)
	    printf("pi is approximately %.16f, Error is %.16f\n",
		   pi, fabs(pi - PI25DT));
    }
    MPI_Finalize();
    return 0;
}