# include <math.h>
# include <stdlib.h>
# include <stdio.h>
# include <omp.h>

int main ( int argc, char *argv[] );
double f ( double x );
double cpu_time ( );

/******************************************************************************/

int main ( int argc, char *argv[] )

/******************************************************************************/
/*
  Purpose:

    MAIN is the main program for QUAD.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    14 December 2011

  Author:

    John Burkardt
*/
{
  double a = 0.0;
  double b = 10.0;
  double error;
  double exact = 0.49936338107645674464;
  int i;
  int n = 10000000;
  double total;
  double wtime;
  double x;

  printf ( "\n" );
  printf ( "QUAD:\n" );
  printf ( "  C version\n" );
  printf ( "  Use OpenMP for parallel execution.\n" );
  printf ( "  Estimate the integral of f(x) from A to B.\n" );
  printf ( "  f(x) = 50 / ( pi * ( 2500 * x * x + 1 ) ).\n" );
  printf ( "\n" );
  printf ( "  A        = %f\n", a );
  printf ( "  B        = %f\n", b );
  printf ( "  N        = %d\n", n );
  printf ( "  Exact    = %24.16f\n", exact );

  wtime = omp_get_wtime ( );

  total = 0.0;
/*
  Begin the parallel region.
*/
# pragma omp parallel shared ( a, b, n ) private ( i, x ) 
{
/*
  Begin a parallel loop.
*/
# pragma omp for reduction ( + : total )

  for ( i = 0; i < n; i++ )
  {
    x = ( ( double ) ( n - 1 - i ) * a 
        + ( double ) (         i ) * b )
        / ( double ) ( n - 1     );
    total = total + f ( x );
  }
/*
  End the parallel region.
*/
}
  wtime = omp_get_wtime ( ) - wtime;

  total = ( b - a ) * total / ( double ) n;
  error = fabs ( total - exact );

  printf ( "\n" );
  printf ( "  Estimate = %24.16f\n", total );
  printf ( "  Error    = %e\n", error );
  printf ( "  Time     = %f\n", wtime );
/*
  Terminate.
*/
  printf ( "\n" );
  printf ( "QUAD:\n" );
  printf ( "  Normal end of execution.\n" );
  printf ( "\n" );

  return 0;
}
/*******************************************************************************/

double f ( double x )

/*******************************************************************************/
/*
  Purpose:
 
    F evaluates the function.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    18 July 2010

  Author:

    John Burkardt

  Parameters:

    Input, double X, the argument.

    Output, double F, the value of the function.
*/
{
  double r8_pi = 3.141592653589793;
  double value;

  value = 50.0 / ( r8_pi * ( 2500.0 * x * x + 1.0 ) );

  return value;
}
/*******************************************************************************/

double cpu_time ( )

/*******************************************************************************/
/*
  Purpose:
 
    CPU_TIME reports the total CPU time for a program.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    27 September 2005

  Author:

    John Burkardt

  Parameters:

    Output, double CPU_TIME, the current total elapsed CPU time in second.
*/
{
  double value;

  value = ( double ) clock ( ) / ( double ) CLOCKS_PER_SEC;

  return value;
}

