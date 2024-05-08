#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>


#define N 11


void printMatrix(int rank, int n, float *matrix)
{
    int i, j;
    printf("---------------%d-----------------\n", rank);
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < N; j++)
        {
            printf("%f ", matrix[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}


void printVector(int rank, int n, float v[])
{
    int i;
    printf("-------%d---------\n", rank);
    for (i = 0; i < n; i++)
    {
        printf("%f ", v[i]);
    }
    printf("\n\n");
}


void printTiempos(int rank, char *str, struct timeval tv1, struct timeval tv2)
{
    int microseconds = (tv2.tv_usec - tv1.tv_usec) + 1000000 * (tv2.tv_sec - tv1.tv_sec);
    printf("rank: %d -> %s = %lf\n", rank, str, (double)microseconds / 1E6);
}


int main(int argc, char *argv[])
{


    int i, j, rank, numprocs, filas, filas_indiv;
    float *matrix, *result, *matrixAux, *resultAux;
    int *sendcountS, *sendcountG, *displsS, *displsG;
    float vector[N];
    struct timeval tv1, tv2;


    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    filas = N / numprocs;
    filas_indiv = filas + (rank < N % numprocs ? 1 : 0);


    if ((matrixAux = malloc(sizeof(float) * filas_indiv * N)) == NULL)
        perror("error 1: ");
    if ((resultAux = malloc(sizeof(float) * filas_indiv)) == NULL)
        perror("error 2: ");


    /* Initialize Matrix and Vector */
    if (rank == 0)
    {
        if ((matrix = malloc(sizeof(float) * N * N)) == NULL)
            perror("error 3: ");
        if ((result = malloc(sizeof(float) * N)) == NULL)
            perror("error 4: ");
        if ((sendcountS = malloc(sizeof(int) * numprocs)) == NULL)
            perror("error 5: ");
        if ((sendcountG = malloc(sizeof(int) * numprocs)) == NULL)
            perror("error 6: ");
        if ((displsS = malloc(sizeof(int) * numprocs)) == NULL)
            perror("error 7: ");
        if ((displsG = malloc(sizeof(int) * numprocs)) == NULL)
            perror("error 8: ");


        for (int i = 0; i < N; i++)
        {
            vector[i] = i;


            for (int j = 0; j < N; j++)
            {
                matrix[i * N + j] = i + j;
            }
        }
        // sendcount = nº elementos para cada proceso
        for (int i = 0; i < numprocs; i++)
        {
            sendcountS[i] = ((N / numprocs) + (i < N % numprocs ? 1 : 0)) * N; // filas_indiv pero calculadas por el proceso 0 ya que no tiene acceso a filas_indiv de cada proceso, luego multiplica por N porque son filas por N columnas
            sendcountG[i] = ((N / numprocs) + (i < N % numprocs ? 1 : 0));
        }
        // displs = desplazamiento en el array del que cogemos elementos para empezar a enviar al siguiente proceso
        displsS[0] = 0;
        displsG[0] = 0;
        for (int i = 1; i < numprocs; i++)
        {
            displsS[i] = displsS[i - 1] + sendcountS[i - 1];
            displsG[i] = displsG[i - 1] + sendcountG[i - 1];
        }
    }
    // COMUNICACIONES SCATTERV
    gettimeofday(&tv1, NULL);
    MPI_Scatterv(matrix, sendcountS, displsS, MPI_FLOAT, matrixAux, filas_indiv * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(vector, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    gettimeofday(&tv2, NULL);


    printTiempos(rank, "Comunicación Scatterv (segundos)", tv1, tv2);


    // CÁLCULOS
    gettimeofday(&tv1, NULL);
    for (i = 0; i < filas_indiv; i++)
    {
        resultAux[i] = 0;
        for (j = 0; j < N; j++)
        {
            resultAux[i] += matrixAux[i * N + j] * vector[j];
        }
    }
    gettimeofday(&tv2, NULL);
    printTiempos(rank, "Cálculos (segundos)", tv1, tv2);


    // COMUNICACIÓN GATHERV
    gettimeofday(&tv1, NULL);
    MPI_Gatherv(resultAux, filas_indiv, MPI_FLOAT, result, sendcountG, displsG, MPI_FLOAT, 0, MPI_COMM_WORLD);
    gettimeofday(&tv2, NULL);
    printTiempos(rank, "Comunicación Gatherv (segundos)", tv1, tv2);


    /*Display result */


    if (rank == 0)
    {
        for (i = 0; i < N; i++)
        {
            printf(" %f \t ", result[i]);
        }
        free(matrix);
        free(result);
        free(sendcountG);
        free(sendcountS);
        free(displsG);
        free(displsS);
    }
    free(matrixAux);
    free(resultAux);


    MPI_Finalize();


    return 0;
}
