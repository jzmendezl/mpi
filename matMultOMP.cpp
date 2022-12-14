#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <omp.h>
#include <string.h>

// #define COLS1 2000
// #define ROWS1 2000
// #define COLS2 2000
// #define ROWS2 2000

#define NTRS 16

void viewMatrix(int *Matrix, int cols, int rows);

using namespace std;

int main(int argc, char **argv)
{
    char *size_mat = (char *)malloc(4 * sizeof(char));
    strcat(size_mat, argv[1]);
    int N = atoi(size_mat);
    printf("%d\n", N);
    int COLS1 = N, COLS2 = N, ROWS1 = N, ROWS2 = N;

    if (ROWS1 != COLS2)
    {
        cout << "Invalid Dimension" << endl;
        return 0;
    }

    srand(30);

    int *A = (int *)malloc((ROWS1 * COLS1) * sizeof(int));
    int *B = (int *)malloc((ROWS2 * COLS2) * sizeof(int));
    int *C = (int *)malloc((ROWS1 * COLS2) * sizeof(int));

    //* Put values in the matrix

#pragma opm parallel
    {
#pragma omp for
        for (int i = 1; i < COLS1 * ROWS1 + 1; i++)
        {
            A[i] = rand() % 10 + 1;
        }
    }

#pragma opm parallel
    {
#pragma omp for
        for (int i = 1; i < COLS2 * ROWS2 + 1; i++)
        {
            B[i] = rand() % 10 + 1;
        }
    }

    // viewMatrix(A, ROWS1, COLS1);
    // viewMatrix(B, ROWS2, COLS2);
    cout << "ok\n";
#pragma omp parallel
    {
#pragma omp for collapse(3)
        for (int i = 0; i < ROWS1; i++)
        {
            for (int j = 0; j < ROWS1; j++)
            {
                for (int k = 0; k < COLS2; k++)
                {
                    C[(i * ROWS1) + j] += A[(i * ROWS1) + k] * B[(k * COLS2) + j];
                }
            }
        }
    }

    // viewMatrix(C, ROWS1, COLS2);

    free(A);
    free(B);
    free(C);

    return 0;
}

void viewMatrix(int *Matrix, int cols, int rows)
{
    int viewMatrix[rows][cols];
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            viewMatrix[i][j] = Matrix[i * rows + j];
            cout << viewMatrix[i][j] << "\t";
        }
        cout << "\n";
    }
    cout << "\n";
}
