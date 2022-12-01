#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

#define COLS1 2000
#define ROWS1 2000
#define COLS2 2000
#define ROWS2 2000

void viewMatrix(int *Matrix, int cols, int rows);

using namespace std;

int main(void)
{
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
    for (int i = 1; i < COLS1 * ROWS1 + 1; i++)
    {
        A[i] = rand() % 10 + 1;
    }

    for (int i = 1; i < COLS2 * ROWS2 + 1; i++)
    {
        B[i] = rand() % 10 + 1;
    }

    // viewMatrix(A, ROWS1, COLS1);
    // viewMatrix(B, ROWS2, COLS2);

    for (int i = 0; i < ROWS1; i++)
    {
        for (int j = 0; j < ROWS1; j++)
        {
            int aux = 0;
            for (int k = 0; k < COLS2; k++)
            {
                C[(i * ROWS1) + j] += A[(i * ROWS1) + k] * B[(k * COLS2) + j];
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
