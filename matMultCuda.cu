#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>

#include <stdio.h>
#include <assert.h>

#define COLS1 2000
#define ROWS1 2000
#define COLS2 2000
#define ROWS2 2000

void viewMatrix(int *Matrix, int cols, int rows);
void mulMatrix(const int *A, const int *B, int *C, int rowA, int colA, int rowB, int colB);
//* declare the kernel function
__global__ void kernel_mulMat(const int *A, const int *B, int *C, int rowA, int colA, int rowB, int colB);


using namespace std;

//* declare the vectors' number of elements and their size in bytes
static const size_t sizeA = (ROWS1 * COLS1) * sizeof(int);
static const size_t sizeB = (ROWS2 * COLS2) * sizeof(int);
static const size_t sizeC = (ROWS1 * COLS2) * sizeof(int);

int main(void)
{
    if (ROWS1 != COLS2)
    {
        cout << "Invalid Dimension" << endl;
        return 0;
    }

    srand(30);

    cudaError_t error = cudaSuccess;

    

    //* Declarate an allocate input vectors in the host (CPU) memory
    int* h_A = (int*)malloc(sizeA);
    int* h_B = (int*)malloc(sizeB);
    int* h_C = (int*)malloc(sizeC);

    //* Put values in the matrix
    for (int i = 1; i < COLS1 * ROWS1 + 1; i++)
    {
        h_A[i] = rand() % 10 + 1;
    }

    for (int i = 1; i < COLS2 * ROWS2 + 1; i++)
    {
        h_B[i] = rand() % 10 + 1;
    }

    // viewMatrix(A, ROWS1, COLS1);
    // viewMatrix(B, ROWS2, COLS2);

    //* Declarate device vectors in the device (GPU) memory
    int *d_A, *d_B, *d_C;

    //* Allocate and trasfer vectors in the device (GPU) memory
    error = cudaMalloc((void **)&d_A, sizeA);
    if (error != cudaSuccess)
    {
        cout << "Failed to cudamalloc (error code " << cudaGetErrorString(error) << ")\n";
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **)&d_B, sizeB);
    if (error != cudaSuccess)
    {
        cout << "Failed to cudamalloc (error code " << cudaGetErrorString(error) << ")\n";
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **)&d_C, sizeC);
    if (error != cudaSuccess)
    {
        cout << "Failed to cudamalloc (error code " << cudaGetErrorString(error) << ")\n";
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        cout << "Failed to cudaMemcpyA (error code " << cudaGetErrorString(error) << ")\n";
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        cout << "Failed to cudaMemcpyB (error code " << cudaGetErrorString(error) << ")\n";
        exit(EXIT_FAILURE);
    }

    mulMatrix(d_A, d_B, d_C, ROWS1, COLS1, ROWS2, COLS2);
    cudaDeviceSynchronize();

    error = cudaMemcpy(h_C, d_C, sizeB, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        cout << "Failed to cudaMemcpyC (error code " << cudaGetErrorString(error) << ")\n";
        exit(EXIT_FAILURE);
    }
    
    cudaDeviceSynchronize();


    // viewMatrix(C, ROWS1, COLS2);

    free(h_A);
    free(h_B);
    free(h_C);

    error = cudaFree(d_A);
    if (error != cudaSuccess)
    {
        cout << "Failed to cudaFree (error code " << cudaGetErrorString(error) << ")\n";
        exit(EXIT_FAILURE);
    }

    error = cudaFree(d_B);
    if (error != cudaSuccess)
    {
        cout << "Failed to cudaFree (error code " << cudaGetErrorString(error) << ")\n";
        exit(EXIT_FAILURE);
    }

    error = cudaFree(d_C);
    if (error != cudaSuccess)
    {
        cout << "Failed to cudaFree (error code " << cudaGetErrorString(error) << ")\n";
        exit(EXIT_FAILURE);
    }

    return 0;
}


//* function which invokes the kernel
void mulMatrix(const int *A, const int *B, int *C, int rowA, int colA, int rowB, int colB)
{
    //* declare the number of blocks per grid and the number of threads per block
    //* use 1 to 512 threads per block
    dim3 threadsPerBlock(rowA, colB);
    dim3 blocksPerGrid(1, 1);
    if (ROWS1 * COLS2 > 512)
    {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(int(COLS2) / int(threadsPerBlock.x));
        blocksPerGrid.y = ceil(int(ROWS1) / int(threadsPerBlock.y));
    }

    //* invoke the kernel
    kernel_mulMat<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, rowA, colA, rowB, colB);
}


//* kernel
__global__ void kernel_mulMat(const int *A, const int *B, int *C, int rowA, int colA, int rowB, int colB)
{
    //* calculate the unique thread index
    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    int aux = 0;

    //* perform tid-th elements multiply
    if (ROW < rowA && COL < colB)
    {
        for (int i = 0; i < COLS1; i++)
        {
            aux += A[ROW * rowA * colB + i] * B[i * rowA * colB + COL];
        }
        C[ROW * rowA * colB + COL] = aux;
    }
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
