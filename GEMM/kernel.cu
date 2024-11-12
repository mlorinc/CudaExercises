
// GEMM excercise. Without alpha and beta.

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

// Reference matrix multiply.
static void matrixMultiply(int* c, const int* a, const int* b, int m, int k, int n)
{
	// Initialize the result matrix.
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			c[i * n + j] = 0;
		}
	}

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			for (int l = 0; l < k; l++)
			{
				c[i * n + j] += a[i * k + l] * b[l * n + j];
			}
		}
	}
}

static void matrixMultiplySIMD(int* c, const int* a, const int* b, int m, int k, int n)
{
	// Initialize the result matrix.
	#pragma omp parallel for
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			// Set c[i][j] to 0 for accumulation.
			c[i * n + j] = 0;
		}
	}

	#pragma omp parallel for collapse(3)
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			for (int l = 0; l < k; l++)
			{
				// c[i][j] += a[i][l] * b[l][j];
				c[i * n + j] += a[i * k + l] * b[l * n + j];
			}
		}
	}
}

static void matrixMultiplySIMDTilled(int* c, const int* a, const int* b, int m, int k, int n)
{
	// Initialize the result matrix.
	#pragma omp parallel for
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			c[i * n + j] = 0;
		}
	}

	constexpr int tile = 8;

	#pragma omp parallel for collapse(3)
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			for (int l = 0; l < k; l++)
			{
				// c[i][j] += a[i][l] * b[l][j];
				c[i * n + j] += a[i * k + l] * b[l * n + j];
			}
		}
	}
}

// CUDA Kernel for Matrix Multiplication without optimizations.
__global__ void matrixMultiplyKernel(int* C, const int* A, const int* B, int m, int k, int n)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < m && j < n)
	{
		int sum = 0;
		for (int l = 0; l < k; l++)
		{
			sum += A[i * k + l] * B[l * n + j];
		}
		C[i * n + j] = sum;
	}
}

constexpr int TILE_SIZE = 16;
// CUDA Kernel for Matrix Multiplication using tiling and block cache.
__global__ void matrixTileMultiplyKernel(int* C, const int* A, const int* B, int m, int k, int n)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int col = bx * blockDim.x + tx;
	int row = by * blockDim.y + ty;

	__shared__ int ATile[TILE_SIZE * TILE_SIZE];
	__shared__ int BTile[TILE_SIZE * TILE_SIZE];

	int sum = 0;
	for (int i = 0; i < (k + TILE_SIZE - 1) / TILE_SIZE; i++)
	{
		if (row < m && (i * TILE_SIZE + tx) < k)
		{
			ATile[ty * TILE_SIZE + tx] = A[row * k + (i * TILE_SIZE + tx)];
		}
		else {
			ATile[ty * TILE_SIZE + tx] = 0;
		}

		if (col < n && (i * TILE_SIZE + ty) < k)
		{
			BTile[tx * TILE_SIZE + ty] = B[((i * TILE_SIZE + ty) * n) + col];
		}
		else {
			BTile[tx * TILE_SIZE + ty] = 0;
		}

		__syncthreads();
		for (int j = 0; j < TILE_SIZE; j++)
		{
			sum += ATile[ty * TILE_SIZE + j] * BTile[tx * TILE_SIZE + j];
		}
		__syncthreads();
	}
	C[row * n + col] = sum;
}

static void generateRandomMatrix(int* matrix, int rows, int cols)
{
	for (int i = 0; i < rows * cols; ++i)
	{
		matrix[i] = rand() % 100;
	}
}

static void printMatrix(const int* matrix, int rows, int cols)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			// Print each element, using appropriate formatting for better visibility
			std::cout << matrix[i * cols + j] << "\t";
		}
		std::cout << std::endl;
	}
}

static float measureOperation(cudaEvent_t& start, cudaEvent_t& stop, void (*fn)(int*, const int*, const int*, int, int, int), int* c, const int* a, const int* b, int m, int k, int n)
{
	// Record the start event
	cudaEventRecord(start);

	// Perform matrix multiplication
	fn(c, a, b, m, k, n);

	// Record the stop event
	cudaEventRecord(stop);

	// Wait for the stop event to finish
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	return milliseconds; // Return the elapsed time
}

static float cudaMultiply(int* result, cudaEvent_t& start, cudaEvent_t& stop, void (*fn)(int*, const int*, const int*, int, int, int), const int* a, const int* b, int m, int k, int n)
{
	int* gpuA, * gpuB, * gpuC;
	size_t sizeA = m * k * sizeof(int);
	size_t sizeB = k * n * sizeof(int);
	size_t sizeC = m * n * sizeof(int);

	// Allocate memory on the GPU
	cudaMalloc(&gpuA, sizeA);
	cudaMalloc(&gpuB, sizeB);
	cudaMalloc(&gpuC, sizeC);

	// Copy data from host to device
	cudaMemcpy(gpuA, a, sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuB, b, sizeB, cudaMemcpyHostToDevice);

	// Define block and grid sizes
	dim3 blockSize(16, 16); // 16x16 threads per block
	dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);

	// Record the start event
	cudaEventRecord(start);

	// Launch the kernel
	fn << <gridSize, blockSize >> > (gpuC, gpuA, gpuB, m, k, n);

	// Copy the result back to the host
	cudaMemcpy(result, gpuC, sizeC, cudaMemcpyDeviceToHost);

	// Record the stop event
	cudaEventRecord(stop);

	// Wait for the stop event to finish
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaFree(gpuA); cudaFree(gpuB); cudaFree(gpuC);
	return milliseconds; // Return the elapsed time
}

// Exact Comparison Function
bool areMatricesEqual(const int* A, const int* B, int rows, int cols)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (A[i * cols + j] != B[i * cols + j])
			{
				return false; // Matrices differ at this element
			}
		}
	}
	return true; // All elements are the same
}

int main()
{
	std::srand(static_cast<unsigned>(std::time(0))); // Seed for random number generation

	int m = 512; // Number of rows in matrix A
	int k = 64; // Number of columns in matrix A and number of rows in matrix B
	int n = 128; // Number of columns in matrix B

	size_t sizeA = m * k * sizeof(int);
	size_t sizeB = k * n * sizeof(int);
	size_t sizeC = m * n * sizeof(int);

	float ms = 0;

	// Allocate memory for the matrices
	//int* a = new int[m * k];
	//int* b = new int[k * n];
	//int* c = new int[m * n];

	int* a = (int*)_mm_malloc(sizeA * sizeof(int), 64);
	int* b = (int*)_mm_malloc(sizeB * sizeof(int), 64);
	int* c = (int*)_mm_malloc(sizeC * sizeof(int), 64);
	int* result = (int*)_mm_malloc(sizeC * sizeof(int), 64);

	// Generate random matrices
	generateRandomMatrix(a, m, k);
	generateRandomMatrix(b, k, n);

	std::cout << "Matrix A:" << std::endl;
	//printMatrix(a, m, k);

	std::cout << "Matrix B:" << std::endl;
	//printMatrix(b, k, n);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//ms = measureOperation(start, stop, matrixMultiply, c, a, b, m, k, n);
	//std::cout << "matrixMultiply:" << ms << " [ms]" << std::endl;
	//ms = measureOperation(start, stop, matrixMultiplySIMD, c, a, b, m, k, n);
	//std::cout << "matrixMultiplySIMD:" << ms << " [ms]" << std::endl;
	matrixMultiply(c, a, b, m, k, n);
	ms = cudaMultiply(result, start, stop, matrixTileMultiplyKernel, a, b, m, k, n);
	//std::cout << "cuda:" << ms << " [ms]" << std::endl;
	// Clean up

	std::cout << (areMatricesEqual(c, result, m, n) ? ("Ok") : "Fail") << std::endl;
	_mm_free(a); _mm_free(b); _mm_free(c);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}
