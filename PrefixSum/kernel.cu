
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <type_traits>
#include <cstdlib>
#include <string.h>
#include <cassert>

constexpr int ARRAY_SIZE = 52000;
constexpr int BLOCK_SIZE = 256;

template<
	typename T,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
void cpuScan(T* out, T* vector, int size)
{
	if (size == 0)
	{
		return;
	}

	out[0] = vector[0];
	for (int i = 1; i < size; i++)
	{
		out[i] = out[i - 1] + vector[i];
	}
}

template<
	typename T,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
__device__ void toScratch(T scratchpad[BLOCK_SIZE], const T* inputVector)
{
	scratchpad[threadIdx.x] = inputVector[blockIdx.x * blockDim.x + threadIdx.x];
	__syncthreads();
}

template<
	typename T,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
__device__ int upsweep(T scratchpad[BLOCK_SIZE])
{
	int stride;
	for (stride = 1; stride < blockDim.x; stride *= 2)
	{
		int index = 2 * stride * threadIdx.x;
		if (index < blockDim.x)
		{
			scratchpad[index] = scratchpad[index] + scratchpad[index + stride];
		}
		__syncthreads();
	}

	return stride;
}

template<
	typename T,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
__device__ T upsweepCorrection(T scratchpad[BLOCK_SIZE], T tempSums[BLOCK_SIZE])
{
	const int max = scratchpad[0];
	if (threadIdx.x == 0)
	{
		tempSums[0] = scratchpad[0];
		scratchpad[0] = 0;
	}
	__syncthreads();
	return max;
}

template<
	typename T,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
__device__ void downsweep(T scratchpad[BLOCK_SIZE], T tempSums[BLOCK_SIZE], int stride)
{
	for (stride /= 2; stride >= 1; stride /= 2)
	{
		int index = 2 * stride * threadIdx.x;
		if (index < blockDim.x)
		{
			const int left = tempSums[index] - scratchpad[index + stride];
			tempSums[index] = left;
			tempSums[index + stride] = scratchpad[index + stride];
			scratchpad[index + stride] = scratchpad[index] + left;
		}
		__syncthreads();
	}
}

template<
	typename T,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
__device__ void downsweepCorrection(T scratchpad[BLOCK_SIZE], T* downsweepVector, int max)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid + 1 < ARRAY_SIZE && threadIdx.x + 1 < blockDim.x)
	{
		downsweepVector[tid] = scratchpad[threadIdx.x + 1];
	}
	else if (tid + 1 == ARRAY_SIZE || threadIdx.x + 1 == blockDim.x)
	{
		downsweepVector[tid] = max;
	}
	else
	{
		// ignore
	}
}

template<
	typename T,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
__global__ void prescanBlockKernel(T* result, T* inputVector, T* maximums)
{
	__shared__ T scratchpad[BLOCK_SIZE];
	__shared__ T tempSums[BLOCK_SIZE];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < ARRAY_SIZE)
	{
		toScratch(scratchpad, inputVector);
		int stride = upsweep(scratchpad);
		T max = upsweepCorrection(scratchpad, tempSums);
		downsweep(scratchpad, tempSums, stride);

		if (maximums != NULL && threadIdx.x == 0)
		{
			maximums[blockIdx.x] = max;
		}

		const int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid < ARRAY_SIZE)
		{
			result[tid] = scratchpad[threadIdx.x];
		}
	}
}

template<
	typename T,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
__global__ void scanBlockKernel(T* result, T* inputVector, T* maximums)
{
	__shared__ T scratchpad[BLOCK_SIZE];
	__shared__ T tempSums[BLOCK_SIZE];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < ARRAY_SIZE)
	{
		toScratch(scratchpad, inputVector);
		int stride = upsweep(scratchpad);
		T max = upsweepCorrection(scratchpad, tempSums);
		downsweep(scratchpad, tempSums, stride);
		downsweepCorrection(scratchpad, result, max);

		if (maximums != NULL && threadIdx.x == 0)
		{
			maximums[blockIdx.x] = max;
		}
	}
}

template<
	typename T,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
__global__ void scanBlockCorrectionKernel(T* inputVector, const T* blockCorrectionVector)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < ARRAY_SIZE)
	{
		inputVector[tid] += blockCorrectionVector[blockIdx.x];
	}
}

template<
	typename T,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
cudaError_t getScanCorrectionVector(T* maximums, int blockCount)
{
	// Define block and grid sizes
	dim3 blockSize(blockCount);
	dim3 gridSize(1);

	prescanBlockKernel<T> << <gridSize, blockSize >> > (maximums, maximums, NULL);
	return cudaGetLastError();
}

template<
	typename T,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
cudaError_t gpuScan(T* scanVector, const T* inputVector, bool blockCorrection = false)
{
	T* dev_inputVector = 0, * dev_scan = 0, * dev_max = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on inputVector multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc(&dev_inputVector, ARRAY_SIZE * sizeof(T));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_inputVector failed!");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc(&dev_scan, ARRAY_SIZE * sizeof(T));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_scan failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_inputVector, inputVector, ARRAY_SIZE * sizeof(T), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy dev_inputVector failed!");
		goto Error;
	}

	// Define block and grid sizes
	dim3 blockSize(BLOCK_SIZE);
	dim3 gridSize((ARRAY_SIZE + blockSize.x - 1) / blockSize.x);

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc(&dev_max, gridSize.x * sizeof(T));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_max failed!");
		goto Error;
	}

	printf("Launching cuda on grid size %d with block size %d\n", gridSize.x, blockSize.x);
	// Launch inputVector kernel on the GPU with one thread for each element.
	scanBlockKernel << <gridSize, blockSize >> > (dev_scan, dev_inputVector, dev_max);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "scanBlockKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching scanBlockKernel!\n", cudaStatus);
		goto Error;
	}

	if (blockCorrection)
	{
		getScanCorrectionVector(dev_max, gridSize.x);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "getScanCorrectionVector launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching getScanCorrectionVector!\n", cudaStatus);
			goto Error;
		}

		scanBlockCorrectionKernel << <gridSize, blockSize >> > (dev_scan, dev_max);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "scanBlockCorrectionKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching scanBlockCorrectionKernel!\n", cudaStatus);
			goto Error;
		}
	}

	cudaStatus = cudaMemcpy(scanVector, dev_scan, ARRAY_SIZE * sizeof(T), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

Error:
	cudaFree(dev_inputVector); cudaFree(dev_scan), cudaFree(dev_max);

	return cudaStatus;
}

static void generateRandomVector(int* matrix, int N) {
	for (int i = 0; i < N; ++i) {
		matrix[i] = rand() % 100; // Fill with random values from 0 to 99
	}
}

template<
	typename T,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
int verify(T* expected, T* actual, int size)
{
	for (int i = 0; i < size; i++)
	{
		if (expected[i] != actual[i])
		{
			return i;
		}
	}

	return -1;
}

int main()
{
	int* gpuVector = (int*)malloc(ARRAY_SIZE * sizeof(int));
	int* cpuVector = (int*)malloc(ARRAY_SIZE * sizeof(int));
	int* gpuScanOutput = (int*)malloc(ARRAY_SIZE * sizeof(int));
	int* cpuScanOutput = (int*)malloc(ARRAY_SIZE * sizeof(int));

	assert(gpuVector); assert(cpuVector); assert(gpuScanOutput); assert(cpuScanOutput);

	generateRandomVector(gpuVector, ARRAY_SIZE);
	memcpy(cpuVector, gpuVector, ARRAY_SIZE * sizeof(int));

	// Add vectors in parallel.
	cudaError_t cudaStatus = gpuScan(gpuScanOutput, gpuVector, true);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	cpuScan(cpuScanOutput, cpuVector, ARRAY_SIZE);

	int index = verify(cpuScanOutput, gpuScanOutput, ARRAY_SIZE);
	if (index == -1)
	{
		printf("OK\n");
	}
	else
	{
		printf("Fail at %d. Expected: %d; Actual: %d\n", index, cpuScanOutput[index], gpuScanOutput[index]);
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}
