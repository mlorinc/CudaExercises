
/**
Calculate array prefix-sum using Blelloch algorithm of upsweep and downsweep phases
utilising capabilities of CUDA.
**/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <type_traits>
#include <cstdlib>
#include <string.h>
#include <cassert>
#include <exception>
#include <string>
#include <memory>

// Number of elements in array.
constexpr int ARRAY_SIZE = 900000003;
// Quantity of numbers a single thread handles at most.
constexpr int DATA_PER_THREAD = 4;
// CUDA block size to use
constexpr int BLOCK_SIZE = 256;
// Number of elements residing in single block.
constexpr int DATA_PER_BLOCK = BLOCK_SIZE * DATA_PER_THREAD;
// Largest possible number of elements resigin in a block.
constexpr int MAX_BLOCK_SIZE = 1024;

// Generic CUDA related exception used in this program.
class GpuException : public std::exception
{
private:
	// Error message.
	std::string message;
public:
	// CUDA error status.
	const cudaError_t status;

	// Create a new exception with given CUDA error and error message.
	GpuException(cudaError_t e, const std::string& message) :
		std::exception(), status(e), message(message) {}

	// Create with given error message. CUDA error will be set to cudaErrorMemoryAllocation.
	GpuException(const std::string& message) :
		std::exception(), status(cudaErrorMemoryAllocation), message(message) {}

	// Get error message.
	const char* what() const noexcept override {
		return message.c_str();
	}
};

// CUDA exception signalling kernel launch error.
class GpuLaunchException : public GpuException
{
public:
	// Create a new exception with given CUDA error and impacted kernel.
	GpuLaunchException(cudaError_t e, const std::string& kernelName) :
		GpuException(e, "kernel \"" + kernelName + "\" launch failed: " + std::string(cudaGetErrorString(e))) {}
};

// CUDA exception signalling kernel synchronization error.
class GpuSynchronizeException : public GpuException
{
public:
	// Create a new exception with given CUDA error and impacted kernel.
	GpuSynchronizeException(cudaError_t e, const std::string& kernelName) :
		GpuException(e, "cudaDeviceSynchronize returned error code " + std::to_string(e) + " after launching \"" + kernelName + "\"") {}
};

// CUDA exception signalling kernel L2 cache error.
class GpuCacheException : public GpuException
{
public:
	// Create a new exception with given CUDA error and impacted kernel, actual cache size and expected maximum cache size.
	GpuCacheException(const std::string& kernelName, int actual, int expected) :
		GpuException(
			cudaErrorMemoryAllocation,
			"CUDA kernel returned error code " + std::to_string(cudaErrorMemoryAllocation) +
			" after launching \"" + kernelName + "\"; not enough cache memory to accomodate data, expected less than " +
			std::to_string(expected) + ", got " + std::to_string(actual)) {}
};

// CUDA exception signalling generic memory error.
class GpuMemoryException : public GpuException
{
public:
	// Create a new exception with given CUDA error and error message.
	GpuMemoryException(cudaError_t e, const std::string& message) :
		GpuException(e, message) {}
};

// Perform prefix sum on CPU. Assumes out != vector.
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

// Perform exclusive prefix sum on CPU. Same vector out == vector is supported here.
template<
	typename T,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
void cpuPrescan(T* out, T* vector, int size)
{
	if (size == 0)
	{
		return;
	}

	T temp = 0;
	for (int i = 0; i < size; i++)
	{
		T current = vector[i];
		out[i] = temp;
		temp += current;	
	}
}

// Move chunk of data from DRAM into block cache.
template<
	typename T,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
__device__ void toScratch(T* scratchpad, const T* inputVector, int dataPerThread, int dataPerBlock, int arraySize, T neutralElement)
{
	#pragma unroll
	for (int i = 0; i < dataPerThread; i++)
	{
		const int localTid = threadIdx.x * dataPerThread + i;
		const int tid = blockIdx.x * dataPerBlock + localTid;
		// If data are withing array size the it is good. Otherwise neutral element must be inserted
		// not to interfere with result. Doing it this way several if checks can be removed in later phases.
		scratchpad[localTid] = (tid < arraySize) ? (inputVector[tid]) : (neutralElement);
	}
	__syncthreads();
}

// Perform upsweep phase. Summing each leaf together and bubbling up until root is reached.
template<
	typename T,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
__device__ int upsweep(T* scratchpad, int dataPerThread, int dataPerBlock)
{
	int stride;
	for (stride = 1; stride < dataPerBlock; stride *= 2)
	{
		#pragma unroll
		for (int i = 0; i < dataPerThread; i++)
		{
			// Striding is not ideal because it is in powers of 2.
			// It is known to cause memory bank conflicts.
			const int localTid = threadIdx.x * dataPerThread + i;
			int index = 2 * stride * localTid;
			if (index < dataPerBlock)
			{
				scratchpad[index] = scratchpad[index] + scratchpad[index + stride];
			}
		}
		__syncthreads();
	}

	return stride;
}

// Perform upsweep correction which replaces overall sum with neutral element
// so the algorith can proceed to next phase. Max value must be saved for later.
template<
	typename T,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
__device__ T upsweepCorrection(T* scratchpad, T* tempSums, T neutralElement)
{
	const int max = scratchpad[0];
	if (threadIdx.x == 0)
	{
		// Set first temp sum to max. It is used to determine value of left children
		// as their value were lost in upsweep. This way they can be recovered on fly.
		tempSums[0] = max;
		scratchpad[0] = neutralElement;
	}
	__syncthreads();
	return max;
}

// Perform downsweep.
template<
	typename T,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
__device__ void downsweep(T* scratchpad, T* tempSums, int stride, int dataPerThread, int dataPerBlock)
{
	for (stride /= 2; stride >= 1; stride /= 2)
	{
		#pragma unroll
		for (int i = 0; i < dataPerThread; i++)
		{
			const int localTid = threadIdx.x * dataPerThread + i;
			int index = 2 * stride * localTid;
			if (index < dataPerBlock)
			{
				// Calculate lost value in upsweep.
				const int left = tempSums[index] - scratchpad[index + stride];
				tempSums[index] = left;
				tempSums[index + stride] = scratchpad[index + stride];
				scratchpad[index + stride] = scratchpad[index] + left;
			}
		}
		__syncthreads();
	}
}

// Perform downsweep correction. After downsweep, there is a neutral element on first index
// and overall accumulated sum is not present on last index. First issue can be fixed by shifting
// by one to left. Max sum is saved in register and available to restore.
template<
	typename T,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
__device__ void downsweepCorrection(T* scratchpad, T* downsweepVector, int max, int dataPerThread, int dataPerBlock, int arraySize)
{
	#pragma unroll
	for (int i = 0; i < dataPerThread; i++)
	{
		const int localTid = threadIdx.x * dataPerThread + i;
		const int tid = blockIdx.x * dataPerBlock + localTid;

		if (tid < arraySize)
		{
			if (tid + 1 != arraySize && localTid + 1 != dataPerBlock)
			{
				// Shift one time to left because neutral element does not belong to prefix sum.
				downsweepVector[tid] = scratchpad[localTid + 1];
			}
			else
			{
				// Set the last element to "max" for prefix sum completion
				downsweepVector[tid] = max;
			}
		}
	}
	//__syncthreads();
}

// Perform exclusive prefix sum. Very similar to prefix sum. Actually it is 
// without downsweep correction.
template<
	typename T,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
>
__device__ void prescanBlockKernelHelper(T* result, T* scratchpad, T* tempSums, T* maximums, int dataPerThread, int dataPerBlock, int arraySize, T neutralElement)
{
	int stride = upsweep(scratchpad, dataPerThread, dataPerBlock);
	T max = upsweepCorrection(scratchpad, tempSums, neutralElement);
	downsweep(scratchpad, tempSums, stride, dataPerThread, dataPerBlock);

	if (maximums != NULL && threadIdx.x == 0)
	{
		maximums[blockIdx.x] = max;
	}

	if (result != scratchpad)
	{
		#pragma unroll
		for (int i = 0; i < dataPerThread; i++)
		{
			const int localTid = threadIdx.x * dataPerThread + i;
			const int tid = blockIdx.x * dataPerBlock + localTid;
			if (tid < arraySize)
			{
				result[tid] = scratchpad[localTid];
			}
		}
	}
}

template<
	typename T,
	int LOCAL_DATA_PER_BLOCK = 256,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
>
__global__ void prescanBlockKernel(T* result, T* inputVector, T* maximums, int dataPerThread, int dataPerBlock, int arraySize, T neutralElement)
{
	assert(dataPerBlock <= LOCAL_DATA_PER_BLOCK);
	__shared__ T scratchpad[LOCAL_DATA_PER_BLOCK];
	__shared__ T tempSums[LOCAL_DATA_PER_BLOCK];
	toScratch(scratchpad, inputVector, dataPerThread, dataPerBlock, arraySize, neutralElement);
	prescanBlockKernelHelper<T, LOCAL_DATA_PER_BLOCK>(result, scratchpad, tempSums, maximums, dataPerThread, dataPerBlock, arraySize, neutralElement);
}

template<
	typename T,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
>
__global__ void prescanBlockKernelDRAM(T* result, T* inputVector, T* tempSums, T* maximums, int dataPerThread, int dataPerBlock, int arraySize, T neutralElement)
{
	prescanBlockKernelHelper<T>(result, inputVector, tempSums, maximums, dataPerThread, dataPerBlock, arraySize, neutralElement);
}

// Perform prefix sum on data. Each block has it is own prefix sum and must be corrected to make it global.
template<
	typename T,
	int LOCAL_DATA_PER_BLOCK = 256,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
>
__global__ void prefixSumBlockKernel(T* result, T* inputVector, T* maximums, int dataPerThread, int dataPerBlock, int arraySize, T neutralElement)
{
	assert(dataPerBlock <= LOCAL_DATA_PER_BLOCK);

	// Use L2 cache to avoid accesing DRAM and slowing down calculation.
	__shared__ T scratchpad[LOCAL_DATA_PER_BLOCK];
	// Save temporary sums so missing values can be calculated again.
	__shared__ T tempSums[LOCAL_DATA_PER_BLOCK];

	toScratch(scratchpad, inputVector, dataPerThread, dataPerBlock, arraySize, neutralElement);
	int stride = upsweep(scratchpad, dataPerThread, dataPerBlock);
	T max = upsweepCorrection(scratchpad, tempSums, neutralElement);
	downsweep(scratchpad, tempSums, stride, dataPerThread, dataPerBlock);
	downsweepCorrection(scratchpad, result, max, dataPerThread, dataPerBlock, arraySize);

	if (maximums != NULL && threadIdx.x == 0)
	{
		maximums[blockIdx.x] = max;
	}
}

// Perform correction to make block prefix sums global. It requires correction vector to function.
// Correction vector can be obtained by calculating prescan (exclusive prefix sum) on maximum values of
// each block in prefix sum.
template<
	typename T,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
__global__ void scanBlockCorrectionKernel(T* inputVector, const T* blockCorrectionVector, int dataPerThread, int dataPerBlock, int arraySize)
{
	#pragma unroll
	for (int i = 0; i < dataPerThread; i++)
	{
		const int localTid = threadIdx.x * dataPerThread + i;
		const int tid = blockIdx.x * dataPerBlock + localTid;
		if (tid < arraySize)
		{
			inputVector[tid] += blockCorrectionVector[blockIdx.x];
		}
	}
}

// CUDA helper class to handle data transfers and block/thread count calculations.
// Moreover, it contains some facade methods for prefix sum.
template<typename T>
class Gpu
{
private:
	T* inputVector = nullptr;
	T* scanVector = nullptr;
	T* maxVector = nullptr;
	T neutralElement = 0;
	int arraySize;
	int blockCount;
	int blockSize;
	int dataPerBlock;
	int dataPerThread;

public:
	Gpu() = delete;
	explicit Gpu(int blockSize, int arraySize, int dataPerThread, T neutralElement)
	{
		this->arraySize = arraySize;
		this->blockSize = blockSize;
		this->dataPerThread = dataPerThread;
		this->dataPerBlock = blockSize * dataPerThread;
		this->blockCount = (arraySize + dataPerBlock - 1) / dataPerBlock;
		this->neutralElement = neutralElement;
	}

	// Disable copy constructor, move constructor and assignment
	Gpu(const Gpu&) = delete;
	Gpu(const Gpu&&) = delete;
	Gpu& operator=(const Gpu&) = delete;

	decltype(inputVector) getInputVector() const
	{
		return inputVector;
	}

	decltype(scanVector) getOutputVector() const
	{
		return scanVector;
	}

	decltype(maxVector) getMaxVector() const
	{
		return maxVector;
	}

	decltype(arraySize) getArraySize() const
	{
		return arraySize;
	}

	decltype(blockCount) getBlockCount() const
	{
		return blockCount;
	}

	void checkKernelLaunch(const std::string& kernel)
	{
		auto cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			throw GpuLaunchException(cudaStatus, kernel);
		}
	}

	void synchronizeKernel(const std::string& kernel)
	{
		auto cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			throw GpuSynchronizeException(cudaStatus, kernel);
		}
	}

	void init(const T* input)
	{
		// Free remaining resources if neccesary
		freeAll();
		// Allocate GPU buffer for input.
		cudaError_t cudaStatus = cudaMalloc(&inputVector, getArraySize() * sizeof(T));
		if (cudaStatus != cudaSuccess)
		{
			throw GpuMemoryException(cudaStatus, "could not allocate memory for GPU input vector; array size: " + std::to_string(getArraySize()));
		}

		// Allocate GPU buffer for output.
		cudaStatus = cudaMalloc(&scanVector, getArraySize() * sizeof(T));
		if (cudaStatus != cudaSuccess) 
		{
			throw GpuMemoryException(cudaStatus, "could not allocate memory for GPU scan output vector; array size: " + std::to_string(getArraySize()));
		}

		// Transfer input vector to GPU.
		cudaStatus = cudaMemcpy(inputVector, input, getArraySize() * sizeof(T), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) 
		{
			throw GpuMemoryException(cudaStatus, "could not copy CPU vector into GPU memory; array size: " + std::to_string(getArraySize()));
		}

		// Allocate GPU max vector. Used for correction later.    .
		cudaStatus = cudaMalloc(&maxVector, blockCount * sizeof(T));
		if (cudaStatus != cudaSuccess) 
		{
			throw GpuMemoryException(cudaStatus, "could not allocate GPU max vector; block count: " + std::to_string(blockCount));
		}
	}

	void blockPrefixSum()
	{
		dim3 blockSize(blockSize);
		dim3 gridSize(blockCount);

		if (dataPerBlock > DATA_PER_BLOCK)
		{
			throw GpuCacheException("prefixSumBlockKernel", dataPerBlock, DATA_PER_BLOCK);
		}

		printf("launching blockPrefixSum on grid size %d with block size %d\n", gridSize.x, blockSize.x);
		prefixSumBlockKernel<T, DATA_PER_BLOCK> << <gridSize, blockSize >> > (
			getOutputVector(),
			getInputVector(),
			getMaxVector(),
			dataPerThread,
			dataPerBlock,
			getArraySize(),
			0);

		checkKernelLaunch("prefixSumBlockKernel");
		synchronizeKernel("prefixSumBlockKernel");
	}

	void prefixSum()
	{
		blockPrefixSum();

		// Find correction values for each block which was calcualtedi independtly hence it does not have information from previous block
		// This can be achived by doing by doing prescan, defined as: y[0] = 0, y[1] = a[0], y[2]= y[1] + a[2], ..., y[n] = y[n-1] + a[n]
		// However number of different blocks is not trivial to handle. Nonetheless, there is not many blocks anyway, so for now do it on CPU.
		// To note, if there is less than 1024 threads a single block prescan can be used to calculate this on GPU.
		auto maxValuesPerBlock = std::make_unique<T[]>(blockCount);
		auto cudaStatus = cudaMemcpy(maxValuesPerBlock.get(), getMaxVector(), blockCount * sizeof(T), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) 
		{
			throw GpuMemoryException(cudaStatus, "could not copy GPU max vector into CPU memory; array size: " + std::to_string(blockCount));
		}

		// Calculate correction vector.
		cpuPrescan(maxValuesPerBlock.get(), maxValuesPerBlock.get(), blockCount);

		cudaStatus = cudaMemcpy(getMaxVector(), maxValuesPerBlock.get(), blockCount * sizeof(T), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) 
		{
			throw GpuMemoryException(cudaStatus, "could not copy CPU max vector into GPU memory; array size: " + std::to_string(blockCount));
		}

		dim3 blockSizeCorrection(blockSize);
		dim3 gridSizeCorrection(blockCount);
		// Use correction vector to create global prefix sum.
		scanBlockCorrectionKernel << <gridSizeCorrection, blockSizeCorrection >> > (getOutputVector(), getMaxVector(), dataPerThread, dataPerBlock, getArraySize());
		checkKernelLaunch("scanBlockCorrectionKernel");
		synchronizeKernel("scanBlockCorrectionKernel");
	}

	// Copy result from GPU to designated output allocated on CPU.
	void getResult(T* output)
	{
		auto cudaStatus = cudaMemcpy(output, getOutputVector(), getArraySize() * sizeof(T), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) 
		{
			throw GpuMemoryException(cudaStatus, "could not copy GPU vector into CPU memory; array size: " + std::to_string(getArraySize()));
		}
	}

	// Free GPU variable.
	void freeGpu(T*& pointer, const std::string& name)
	{
		if (pointer == nullptr)
		{
			return;
		}

		auto cudaStatus = cudaFree(pointer);

		if (cudaStatus != cudaSuccess)
		{
			throw GpuMemoryException(cudaStatus, "could not deallocate GPU buffer " + name);
		}
		pointer = nullptr;
	}

	void freeAll()
	{
		freeGpu(inputVector, "inputVector");
		freeGpu(scanVector, "scanVector");
		freeGpu(maxVector, "maxVector");
	}

	~Gpu()
	{
		freeAll();
	}
};

template<
	typename T,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
void gpuScan(T* scanVector, const T* inputVector, T neutralElement)
{
	Gpu<T> gpuData(BLOCK_SIZE, ARRAY_SIZE, DATA_PER_THREAD, neutralElement);
	// Choose which GPU to run on, change this on inputVector multi-GPU system.
	auto cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) 
	{
		throw GpuException(cudaStatus, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	gpuData.init(inputVector);
	gpuData.prefixSum();
	gpuData.getResult(scanVector);
}

static void generateRandomVector(int* matrix, int N)
{
	for (int i = 0; i < N; ++i)
	{
		matrix[i] = rand() % 100;
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
	//static_assert(ARRAY_SIZE % DATA_PER_THREAD == 0, "ARRAY_SIZE must be a multiple of DATA_PER_THREAD");
	static_assert(DATA_PER_BLOCK * sizeof(int) <= (1 << 12) * 12 / 2, "DATA_PER_BLOCK must be smaller than 48 KiB");
	//static_assert(DATA_PER_BLOCK * BLOCK_COUNT * sizeof(int) <= (48 * 1024 * 1024 / 2), "DATA_PER_BLOCK must be smaller than 48 MB");


	int* gpuVector = (int*)malloc(ARRAY_SIZE * sizeof(int));
	int* cpuVector = (int*)malloc(ARRAY_SIZE * sizeof(int));
	int* gpuScanOutput = (int*)malloc(ARRAY_SIZE * sizeof(int));
	int* cpuScanOutput = (int*)malloc(ARRAY_SIZE * sizeof(int));
	assert(gpuVector); assert(cpuVector); assert(gpuScanOutput); assert(cpuScanOutput);

	generateRandomVector(gpuVector, ARRAY_SIZE);
	memcpy(cpuVector, gpuVector, ARRAY_SIZE * sizeof(int));

	try {
		gpuScan(gpuScanOutput, gpuVector, 0);
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
		auto cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}
	}
	catch (const GpuException& e)
	{
		fprintf(stderr, e.what());
		return e.status;
	}

	return 0;
}
