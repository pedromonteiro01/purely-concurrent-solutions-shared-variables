/**
 *  \file prog1.cu
 *
 *  \brief Problem name: Int Sort Row processing.
 *
 *  \authors Pedro Monteiro & José Trigo - June 2023
 */

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"
#include <cuda_runtime.h>

/**
 *   program configuration
 */

#ifndef DIM
# define DIM 1024
#endif

/* allusion to internal functions */

/* returns the number of seconds elapsed between the two specified times */
static double get_delta_time(void);

/* returns 1 if the specified array is sorted, and 0 otherwise */
int validateSort(int *arr, int N);

/* Function to merge two haves of array */
__device__ void merge(int A[], int temp[], int from, int mid, int to);

/* Iterative mergesort */
__device__ void mergeSort(int array[], int temp[], int size);

/* kernel function */
__global__ void processor(int *data, int *temp, int iter);


/**
 *  \brief Function merge.
 *
 *  This function merges two sorted subarrays into a single sorted subarray.
 *
 *  \param arr: pointer to the array containing the subarrays
 *  \param l: starting index of the first subarray
 *  \param m: ending index of the first subarray and starting index of the second subarray
 *  \param r: ending index of the second subarray
 *
 *  The function creates temporary arrays to store the subarrays and then merges them into the original 
 *  array in a sorted order.
 */
```c
// Merge two sorted subarrays `A[from…mid]` and `A[mid+1…to]`
__device__ void merge(int A[], int temp[], int from, int mid, int to)
{
    int k = from, i = from, j = mid + 1;
 
    // loop till no elements are left in the left and right runs
    while (i <= mid && j <= to)
    {
        if (A[i] < A[j]) {
            temp[k++] = A[i++];
        }
        else {
            temp[k++] = A[j++];
        }
    }
 
    // copy remaining elements
    while (i < N && i <= mid) {
        temp[k++] = A[i++];
    }
 
    /* no need to copy the second half (since the remaining items
       are already in their correct position in the temporary array) */
 
    // copy back to the original array to reflect sorted order
    for (int i = from; i <= to; i++) {
        A[i] = temp[i];
    }
}
```

/**
 *  \brief Function mergeSort.
 *
 *  This function sorts an array using the merge sort algorithm.
 *
 *  \param array: pointer to the array to be sorted
 *  \param size: size of the array
 *
 *  The function divides the array into smaller subarrays and recursively sorts them using merge sort. 
 *  It then merges the sorted subarrays to obtain the final sorted array.
 */
__device__ void mergeSort(int array[], int temp[], int size) {
   int currentSize, leftStart;
	
	for (currentSize = 1; currentSize <= size - 1; currentSize = 2 * currentSize) {
		for (leftStart = 0; leftStart < size - 1; leftStart += 2 * currentSize) {
           int middle = min(leftStart + currentSize - 1, size - 1);
           int rightEnd = min(leftStart + 2 * currentSize - 1, size - 1);
           merge(array, temp, leftStart, middle, rightEnd);
       	}
   	}
}

/**
 *  \brief Function processor.
 *
 *  This CUDA kernel function performs parallel processing on the input array using merge sort algorithm.
 *
 *  \param data: pointer to the input array
 *  \param iter: iteration number indicating the level of merge sort
 *
 *  The function divides the input array into subsequences and sorts them using merge sort.
 *  Each thread is responsible for sorting a specific subsequence.
 *  In each iteration, the function performs either an independent merge sort on a subsequence (when iter is 0) 
 *  or merges two previously sorted subsequences.
 */
__global__ void processor(int *data, int *temp, int iter) {
	int N = DIM;
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int idx = blockDim.x * gridDim.x * y + x;

	if(idx >= (N >> iter)) return;

	int start = N * (1 << iter) * idx;
	int end = start + (1 << iter) * N;
	int mid = (start+end)/2;
	int subseq_len = (1 << iter) * N;
	int *subseq_start = data + start;

	(iter == 0) ? mergeSort(subseq_start, temp, subseq_len) : merge(data, temp, start, mid-1, end-1);
}


/**
 *  \brief Main function.
 *
 *  This function is the entry point of the program.
 *
 *  \param argc: number of command-line arguments
 *  \param argv: array of command-line argument strings
 *
 *  The function reads an input file containing integers, performs merge sort using CUDA, 
 *  and validates the sorted array.
 */
int main (int argc, char **argv)
{
	if (argc != 2) {
		printf("Usage: %s <filename>\n", argv[0]);
		return 1;
	}

	/* Open the file for reading */

	FILE *file = fopen(argv[1], "rb");
	if (file == NULL) {
		printf("Failed to open file: %s\n", argv[1]);
		return 1;
	}

	fseek(file, 0, SEEK_END);
	int size = ftell(file) / sizeof(int);
	fseek(file, 0, SEEK_SET);

	int *host_matrix = (int*) malloc(size * sizeof(int));
	if (host_matrix == NULL) {
		printf("Error: cannot allocate memory\n");
		return 1;
	}

	int count = fread(host_matrix, sizeof(int), size, file);

	if (count != size) {
		printf("Error: could not read all integers from file\n");
		return 1;
	}

	fclose(file);


	/* set up the device */

	int dev = 0;

	cudaDeviceProp deviceProp;
	CHECK (cudaGetDeviceProperties (&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK (cudaSetDevice (dev));

	/* copy the host data to the device memory */
	int *device_matrix;
	CHECK(cudaMalloc((void**)&device_matrix, DIM * DIM * sizeof(int)));
	CHECK(cudaMemcpy(device_matrix, host_matrix, DIM * sizeof(int[DIM]), cudaMemcpyHostToDevice));


	/* launch the kernel */

	int gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ;

	// Number of threads in each dimension of a block
	blockDimX = 1 << 10;                                             // optimize!
	blockDimY = 1 << 0;                                             // optimize!
	blockDimZ = 1 << 0;                                             // do not change!

	// Number of blocks in each dimension of the grid
	gridDimX = DIM / (blockDimX*blockDimY*blockDimZ);
	gridDimY = 1 << 0;
	gridDimZ = 1 << 0;                                              // do not change!

	dim3 grid (gridDimX, gridDimY, gridDimZ);
	dim3 block (blockDimX, blockDimY, blockDimZ);

	if ((gridDimX * gridDimY * gridDimZ * blockDimX * blockDimY * blockDimZ) != DIM) {
		printf ("Wrong configuration!\n");
		printf("blockDimX = %d, blockDimY = %d, blockDimZ = %d\n", blockDimX, blockDimY, blockDimZ);
		printf("gridDimX = %d, gridDimY = %d, gridDimZ = %d\n", gridDimX, gridDimY, gridDimZ);
		return 1;
	}

	// Perform merge sort
	(void) get_delta_time ();

	for (int iter = 0; iter < 10; iter++) {
		processor<<<grid, block>>>(device_matrix, temp, iter);

		blockDimX = DIM / (1 << (iter + 1));  // Divides by 2 each iteration
		gridDimX = DIM / blockDimX;
		dim3 block (blockDimX, blockDimY, blockDimZ);
		dim3 grid (gridDimX, gridDimY, gridDimZ);

		CHECK (cudaDeviceSynchronize ());                            // wait for kernel to finish
		CHECK (cudaGetLastError ());                                 // check for kernel errors
	}

	// Process one more iteration to merge the two halves (without updating the grid and block dimensions)
	processor<<<grid, block>>>(device_matrix, temp, 10);
	CHECK (cudaDeviceSynchronize ());                            // wait for kernel to finish
	CHECK (cudaGetLastError ());                                 // check for kernel errors

	printf("The CUDA kernel <<<(%d,%d,%d), (%d,%d,%d)>>> took %.3e seconds to run\n",
			gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, get_delta_time ());

	/* copy kernel result back to host side */
	CHECK (cudaMemcpy (host_matrix, device_matrix, DIM * sizeof(int[DIM]), cudaMemcpyDeviceToHost));

	/* free device global memory */
	CHECK (cudaFree (device_matrix));
	CHECK (cudaFree (device_matrix));

	/* reset the device */
	CHECK (cudaDeviceReset ());

	// validate if the array is sorted correctly
	validateSort(host_matrix, DIM*DIM);
	free(host_matrix);
	return 0;
}

/**
 *  \brief Get delta time.
 *
 *  This function measures the elapsed time between successive calls.
 *
 *  \return The time elapsed between successive calls in seconds.
 *
 *  The function uses the CLOCK_MONOTONIC clock to measure time.
 */
static double get_delta_time(void)
{
	static struct timespec t0,t1;

	t0 = t1;
	if(clock_gettime(CLOCK_MONOTONIC,&t1) != 0)
	{
		perror("clock_gettime");
		exit(1);
	}
	return (double)(t1.tv_sec - t0.tv_sec) + 1.0e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
}

/**
 *  \brief Validate Sort.
 *
 *  This function checks if an array is sorted in ascending order.
 *
 *  \param arr: pointer to the array to be validated
 *  \param N: size of the array
 *
 */
int validateSort(int *arr, int N) {
    int i;

    for (i = 0; i < N - 1; i++)
    {
        if (arr[i] > arr[i + 1])
        {
            printf("Error in position %d between element %d and %d\n", i, arr[i], arr[i + 1]);
            return 0;
        }
    }
	if (i == (N - 1))
		printf("Everything is OK!\n");

    return 1;
}