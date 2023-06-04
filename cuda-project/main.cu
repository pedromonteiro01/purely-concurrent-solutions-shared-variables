/**
 *   José Trigo
 *   Pedro Monteiro
 *   May 2023
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
# define DIM 8
#endif

/* allusion to internal functions */

static double get_delta_time(void);

__device__ void merge(int arr[], int temp[], int left, int mid, int right);
__device__ void mergeSort(int arr[], int temp[], int size);
__global__ void processor(int *data, int iter, int *temp);


/** \brief check if the array of integers has been sorted correctly */
int validateSort(int *arr, int N);


__device__ void merge(int arr[], int temp[], int left, int mid, int right) {
	int i = left;
	int j = mid;
	int k = left;

	while (i < mid && j <= right) {
		if (arr[i] <= arr[j]) {
			temp[k] = arr[i];
			i++;
		} else {
			temp[k] = arr[j];
			j++;
		}
		k++;
	}

	while (i < mid) {
		temp[k] = arr[i];
		i++;
		k++;
	}

	while (j <= right) {
		temp[k] = arr[j];
		j++;
		k++;
	}

	for (i = left; i <= right; i++) {
		arr[i] = temp[i];
	}
}

__device__ void mergeSort(int arr[], int temp[], int size) {
	int mid, i;

	for (mid = 1; mid < size; mid *= 2) {
		for (i = 0; i < size; i += 2 * mid) {
			int end = i + 2 * mid - 1;
			if (end >= size) {
				end = size - 1;
			}
			//printf("start: %d, mid: %d, end: %d\n", i, i + mid, end);
			merge(arr, temp, i, i + mid, end);
		}
	}
}

__global__ void processor(int *data, int iter, int *temp) {
	int N = DIM;
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int idx = blockDim.x * gridDim.x * y + x;

	if(idx >= (N >> iter)) return;

	int start = N * (1 << iter) * idx;
	//int mid = start + (1 << iter) * N / 2;
	int end = start + (1 << iter) * N;
	mergeSort(data, temp, end);

	/*
	printf("iter: %d, thread: %d, start: %d, end: %d\n", iter, idx, start, end);
	if (idx == 3) {
		printf("sorted array for thread %d:\n", idx);
		//print array
		for (int i = start; i < end; i++) {
			printf("%d ", data[i]);
		}
		printf("\n");
	}
	*/
}


/**
 *   main program
 */

int main (int argc, char **argv)
{
	printf("%s Starting...\n", argv[0]);

	if (argc != 2) {
		printf("Usage: %s <filename>\n", argv[0]);
		return 1;
	}

	// Open the file for reading
	/*
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
	*/

	// generate array of 64 random integers
	int *host_matrix = (int*) malloc(DIM*DIM * sizeof(int));
	srand(time(NULL));
	for (int i = 0; i < 64; i++)
		host_matrix[i] = rand() % 100;

	/* set up the device */
	int dev = 0;

	cudaDeviceProp deviceProp;
	CHECK (cudaGetDeviceProperties (&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK (cudaSetDevice (dev));

  /* copy the host data to the device memory */
  int *device_matrix;
  (void) get_delta_time ();
  CHECK(cudaMalloc((void**)&device_matrix, DIM * DIM * sizeof(int)));
  CHECK(cudaMemcpy(device_matrix, host_matrix, DIM * sizeof(int[DIM]), cudaMemcpyHostToDevice));
  printf ("The transfer of %ld bytes from the host to the device took %.3e seconds\n",
		  DIM * sizeof(int[DIM]), get_delta_time ());

  /* run the computational kernel
	 as an example, DIM threads are launched where each thread deals with one subsequence */

	int gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ;

	// Number of threads in each dimension of a block
	blockDimX = 1 << 3;                                             // optimize!
	blockDimY = 1 << 0;                                             // optimize!
	blockDimZ = 1 << 0;                                             // do not change!

	// Number of blocks in each dimension of the grid
	gridDimX = DIM / blockDimX;
	gridDimY = 1 << 0;
	gridDimZ = 1 << 0;                                              // do not change!

	dim3 grid (gridDimX, gridDimY, gridDimZ);
	dim3 block (blockDimX, blockDimY, blockDimZ);

	if ((gridDimX * gridDimY * gridDimZ * blockDimX * blockDimY * blockDimZ) != DIM)
	{ printf ("Wrong configuration!\n");
	  return 1;
	}
  (void) get_delta_time ();

  // Initialize temporary array
  int *device_temp;
  CHECK(cudaMalloc((void**)&device_temp, DIM * sizeof(int[DIM])));

	// Perform merge sort
	for (int iter = 0; iter < 3; iter++) {  // Adjusted iteration count to 3
		// Adjust block and grid dimensions for each iteration
		blockDimX = DIM / (1 << (iter + 1));  // Divides by 2 each iteration
		gridDimX = 1 << (iter + 1);  // Multiplies by 2 each iteration

		dim3 grid (gridDimX, gridDimY, gridDimZ);
		dim3 block (blockDimX, blockDimY, blockDimZ);

		if ((gridDimX * gridDimY * gridDimZ * blockDimX * blockDimY * blockDimZ) != DIM)
		{ printf ("Wrong configuration!\n");
		return 1;
		}

		processor<<<grid, block>>>(device_matrix, iter, device_temp);
		CHECK (cudaDeviceSynchronize ());                            // wait for kernel to finish
		CHECK (cudaGetLastError ());                                 // check for kernel errors
	}

  printf("The CUDA kernel <<<(%d,%d,%d), (%d,%d,%d)>>> took %.3e seconds to run\n",
		 gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, get_delta_time ());

  /* copy kernel result back to host side */
  CHECK (cudaMemcpy (host_matrix, device_matrix, DIM * sizeof(int[DIM]), cudaMemcpyDeviceToHost));
  printf ("The transfer of %ld bytes from the device to the host took %.3e seconds\n",
		  (long) DIM * sizeof(int[DIM]), get_delta_time ());

  /* free device global memory */
  CHECK (cudaFree (device_matrix));
  CHECK(cudaFree(device_temp));

  /* reset the device */
  CHECK (cudaDeviceReset ());

	//print array
	//for (int i = 0; i < DIM*DIM; i++) {
	//	printf("%d ", host_matrix[i]);
	//}

  // validate if the array is sorted correctly
  validateSort(host_matrix, DIM*DIM);

  return 0;
}

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

int validateSort(int *arr, int N)
{
	int i;

	for (i = 0; i < N - 1; i++)
	{
		if (arr[i] > arr[i + 1])
		{
			printf("Error in position %d between element %d and %d\n", i, arr[i], arr[i + 1]);
			return 0;
		}
		if (i == (N - 1))
			printf("Everything is OK!\n");
	}

	return 1;
}