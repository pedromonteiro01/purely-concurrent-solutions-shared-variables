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

#ifndef MATRIX_DIMENSION
# define MATRIX_DIMENSION 16
#endif

/* allusion to internal functions */

static double get_delta_time(void);

__device__ void merge(int *data, int start, int mid, int end, int *temp, int idx);

__global__ void mergeSortRowKernel(int *data, int N, int iter, int *temp);

/** \brief check if the array of integers has been sorted correctly */
int validateSort(int *arr, int N);


__device__ void merge(int *data, int start, int mid, int end, int *temp, int idx) {
  int i = start;
  int j = mid;

  if (idx == 0) {
    printf("\n\n-------------DEBUG MERGE-------------\n\n");
    for (int i = start; i < end; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");
  }

  for (int k = start; k < end; k++) {
    if (idx == 0)
      printf("k: %d, i: %d, j: %d, end: %d", k, i, j, end);
    temp[k] = (i < mid && (j >= end || data[i] <= data[j])) ? data[i++] : data[j++];
    if (idx == 0)
      printf(" temp[k]: %d\n", temp[k]);
  }
  for (int k = start; k < end; k++) {
      data[k] = temp[k];
  }
  if (idx == 0)
    printf("\n\n-------------END DEBUG MERGE-------------\n\n");
}


__global__ void mergeRowKernel(int *data, int iter, int *temp) {
    int N = MATRIX_DIMENSION;
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = blockDim.x * gridDim.x * y + x;

    if(idx >= (N >> iter)) return;
    
    int start = N * (1 << iter) * idx;
    int mid = start + (1 << iter) * N / 2;
    int end = start + (1 << iter) * N;
    merge(data, start, mid, end, temp, idx);
    
    if (iter == 0)
        printf("idx: %d, start: %d, mid: %d, end: %d, iter: %d, N: %d\n", idx, start, mid, end, iter, N);
    
    if (idx == 0) {
      //print array
      printf("array sorted by thread %d:\n", idx);
      for (int i = start; i < end; i++) {
          printf("%d ", data[i]);
      }
      printf("\n");
  }
}


__global__ void mergeSortRowKernel(int *data, int iter, int *temp) {
    int N = MATRIX_DIMENSION;
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = blockDim.x * gridDim.x * y + x;

    if(idx >= (N >> iter)) return;
    
    for (int curr_size=1; curr_size<=MATRIX_DIMENSION; curr_size = 2*curr_size) {
      int start = idx*MATRIX_DIMENSION;
      int mid = min(start + curr_size/2, MATRIX_DIMENSION*MATRIX_DIMENSION-1);
      int end = min(start + curr_size, MATRIX_DIMENSION*MATRIX_DIMENSION-1);

      if (idx == 0) {
        printf("\n\n-------------DEBUG MERGESORT-------------\n\nidx: %d, start: %d, mid: %d, end: %d, iter: %d, N: %d\n", idx, start, mid, end, iter, N);
        printf("array for thread %d:\n", idx);
        for (int i = start; i < end; i++) {
            printf("%d ", data[i]);
        }
        printf("\n\n-------------END DEBUG MERGESORT-------------\n\n");
      }
      merge(data, start, mid, end, temp, idx);
    }
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
  (void) get_delta_time ();
  CHECK(cudaMalloc((void**)&device_matrix, MATRIX_DIMENSION * MATRIX_DIMENSION * sizeof(int)));
  CHECK(cudaMemcpy(device_matrix, host_matrix, MATRIX_DIMENSION * sizeof(int[MATRIX_DIMENSION]), cudaMemcpyHostToDevice));
  printf ("The transfer of %ld bytes from the host to the device took %.3e seconds\n",
		  MATRIX_DIMENSION * sizeof(int[MATRIX_DIMENSION]), get_delta_time ());

  /* run the computational kernel
	 as an example, MATRIX_DIMENSION threads are launched where each thread deals with one subsequence */

    int gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ;

    // Number of threads in each dimension of a block
    blockDimX = 1 << 2;                                             // optimize!
    blockDimY = 1 << 0;                                             // optimize!
    blockDimZ = 1 << 0;                                             // do not change!

    // Number of blocks in each dimension of the grid
    gridDimX = MATRIX_DIMENSION / blockDimX;
    gridDimY = 1 << 0;
    gridDimZ = 1 << 0;                                              // do not change!

    dim3 grid (gridDimX, gridDimY, gridDimZ);
    dim3 block (blockDimX, blockDimY, blockDimZ);

    if ((gridDimX * gridDimY * gridDimZ * blockDimX * blockDimY * blockDimZ) != MATRIX_DIMENSION)
    { printf ("Wrong configuration!\n");
      return 1;
    }
  (void) get_delta_time ();

  // Initialize temporary array
  int *device_temp;
  CHECK(cudaMalloc((void**)&device_temp, MATRIX_DIMENSION * sizeof(int[MATRIX_DIMENSION])));

  // Perform merge sort
  for (int iter = 0; iter < 10; iter++) {
    mergeSortRowKernel<<<grid, block>>>(device_matrix, iter, device_temp);
    CHECK (cudaDeviceSynchronize ());                            // wait for kernel to finish
    CHECK (cudaGetLastError ());                                 // check for kernel errors
    break;
  }

  printf("The CUDA kernel <<<(%d,%d,%d), (%d,%d,%d)>>> took %.3e seconds to run\n",
		 gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, get_delta_time ());

  /* copy kernel result back to host side */
  CHECK (cudaMemcpy (host_matrix, device_matrix, MATRIX_DIMENSION * sizeof(int[MATRIX_DIMENSION]), cudaMemcpyDeviceToHost));
  printf ("The transfer of %ld bytes from the device to the host took %.3e seconds\n",
		  (long) MATRIX_DIMENSION * sizeof(int[MATRIX_DIMENSION]), get_delta_time ());

  /* free device global memory */
  CHECK (cudaFree (device_matrix));
  CHECK(cudaFree(device_temp));

  /* reset the device */
  CHECK (cudaDeviceReset ());


  // validate if the array is sorted correctly
  validateSort(host_matrix, size);

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