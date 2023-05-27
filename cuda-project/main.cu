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
# define MATRIX_DIMENSION 1024
#endif
#ifndef SECTOR_SIZE
# define SECTOR_SIZE  1024
#endif
#ifndef N_SECTORS
# define N_SECTORS (MATRIX_DIMENSION * MATRIX_DIMENSION)
#endif

/* allusion to internal functions */

static double get_delta_time(void);

int validateSort(unsigned int *arr, int N);

__device__ void merge(unsigned int *data, int start, int mid, int end, unsigned int *temp);

__global__ void mergeSortRowKernel(unsigned int *data, int N, int iter, unsigned int *temp);

__global__ void mergeSortColumnKernel(unsigned int *data, int N, int iter, unsigned int *temp);


__device__ void merge(unsigned int *data, int start, int mid, int end, unsigned int *temp) {
  int i = start;
  int j = mid;
  for (int k = start; k < end; k++) {
      temp[k] = (i<mid && (j>=end || data[i] <= data[j])) ? data[i++] : data[j++];
  }
  for (int k = start; k < end; k++) {
      data[k] = temp[k];
  }
}

__global__ void mergeSortRowKernel(unsigned int *data, int N, int iter, unsigned int *temp) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = blockDim.x * gridDim.x * y + x;

    if (idx < (N >> iter)) {
        int start = N * (1 << iter) * idx;
        int mid = start + (1 << iter) * N / 2;
        int end = start + (1 << iter) * N;
        merge(data, start, mid, end, temp);
    }
}

__global__ void mergeSortColumnKernel(unsigned int *data, int N, int iter, unsigned int *temp) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = blockDim.x * gridDim.x * y + x;

    if (idx < (N >> iter)) {
        int start = (1 << iter) * idx;
        int mid = start + ((1 << iter) * N / 2) % N + ((1 << iter) * N / 2) / N;
        int end = start + (1 << iter) * N;
        merge(data, start, mid, end, temp);
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

    int *arr = (int*) malloc(size * sizeof(int));
    if (arr == NULL) {
        printf("Error: cannot allocate memory\n");
        return 1;
    }

    int count = fread(arr, sizeof(int), size, file);

    if (count != size) {
        printf("Error: could not read all integers from file\n");
        return 1;
    }

    fclose(file);

  // allocate memory for a 1024x1024 matrix on the host side
  unsigned int *host_matrix = (unsigned int *)malloc(MATRIX_DIMENSION * MATRIX_DIMENSION * sizeof(unsigned int));

  // Convert the 1D array into a 2D matrix
  for (int i = 0; i < MATRIX_DIMENSION; i++) {
    for (int j = 0; j < MATRIX_DIMENSION; j++) {
        host_matrix[i * MATRIX_DIMENSION + j] = arr[i * MATRIX_DIMENSION + j];
    }
  }

	/* set up the device */
	int dev = 0;

	cudaDeviceProp deviceProp;
	CHECK (cudaGetDeviceProperties (&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK (cudaSetDevice (dev));

  /* copy the host data to the device memory */
  unsigned int *device_matrix;
  (void) get_delta_time ();
  CHECK(cudaMalloc((void**)&device_matrix, MATRIX_DIMENSION * MATRIX_DIMENSION * sizeof(unsigned int)));
  CHECK(cudaMemcpy(device_matrix, host_matrix, MATRIX_DIMENSION * sizeof(unsigned int[MATRIX_DIMENSION]), cudaMemcpyHostToDevice));
  printf ("The transfer of %ld bytes from the host to the device took %.3e seconds\n",
		  MATRIX_DIMENSION * sizeof(unsigned int[MATRIX_DIMENSION]), get_delta_time ());

  /* run the computational kernel
	 as an example, N_SECTORS threads are launched where each thread deals with one sector */

    unsigned int gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ;
    int n_sectors;

    n_sectors = N_SECTORS;

    // Number of threads in each dimension of a block
    blockDimX = 1 << 5;                                             // optimize!
    blockDimY = 1 << 5;                                             // optimize!
    blockDimZ = 1 << 0;                                             // do not change!

    // Number of blocks in each dimension of the grid
    gridDimX = MATRIX_DIMENSION / blockDimX;
    gridDimY = MATRIX_DIMENSION / blockDimY;
    gridDimZ = 1 << 0;                                              // do not change!

    dim3 grid (gridDimX, gridDimY, gridDimZ);
    dim3 block (blockDimX, blockDimY, blockDimZ);

    if ((gridDimX * gridDimY * gridDimZ * blockDimX * blockDimY * blockDimZ) != n_sectors)
    { printf ("Wrong configuration!\n");
      return 1;
    }
  (void) get_delta_time ();

  // Initialize temporary array
  unsigned int *device_temp;
  CHECK(cudaMalloc((void**)&device_temp, MATRIX_DIMENSION * sizeof(unsigned int[MATRIX_DIMENSION])));

  // Perform merge sort
  for (int iter = 1, size = 1024; iter <= 10; iter++, size *= 2) {
    printf("Iteration: %d, subseq_num: %d, subseq_len: %d\n", iter, MATRIX_DIMENSION*MATRIX_DIMENSION/size, size);
    mergeSortRowKernel<<<grid, block>>>(device_matrix, size, size*2, device_temp);
    CHECK (cudaDeviceSynchronize ());                            // wait for kernel to finish
    CHECK (cudaGetLastError ());                                 // check for kernel errors
  }

  printf("The CUDA kernel <<<(%d,%d,%d), (%d,%d,%d)>>> took %.3e seconds to run\n",
		 gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, get_delta_time ());

  /* copy kernel result back to host side */

  CHECK (cudaMemcpy (host_matrix, device_matrix, MATRIX_DIMENSION * sizeof(unsigned int[MATRIX_DIMENSION]), cudaMemcpyDeviceToHost));
  printf ("The transfer of %ld bytes from the device to the host took %.3e seconds\n",
		  (long) MATRIX_DIMENSION * sizeof(unsigned int[MATRIX_DIMENSION]), get_delta_time ());

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

int validateSort(unsigned int *arr, int N)
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