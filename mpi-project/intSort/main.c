#include "intSort.h"
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>

#define DEFAULT_NUM_WORKERS 8
#define DISTRIBUTOR_RANK 0

void mergeSortedSubsequences(int *arr, int num_workers, int size);

void printUsage(char *cmdName);

int main(int argc, char *argv[]) {
    int num_workers = DEFAULT_NUM_WORKERS;
    char *filename = NULL;

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *arr = NULL;
    int arr_size = 0;

    if (rank == DISTRIBUTOR_RANK) {
        // Parse command line arguments
        int c;
        while ((c = getopt(argc, argv, "t:f:h")) != -1) {
            switch (c) {
            case 't':
                num_workers = atoi(optarg);
                break;
            case 'f':
                filename = optarg;
                break;
            case 'h':
                printUsage(argv[0]);
                return 0;
            case '?':
                if (optopt == 'f' || optopt == 't') {
                    fprintf(stderr, "Option -%c requires an argument.\n", optopt);
                } else {
                    printUsage(argv[0]);
                }
                return 1;
            default:
                abort();
            }
        }

        if (filename == NULL) {
            printUsage(argv[0]);
            return 1;
        }

        // Read data
        FILE *file = fopen(filename, "rb");

        if (file == NULL) {
            printf("Error opening file\n");
            return 1;
        }

        fseek(file, 0, SEEK_END);
        arr_size = ftell(file) / sizeof(int);
        fseek(file, 0, SEEK_SET);

        arr = malloc(arr_size * sizeof(int));
        if (arr == NULL) {
            printf("Error: cannot allocate memory\n");
            return 1;
        }

        int count = fread(arr, sizeof(int), arr_size, file);

        if (count != arr_size) {
            printf("Error: could not read all integers from file\n");
            return 1;
        }

        fclose(file);
    }

    // Distribute data size
    MPI_Bcast(&arr_size, 1, MPI_INT, DISTRIBUTOR_RANK, MPI_COMM_WORLD);

    int chunk_size = arr_size / size;
    int *local_arr = malloc(chunk_size * sizeof(int));

    // Distribute data
    MPI_Scatter(arr, chunk_size, MPI_INT, local_arr, chunk_size, MPI_INT, DISTRIBUTOR_RANK, MPI_COMM_WORLD);

    // Perform local sort
    mergeSort(local_arr, 0, chunk_size - 1);

    // Gather sorted subsequences
    MPI_Gather(local_arr, chunk_size, MPI_INT, arr, chunk_size, MPI_INT, DISTRIBUTOR_RANK, MPI_COMM_WORLD);

    // Merge sorted subsequences
    if (rank == DISTRIBUTOR_RANK) {
        mergeSortedSubsequences(arr, size, arr_size);

        if (validateSort(arr, arr_size))
            printf("The array is sorted correctly.\n");
        else
            printf("The array is not sorted correctly.\n");

        //printArray(arr, arr_size);

        free(arr);
    }

    free(local_arr);

    MPI_Finalize();

    return 0;
}

void mergeSortedSubsequences(int *arr, int num_workers, int size) {
    int chunk_size = size / num_workers;
    int *temp = (int *)malloc(size * sizeof(int));

    int *indexes = (int *)calloc(num_workers, sizeof(int));
    for (int i = 0; i < num_workers; ++i) {
        indexes[i] = i * chunk_size;
    }

    int *end_indexes = (int *)calloc(num_workers, sizeof(int));
    for (int i = 0; i < num_workers; ++i) {
        end_indexes[i] = (i == num_workers - 1) ? size : (i + 1) * chunk_size;
    }

    for (int i = 0; i < size; ++i) {
        int min_idx = -1;
        int min_val = INT_MAX;

        for (int j = 0; j < num_workers; ++j) {
            if (indexes[j] < end_indexes[j] && arr[indexes[j]] < min_val) {
                min_val = arr[indexes[j]];
                min_idx = j;
            }
        }

        indexes[min_idx]++;
        temp[i] = min_val;
    }

    memcpy(arr, temp, size * sizeof(int));
    free(temp);
    free(indexes);
    free(end_indexes);
}

void printUsage(char *cmdName) {
    fprintf(stderr, "\nSynopsis: %s [OPTIONS]\n"
                    "  OPTIONS:\n"
                    "  -t nThreads  --- set the number of threads to be created (default: 8)\n"
                    "  -f file --- set the file to be processed\n"
                    "  -h           --- print this help\n",
            cmdName);
}
