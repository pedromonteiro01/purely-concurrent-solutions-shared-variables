#include "intSort.h"
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>

#define DISTRIBUTOR_RANK 0

#define MPI_DEBUG() {                  \
    int i = 0;                         \
    char hostname[256];                \
    gethostname(hostname, sizeof(hostname));\
    printf("PID %d on %s ready for attach\n", getpid(), hostname);\
    fflush(stdout);                    \
    while (!i)                         \
        if (fopen("continue.txt", "r"))\
            i = 1;                     \
}


void mergeSortedSubsequences(int *arr, int num_workers, int size, int *send_counts, int *displs);

static void printUsage(char *cmdName);

int main(int argc, char *argv[])
{
    MPI_Init(NULL, NULL);
    //MPI_DEBUG();

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num_workers = size;
    char *filename = NULL;
    int c;
    while ((c = getopt(argc, argv, "t:f:h")) != -1)
    {
        switch (c)
        {
        case 'f':
            filename = optarg;
            break;
        case 'h':
            printUsage(argv[0]);
            return 0;
        case '?':
            if (optopt == 'f' || optopt == 't')
            {
                fprintf(stderr, "Option -%c requires an argument.\n", optopt);
            }
            else
            {
                printUsage(argv[0]);
            }
            return 1;
        default:
            abort();
        }
    }

    if (filename == NULL)
    {
        printUsage(argv[0]);
        return 1;
    }

    int *arr = NULL;
    int arr_size = 0;

    if (rank == DISTRIBUTOR_RANK)
    {
        FILE *file = fopen(filename, "rb");
        if (file == NULL)
        {
            printf("Error opening file\n");
            return 1;
        }

        fseek(file, 0, SEEK_END);
        arr_size = ftell(file) / sizeof(int);
        fseek(file, 0, SEEK_SET);

        arr = malloc(arr_size * sizeof(int));
        if (arr == NULL)
        {
            printf("Error: cannot allocate memory\n");
            return 1;
        }

        int count = fread(arr, sizeof(int), arr_size, file);
        if (count != arr_size)
        {
            printf("Error: could not read all integers from file\n");
            return 1;
        }

        fclose(file);
    }

    int result;
    result = MPI_Bcast(&arr_size, 1, MPI_INT, DISTRIBUTOR_RANK, MPI_COMM_WORLD);
    if (result != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI_Bcast failed\n");
        return 1;
    }

    int chunk_size = arr_size / size;
    int remainder = arr_size % size;
    int local_arr_size = chunk_size + (rank < remainder ? 1 : 0);

    int *local_arr = malloc(local_arr_size * sizeof(int));
    if (local_arr == NULL)
    {
        printf("Error: cannot allocate memory\n");
        return 1;
    }

    int *send_counts = NULL;
    int *displs = NULL;

    if (rank == DISTRIBUTOR_RANK)
    {
        send_counts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        if (send_counts == NULL || displs == NULL)
        {
            printf("Error: cannot allocate memory\n");
            return 1;
        }
        for (int i = 0; i < size; ++i)
        {
            send_counts[i] = chunk_size + (i < remainder ? 1 : 0);
            displs[i] = (i * chunk_size) + (i < remainder ? i : remainder);
        }
    }

    result = MPI_Scatterv(arr, send_counts, displs, MPI_INT, local_arr, local_arr_size, MPI_INT, DISTRIBUTOR_RANK, MPI_COMM_WORLD);
    if (result != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI_Scatterv failed\n");
        return 1;
    }

    mergeSort(local_arr, 0, local_arr_size - 1);

    if (rank == DISTRIBUTOR_RANK)
    {
        arr = realloc(arr, arr_size * sizeof(int));
        if (arr == NULL)
        {
            printf("Error: cannot allocate memory\n");
            return 1;
        }
    }

    result = MPI_Gatherv(local_arr, local_arr_size, MPI_INT, arr, send_counts, displs, MPI_INT, DISTRIBUTOR_RANK, MPI_COMM_WORLD);
    if (result != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI_Gatherv failed\n");
        return 1;
    }

    if (rank == DISTRIBUTOR_RANK)
    {
        mergeSortedSubsequences(arr, num_workers, arr_size, send_counts, displs);

        if (validateSort(arr, arr_size))
            printf("The array is sorted correctly.\n");
        else
            printf("The array is not sorted correctly.\n");

        // printArray(arr, arr_size);

        free(arr);
        free(send_counts);
        free(displs);
    }

    free(local_arr);

    MPI_Finalize();

    return 0;
}

void mergeSortedSubsequences(int *arr, int num_workers, int size, int *send_counts, int *displs)
{
    int *temp = (int *)malloc(size * sizeof(int));
    int *indexes = (int *)calloc(num_workers, sizeof(int));
    int *end_indexes = (int *)calloc(num_workers, sizeof(int));

    if (temp == NULL || indexes == NULL || end_indexes == NULL)
    {
        fprintf(stderr, "Failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_workers; ++i)
    {
        indexes[i] = displs[i];
        end_indexes[i] = displs[i] + send_counts[i];
    }

    for (int i = 0; i < size; ++i)
    {
        int min_idx = -1;
        int min_val = INT_MAX;

        for (int j = 0; j < num_workers; ++j)
        {
            if (indexes[j] < end_indexes[j] && arr[indexes[j]] < min_val)
            {
                min_val = arr[indexes[j]];
                min_idx = j;
            }
        }

        if (min_idx == -1)
        {
            fprintf(stderr, "Failed to find minimum value\n");
            free(temp);
            free(indexes);
            free(end_indexes);
            exit(EXIT_FAILURE);
        }

        indexes[min_idx]++;
        temp[i] = min_val;
    }

    memcpy(arr, temp, size * sizeof(int));

    free(temp);
    free(indexes);
    free(end_indexes);
}

void printUsage(char *cmdName)
{
    fprintf(stderr, "\nSynopsis: %s [OPTIONS]\n"
                    "  OPTIONS:\n"
                    "  -f file --- set the file to be processed\n"
                    "  -h           --- print this help\n",
            cmdName);
}