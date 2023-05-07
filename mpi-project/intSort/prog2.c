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

// Define the distributor rank.
#define DISTRIBUTOR_RANK 0

/** \brief print command usage */
static void printUsage(char *cmdName);

/** \brief merge more than one sorted subsequences of an array into a single sorted sequence*/
void mergeSortedSubsequences(int *arr, int num_workers, int size, int *send_counts, int *displs);

/**
 *  \brief Main Function.
 *
 *  Its role is to sort an array of integers in parallel using the Merge Sort algorithm with MPI
 *
 *  \param argc number of words of the command line
 *  \param argv list of words of the command line
 *
 *  \return status of operation
 */

int main(int argc, char *argv[])
{
    MPI_Init(NULL, NULL);

    int size, rank;

    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Set the number of worker processes
    int num_workers = size;
    
    char *filename = NULL;

    // Parse command line options
    int c;
    while ((c = getopt(argc, argv, "t:f:h")) != -1)
    {
        switch (c)
        {
        case 'f': // Set the filename
            filename = optarg;
            break;
        case 'h': // Print usage information and exit
            printUsage(argv[0]);
            return 0;
        case '?': // Handle missing option arguments or invalid option
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

    // Read input data from file if current process is the distributor
    if (rank == DISTRIBUTOR_RANK)
    {
        FILE *file = fopen(filename, "rb");
        if (file == NULL)
        {
            printf("Error opening file\n");
            return 1;
        }

        // Determine the size of the file
        fseek(file, 0, SEEK_END);
        arr_size = ftell(file) / sizeof(int);
        fseek(file, 0, SEEK_SET);

        // Allocate memory for the array
        arr = malloc(arr_size * sizeof(int));
        if (arr == NULL)
        {
            printf("Error: cannot allocate memory\n");
            return 1;
        }

        // Read integers from the file
        int count = fread(arr, sizeof(int), arr_size, file);
        if (count != arr_size)
        {
            printf("Error: could not read all integers from file\n");
            return 1;
        }

        fclose(file);
    }

    // Broadcast array size to all processes
    int result;
    result = MPI_Bcast(&arr_size, 1, MPI_INT, DISTRIBUTOR_RANK, MPI_COMM_WORLD);
    if (result != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI_Bcast failed\n");
        return 1;
    }

    // Calculate chunk size and local array size for each process
    int chunk_size = arr_size / size;
    int remainder = arr_size % size;
    int local_arr_size = chunk_size + (rank < remainder ? 1 : 0);

    // Allocate memory for local array
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
        // Allocate memory for send_counts and displs arrays
        send_counts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        if (send_counts == NULL || displs == NULL)
        {
            printf("Error: cannot allocate memory\n");
            return 1;
        }

        // Calculate send_counts and displs for each process
        for (int i = 0; i < size; ++i)
        {
            send_counts[i] = chunk_size + (i < remainder ? 1 : 0);
            displs[i] = (i * chunk_size) + (i < remainder ? i : remainder);
        }
    }

    // Distribute the array chunks to all processes using Scatterv
    result = MPI_Scatterv(arr, send_counts, displs, MPI_INT, local_arr, local_arr_size, MPI_INT, DISTRIBUTOR_RANK, MPI_COMM_WORLD);
    if (result != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI_Scatterv failed\n");
        return 1;
    }

    // Each process sorts its local array using merge sort
    mergeSort(local_arr, 0, local_arr_size - 1);

    // Reallocate memory for the array in the distributor process before gathering the sorted subsequences
    if (rank == DISTRIBUTOR_RANK)
    {
        arr = realloc(arr, arr_size * sizeof(int));
        if (arr == NULL)
        {
            printf("Error: cannot allocate memory\n");
            return 1;
        }
    }

    // Gather sorted local arrays from all processes to the distributor process using Gatherv
    result = MPI_Gatherv(local_arr, local_arr_size, MPI_INT, arr, send_counts, displs, MPI_INT, DISTRIBUTOR_RANK, MPI_COMM_WORLD);
    if (result != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI_Gatherv failed\n");
        return 1;
    }

    // Merge sorted subsequences in the distributor process
    if (rank == DISTRIBUTOR_RANK)
    {
        mergeSortedSubsequences(arr, num_workers, arr_size, send_counts, displs);

        // Check if the final array is sorted correctly
        if (validateSort(arr, arr_size))
            printf("The array is sorted correctly.\n");
        else
            printf("The array is not sorted correctly.\n");

        // Free memory for arrays used in the distributor process
        free(arr);
        free(send_counts);
        free(displs);
    }

    free(local_arr);

    MPI_Finalize();

    return 0;
}

/**
 *  \brief Merge sorted sub arrays.
 *
 *  The purpose of this function is to merge num_workers sorted subsequences of arr into a single sorted sequence.
 *
 *  \param arr pointer to an array of integers that contains the sorted subsequences to be merged
 *  \param num_workers  integer that represents the number of workers involved
 *  \param size integer that represents the total size of the input array
 *  \param send_counts pointer to an array that represents the number of elements to be sent by each worker
 *  \param displs pointer to an array of integers that represents the displacement for each worker's send buffer
 */

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

/**
 *  \brief Print command usage.
 *
 *  A message specifying how the program should be called is printed.
 *
 *  \param cmdName string with the name of the command
 */

void printUsage(char *cmdName)
{
    fprintf(stderr, "\nSynopsis: %s [OPTIONS]\n"
                    "  OPTIONS:\n"
                    "  -f file --- set the file to be processed\n"
                    "  -h           --- print this help\n",
            cmdName);
}