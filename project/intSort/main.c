/**
 *  \file main.c (implementation file)
 *
 *  \brief Problem name: Integer Sort.
 *
 * Implementation of a parallel merge sort algorithm for sorting an array of integers in 
 * ascending order using multiple threads. It takes a file of integers as input, creates a 
 * specified number of worker threads to sort different chunks of the array in parallel, and 
 * then merges the results into a single sorted array.
 *  \authors Pedro Monteiro & José Trigo - March 2023
 */

#include "intSort.h"
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define DEFAULT_NUM_WORKERS 8

/** \brief sort an array of integers in ascending order */
static void *parallel_merge_sort(void *args);

/** \brief default number of worker threads */
int NUM_WORKERS = DEFAULT_NUM_WORKERS;

/** \brief print command usage */
static void printUsage(char *cmdName);

/**
 *  \brief Struct to pass arguments to worker threads.
 *
 *  Its role is to store the arguments for each worker thread.
 *
 *  \param id unique identifier of the worker thread
 *  \param chunks pointer to chunks array
 *  \param total_chunks variable with the total number of chunks
 *  \param total_words pointer to an integer that holds the total number of words
 *  \param vowel_count  pointer to an integer array that holds the count of vowels
 *  \param mutex pointer to a pthread_mutex_t used for synchronization between threads
 */
typedef struct WorkerArgs
{
    int id;
    int *arr;
    int left;
    int right;
} WorkerArgs;


/**
 *  \brief Function parallel_merge_sort.
 *
 *  Its role is to sort an array of integers in ascending order.
 *
 *  \param arr: a pointer to the array to be sorted
 *  \param left: an integer representing the starting index of the subarray to be sorted
 *  \param right: an integer representing the ending index of the subarray to be sorted
 */

void *parallel_merge_sort(void *args)
{
    WorkerArgs *worker_args = (WorkerArgs *)args;
    // int worker_id = worker_args->id;
    int *arr = worker_args->arr;
    int left = worker_args->left;
    int right = worker_args->right;

    if (left < right)
    {
        int middle = left + (right - left) / 2;
        // printf("Worker %d sorting array -> left: %d, middle: %d, right: %d\n", worker_id, left, middle, right);
        mergeSort(arr, left, middle);
        mergeSort(arr, middle + 1, right);
        merge(arr, left, middle, right);
    }

    return NULL;
}

/**
 *  \brief Main thread.
 *
 *  Its role is starting the simulation by generating the intervening entities threads (producers and consumers) and
 *  waiting for their termination.
 *
 *  \param argc number of words of the command line
 *  \param argv list of words of the command line
 *
 *  \return status of operation
 */

int main(int argc, char *argv[])
{
    int *arr, size, count;
    int num_workers = NUM_WORKERS;
    char *filename = NULL;

    int c;
    while ((c = getopt(argc, argv, "t:f:h")) != -1)
    {
        switch (c)
        {
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

    FILE *file = fopen(filename, "rb");
    if (file == NULL) // check if file is empty
    {
        printf("Error opening file\n");
        return 1;
    }

    fseek(file, 0, SEEK_END);
    size = ftell(file) / sizeof(int);
    fseek(file, 0, SEEK_SET);

    arr = malloc(size * sizeof(int));

    if (arr == NULL)
    {
        printf("Error: cannot allocate memory\n");
        return 1;
    }

    count = fread(arr, sizeof(int), size, file);

    if (count != size)
    {
        printf("Error: could not read all integers from file\n");
        return 1;
    }

    fclose(file);

    // Create worker threads.
    pthread_t worker_threads[num_workers];
    WorkerArgs worker_args[num_workers];

    // Calculate the chunk size for each worker thread
    int chunk_size = size / num_workers;

    printf("using %d threads\n", num_workers);
    printf("processing file %s\n", filename);

    // Assign work to worker threads
    for (int i = 0; i < num_workers; i++)
    {
        worker_args[i].id = i;
        worker_args[i].arr = arr;
        worker_args[i].left = i * chunk_size;
        worker_args[i].right = (i == num_workers - 1) ? size - 1 : (i + 1) * chunk_size - 1;
        pthread_create(&worker_threads[i], NULL, parallel_merge_sort, &worker_args[i]);
    }

    // Wait for worker threads to finish
    for (int i = 0; i < num_workers; i++)
        pthread_join(worker_threads[i], NULL);

    // Merge sorted chunks
    for (int i = 1; i < num_workers; i++)
        merge(arr, 0, (i * chunk_size) - 1, (i == num_workers - 1) ? size - 1 : (i + 1) * chunk_size - 1);

    // validateSort(arr, size);

    // printf("\nSorted array:\n");
    // printArray(arr, size);

    free(arr);
    return 0;
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
                    "  -t nThreads  --- set the number of threads to be created (default: 8)\n"
                    "  -f file --- set the file to be processed\n"
                    "  -h           --- print this help\n",
            cmdName);
}