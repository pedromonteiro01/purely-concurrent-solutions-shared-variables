#include "intSort.h"
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>

#define DEFAULT_NUM_WORKERS 8

static void *distributor_thread(void *args);
static void *worker_thread(void *args);
static void printUsage(char *cmdName);

typedef struct SharedData {
    int *arr;
    int size;
    int num_workers;
    int current_worker;
    char *filename;
    int done;
    pthread_mutex_t mutex;
    pthread_cond_t work_request;
    pthread_cond_t work_done;
} SharedData;


typedef struct WorkerArgs {
    int id;
    SharedData *shared_data;
} WorkerArgs;

void *distributor_thread(void *args) {
    SharedData *shared_data = (SharedData *)args;
    FILE *file = fopen(shared_data->filename, "rb");

    if (file == NULL) {
        printf("Error opening file\n");
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    shared_data->size = ftell(file) / sizeof(int);
    fseek(file, 0, SEEK_SET);

    shared_data->arr = malloc(shared_data->size * sizeof(int));
    if (shared_data->arr == NULL) {
        printf("Error: cannot allocate memory\n");
        return NULL;
    }

    int count = fread(shared_data->arr, sizeof(int), shared_data->size, file);

    if (count != shared_data->size) {
        printf("Error: could not read all integers from file\n");
        return NULL;
    }

    fclose(file);

    pthread_mutex_lock(&shared_data->mutex);

    for (int i = 0; i < shared_data->num_workers; i++) {
        pthread_cond_signal(&shared_data->work_request);
    }
    shared_data->done = 1;

    pthread_mutex_unlock(&shared_data->mutex);
    return NULL;
}


void *worker_thread(void *args) {
    WorkerArgs *worker_args = (WorkerArgs *)args;
    SharedData *shared_data = worker_args->shared_data;
    int worker_id = worker_args->id;

    while (1) {
        pthread_mutex_lock(&shared_data->mutex);

        while (shared_data->current_worker != worker_args->id && !shared_data->done) {
            pthread_cond_wait(&shared_data->work_request, &shared_data->mutex);
        }

        if (shared_data->done) {
            pthread_mutex_unlock(&shared_data->mutex);
            break;
        }

        int chunk_size = shared_data->size / shared_data->num_workers;
        int left = worker_id * chunk_size;
        int right = (worker_id == shared_data->num_workers - 1) ? shared_data->size - 1 : (worker_id + 1) * chunk_size - 1;

        pthread_mutex_unlock(&shared_data->mutex);

        mergeSort(shared_data->arr, left, right);

        pthread_mutex_lock(&shared_data->mutex);
        pthread_cond_signal(&shared_data->work_done);
        pthread_mutex_unlock(&shared_data->mutex);
    }

    return NULL;
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



int main(int argc, char *argv[]) {
    int num_workers = DEFAULT_NUM_WORKERS;
    char *filename = NULL;

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

    SharedData shared_data;
    shared_data.filename = filename;
    shared_data.num_workers = num_workers;
    shared_data.current_worker = 0;
    shared_data.done = 0;
    pthread_mutex_init(&shared_data.mutex, NULL);
    pthread_cond_init(&shared_data.work_request, NULL);
    pthread_cond_init(&shared_data.work_done, NULL);

    pthread_t distributor;
    pthread_create(&distributor, NULL, distributor_thread, &shared_data);
    pthread_join(distributor, NULL);

    pthread_t worker_threads[num_workers];
    WorkerArgs worker_args[num_workers];

    for (int i = 0; i < num_workers; i++) {
        worker_args[i].id = i;
        worker_args[i].shared_data = &shared_data;
        pthread_create(&worker_threads[i], NULL, worker_thread, &worker_args[i]);
    }

    for (int i = 0; i < num_workers; i++) {
        pthread_join(worker_threads[i], NULL);
        if (i == num_workers - 1) {
            mergeSortedSubsequences(shared_data.arr, num_workers, shared_data.size);
        }
    }

    if (validateSort(shared_data.arr, shared_data.size))
        printf("The array is sorted correctly.\n");
    else
        printf("The array is not sorted correctly.\n");

    printArray(shared_data.arr, shared_data.size);

    free(shared_data.arr);
    pthread_mutex_destroy(&shared_data.mutex);
    pthread_cond_destroy(&shared_data.work_request);
    pthread_cond_destroy(&shared_data.work_done);

    return 0;
}


void printUsage(char *cmdName) {
    fprintf(stderr, "\nSynopsis: %s [OPTIONS]\n"
                    "  OPTIONS:\n"
                    "  -t nThreads  --- set the number of threads to be created (default: 8)\n"
                    "  -f file --- set the file to be processed\n"
                    "  -h           --- print this help\n",
            cmdName);
}
