#include "../utils/file_splitter.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Define the number of worker threads.
#define NUM_WORKERS 8

// Define a struct to pass arguments to worker threads.
typedef struct WorkerArgs {
    int id;
    uint8_t** chunks;
    int total_chunks;
} WorkerArgs;

void print_binary_contents(const uint8_t* data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        printf("%c", data[i]);
        if ((i + 1) % 16 == 0) {
            printf("\n");
        }
    }
    printf("\n");
}

// Worker thread function.
void* worker(void* args) {
    WorkerArgs* worker_args = (WorkerArgs*)args;
    int worker_id = worker_args->id;
    uint8_t** chunks = worker_args->chunks;
    int total_chunks = worker_args->total_chunks;

    // Process alternate chunks in the array.
    for (int i = worker_id; i < total_chunks; i += NUM_WORKERS) {
        uint8_t* chunk = chunks[i];
        printf("Worker %d processing chunk\n", worker_id);
        // processing logic
        //print_binary_contents(chunk, 4096);
    }

    return NULL;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <file1> [file2] ...\n", argv[0]);
        return 1;
    }

    // Create worker threads.
    pthread_t worker_threads[NUM_WORKERS];
    WorkerArgs worker_args[NUM_WORKERS];

    // Iterate over the input files.
    for (int i = 1; i < argc; i++) {
        printf("Processing file: %s\n", argv[i]);

        // Split the binary file into chunks.
        int total_chunks;
        uint8_t** chunks = split_file_into_chunks(argv[i], &total_chunks);
        if (!chunks) {
            printf("Failed to split file: %s\n", argv[i]);
            continue;
        }

        // Assign chunks to worker threads.
        for (int j = 0; j < NUM_WORKERS; j++) {
            worker_args[j].id = j;
            worker_args[j].chunks = chunks; // This line is passing a pointer to chunks, so it is memory efficient
            worker_args[j].total_chunks = total_chunks;
            pthread_create(&worker_threads[j], NULL, worker, &worker_args[j]);
        }

        // Wait for worker threads to finish.
        for (int j = 0; j < NUM_WORKERS; j++) {
            pthread_join(worker_threads[j], NULL);
        }

        // Free the memory allocated for the array of chunks.
        free_chunks(chunks, total_chunks);
    }

    return 0;
}