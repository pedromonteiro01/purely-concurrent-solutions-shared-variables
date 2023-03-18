// main.c

#include "../utils/file_splitter.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Define the number of worker threads.
#define NUM_WORKERS 8

// Define a struct to pass arguments to worker threads.
typedef struct WorkerArgs {
    int id;
    ChunkQueue* queue;
} WorkerArgs;

// Worker thread function.
void* worker(void* args) {
    WorkerArgs* worker_args = (WorkerArgs*)args;
    int worker_id = worker_args->id;
    ChunkQueue* queue = worker_args->queue;

    // Process chunks in the queue.
    Chunk* chunk = queue->head;
    while (chunk != NULL) {
        printf("Worker %d processing chunk with size %zu bytes\n", worker_id, chunk->size);
        // Add your processing logic here.
        chunk = chunk->next;
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
        ChunkQueue* queue = split_file_into_chunks(argv[i]);
        if (!queue) {
            printf("Failed to split file: %s\n", argv[i]);
            continue;
        }

        // Assign chunks to worker threads.
        for (int j = 0; j < NUM_WORKERS; j++) {
            worker_args[j].id = j;
            worker_args[j].queue = queue;
            pthread_create(&worker_threads[j], NULL, worker, &worker_args[j]);
        }

        // Wait for worker threads to finish.
        for (int j = 0; j < NUM_WORKERS; j++) {
            pthread_join(worker_threads[j], NULL);
        }

        // Free the memory allocated for the ChunkQueue.
        free_chunk_queue(queue);
    }

    return 0;
}