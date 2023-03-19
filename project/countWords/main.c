#include "../utils/file_splitter.h"
#include "countWords.h"
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
    int* total_words;
    int* vowel_count;
    pthread_mutex_t* mutex; // Add a mutex for synchronization
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
    int* total_words = worker_args->total_words;
    int* vowel_count = worker_args->vowel_count;
    pthread_mutex_t* mutex = worker_args->mutex;

    int local_total_words = 0;
    int local_vowel_count[6] = {0};

    // Process alternate chunks in the array.
    for (int i = worker_id; i < total_chunks; i += NUM_WORKERS) {
        uint8_t* chunk = chunks[i];
        printf("Worker %d processing chunk\n", worker_id);

        // processing logic
        count_words_in_chunk(chunk, 4096, &local_total_words, local_vowel_count);
    }

    pthread_mutex_lock(mutex); // Lock the mutex before updating shared variables

    // Update shared variables
    *total_words += local_total_words;
    for (int i = 0; i < 6; i++) {
        vowel_count[i] += local_vowel_count[i];
    }

    pthread_mutex_unlock(mutex); // Unlock the mutex after updating shared variables

    return NULL;
}

int main(int argc, char* argv[]) {
    int total_words;
    pthread_mutex_t mutex;

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

        total_words = 0;
        int vowel_count[6] = {0};
        pthread_mutex_init(&mutex, NULL); // Initialize the mutex

        // Assign chunks to worker threads.
        for (int j = 0; j < NUM_WORKERS; j++) {
            worker_args[j].id = j;
            worker_args[j].chunks = chunks;
            worker_args[j].total_chunks = total_chunks;
            worker_args[j].total_words = &total_words;
            worker_args[j].vowel_count = vowel_count;
            worker_args[j].mutex = &mutex; // Pass the mutex to the worker thread
            pthread_create(&worker_threads[j], NULL, worker, &worker_args[j]);
        }

        // Wait for worker threads to finish.
        for (int j = 0; j < NUM_WORKERS; j++) {
            pthread_join(worker_threads[j], NULL);
        }

        // Free the memory allocated for the array of chunks.
        free_chunks(chunks, total_chunks);
    }

    pthread_mutex_destroy(&mutex);
    return 0;
}