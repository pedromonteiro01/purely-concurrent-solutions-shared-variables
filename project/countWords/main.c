#include "countWords.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Define the number of worker threads.
#define NUM_WORKERS 8
#define NUM_VOWELS 6

// Define a struct to pass arguments to worker threads.
typedef struct WorkerArgs {
    int id;
    uint8_t** chunks;
    int total_chunks;
    int* total_words;
    int* vowel_count;
    pthread_mutex_t* mutex; // Add a mutex for synchronization
} WorkerArgs;

// Function to split a binary file into 4KB chunks and return an array of chunks.
// The total number of chunks is stored in the `total_chunks` output parameter.
uint8_t** split_file_into_chunks(const char* file_path, int* total_chunks) {
    FILE* file = fopen(file_path, "rb");
    if (!file) {
        return NULL;
    }

    // Calculate the number of chunks.
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    int num_chunks = (file_size + 4095) / 4096;

    // Allocate memory for the array of chunks.
    uint8_t** chunks = malloc(num_chunks * sizeof(uint8_t*));

    // Read the chunks from the file.
    for (int i = 0; i < num_chunks; i++) {
        uint8_t* chunk = (uint8_t*)malloc(4096 * sizeof(uint8_t));
        size_t chunk_size = fread(chunk, 1, 4096, file);

        if (chunk_size < 4096 && !feof(file))
            exit(EXIT_FAILURE);

        // If this is not the last chunk and it doesn't end in a whitespace, adjust the boundary
        if (i < num_chunks - 1 && is_word_character(chunk[chunk_size - 1], 0)) {
            int adjustment = 0;

            // Move the boundary to the previous whitespace or non-word character
            while (chunk_size > 0 && is_word_character(chunk[chunk_size - 1], 0)) {
                adjustment++;
                chunk_size--;
            }

            if (adjustment > 0) {
                // update the file position (bring it back to the previous whitespace)
                fseek(file, -(adjustment), SEEK_CUR);

                // Fill the remaining bytes with whitespace
                for (int i = 0; i < adjustment; i++)
                    chunk[4095-i] = ' ';
            }
        }

        chunks[i] = chunk;
    }

    fclose(file);
    *total_chunks = num_chunks;
    return chunks;
}




// Function to free the memory allocated for the array of chunks.
void free_chunks(uint8_t** chunks, int total_chunks) {
    for (int i = 0; i < total_chunks; i++) {
        free(chunks[i]);
    }
    free(chunks);
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
        printf("Worker %d processing chunk\n", worker_id);
        uint8_t* chunk = chunks[i];

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

// Function to print the total number of words and the count of words containing each vowel
void print_results(int total_words, int *vowel_count) {
    printf("Total number of words = %d\nN. of words with an\n", total_words);

    // Print the count of words containing each vowel
    char vowels[NUM_VOWELS] = {'a', 'e', 'i', 'o', 'u', 'y'};

    for (int i = 0; i < NUM_VOWELS; i++) {
        printf("%5c ", toupper(vowels[i]));
    }

    printf("\n");

    for (int i = 0; i < NUM_VOWELS; i++) {
        printf("%5d ", vowel_count[i]);
    }

    printf("\n\n");
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

        print_results(total_words, vowel_count);
    }

    pthread_mutex_destroy(&mutex);
    return 0;
}