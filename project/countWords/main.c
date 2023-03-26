/**
 *  \file main.c (implementation file)
 *
 *  \brief Problem name: Count Words.
 *
 *  Word count application that can process a 
 *  binary file and count the number of words in it, 
 *  as well as the number of vowels in each word.
 *
 *
 *  \authors Pedro Monteiro & José Trigo - March 2023
 */

#include "countWords.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

// Define the number of worker threads.
#define DEFAULT_NUM_WORKERS 8
#define NUM_VOWELS 6

/** \brief checks if a given character is a whitespace, separator symbol, or a punctuation symbol */
static int is_separator_or_whitespace_or_punctuation(char c);

/** \brief takes a file path as input and returns an array of 4KB chunks */
static uint8_t **split_file_into_chunks(const char *file_path, int *total_chunks);

/** \brief frees the memory allocated for the array of chunks */
void free_chunks(uint8_t **chunks, int total_chunks);

/** \brief print command usage */
static void printUsage(char *cmdName);

/** \brief worker life cycle routine */
void *worker(void *args);

/** \brief default number of worker threads */
int NUM_WORKERS = DEFAULT_NUM_WORKERS;

/**
 *  \brief Function is_separator_or_whitespace_or_punctuation.
 *
 *  Its role is to check if a given character is a whitespace, separator symbol, or a punctuation symbol.
 *
 *  \param c word character
 */

int is_separator_or_whitespace_or_punctuation(char c)
{
    // Check if the character is a whitespace
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
    {
        return 1;
    }

    // Check if the character is a separation symbol
    if (c == '-' || c == '\"' || c == '[' || c == ']' || c == '(' || c == ')' ||
        c == '\xe2' || c == '\x80' || c == '\x9c' || c == '\x9d')
    {
        return 1;
    }

    // Check if the character is a punctuation symbol
    if (c == '.' || c == ',' || c == ':' || c == ';' || c == '?' || c == '!')
    {
        return 1;
    }

    return 0;
}

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
    uint8_t **chunks;
    int total_chunks;
    int *total_words;
    int *vowel_count;
    pthread_mutex_t *mutex; // Add a mutex for synchronization
} WorkerArgs;

/**
 *  \brief Function split_file_into_chunks.
 *
 *  Its role is to split a binary file into 4KB chunks and return an array of chunks.
 *
 *  \param file_path  pointer to a string that represents the path of the file
 *  \param total_chunks variable with the total number of chunks
 */

uint8_t **split_file_into_chunks(const char *file_path, int *total_chunks)
{
    FILE *file = fopen(file_path, "rb");
    if (!file)
    {
        return NULL;
    }

    // Calculate the number of chunks.
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    int num_chunks = (file_size + 4095) / 4096;

    // Allocate memory for the array of chunks.
    uint8_t **chunks = malloc(num_chunks * sizeof(uint8_t *));

    // Read the chunks from the file.
    for (int i = 0; i < num_chunks; i++)
    {
        uint8_t *chunk = (uint8_t *)malloc(4096 * sizeof(uint8_t));
        size_t chunk_size = fread(chunk, 1, 4096, file);

        if (chunk_size < 4096 && !feof(file))
            exit(EXIT_FAILURE);

        // If this is not the last chunk and it doesn't end in a whitespace, adjust the boundary
        if (i < num_chunks - 1 && !is_separator_or_whitespace_or_punctuation(chunk[chunk_size - 1]))
        {
            int adjustment = 0;

            // Move the boundary to the previous whitespace or non-word character
            while (chunk_size > 0 && !is_separator_or_whitespace_or_punctuation(chunk[chunk_size - 1]))
            {
                adjustment++;
                chunk_size--;
            }

            if (adjustment > 0)
            {
                // update the file position (bring it back to the previous whitespace)
                fseek(file, -(adjustment), SEEK_CUR);

                // Fill the remaining bytes with whitespace
                for (int i = 0; i < adjustment; i++)
                    chunk[4095 - i] = ' ';
            }
        }

        chunks[i] = chunk;
    }

    fclose(file);
    *total_chunks = num_chunks;
    return chunks;
}

/**
 *  \brief Function free_chunks.
 *
 *  Its role is to free the memory allocated for the array of chunks.
 *
 *  \param chunks pointer to an array of pointers to uint8_t
 *  \param total_chunks variable with the total number of chunks
 */
void free_chunks(uint8_t **chunks, int total_chunks)
{
    for (int i = 0; i < total_chunks; i++)
    {
        free(chunks[i]);
    }
    free(chunks);
}

/**
 *  \brief Function worker.
 *
 *  Its role is to simulate the life cycle of a worker.
 *
 *  \param args pointer to the Worker structure
 */
void *worker(void *args)
{
    WorkerArgs *worker_args = (WorkerArgs *)args;
    int worker_id = worker_args->id;
    uint8_t **chunks = worker_args->chunks;
    int total_chunks = worker_args->total_chunks;
    int *total_words = worker_args->total_words;
    int *vowel_count = worker_args->vowel_count;
    pthread_mutex_t *mutex = worker_args->mutex;

    int local_total_words = 0;
    int local_vowel_count[6] = {0};

    // Process alternate chunks in the array.
    for (int i = worker_id; i < total_chunks; i += NUM_WORKERS)
    {
        // printf("Worker %d processing chunk\n", worker_id);
        uint8_t *chunk = chunks[i];

        // processing logic
        count_words_in_chunk(chunk, 4096, &local_total_words, local_vowel_count);
    }

    pthread_mutex_lock(mutex); // Lock the mutex before updating shared variables

    // Update shared variables
    *total_words += local_total_words;
    for (int i = 0; i < 6; i++)
    {
        vowel_count[i] += local_vowel_count[i];
    }
    pthread_mutex_unlock(mutex); // Unlock the mutex after updating shared variables

    return NULL;
}

/**
 *  \brief Function print_results.
 *
 *  Its role is to print the total number of words and the count of words containing each vowel.
 *
 *  \param total_words the total number of words
 *  \param vowel_count pointer to an integer array that contains the count of words containing each vowel
 */
void print_results(int total_words, int *vowel_count)
{
    printf("Total number of words = %d\nN. of words with an\n", total_words);

    // Print the count of words containing each vowel
    char vowels[NUM_VOWELS] = {'a', 'e', 'i', 'o', 'u', 'y'};

    for (int i = 0; i < NUM_VOWELS; i++)
    {
        printf("%5c ", toupper(vowels[i]));
    }

    printf("\n");

    for (int i = 0; i < NUM_VOWELS; i++)
    {
        printf("%5d ", vowel_count[i]);
    }

    printf("\n\n");
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

#define MAX_FILENAMES 100

int main(int argc, char *argv[])
{
    int total_words;
    pthread_mutex_t mutex;

    if (argc < 2)
    {
        printUsage(argv[0]);
        return 1;
    }

    int opt;
    char *file_names[MAX_FILENAMES] = {NULL};
    int num_files = 0;
    while ((opt = getopt(argc, argv, "t:f:h")) != -1)
    {
        switch (opt)
        {
        case 't':
            NUM_WORKERS = atoi(optarg);
            break;
        case 'f':
            // Split the file names by comma and save them to the file_names array
            char *token;
            token = strtok(optarg, ",");
            while (token != NULL)
            {
                file_names[num_files] = strdup(token);
                num_files++;
                if (num_files >= MAX_FILENAMES)
                {
                    printf("Too many file names\n");
                    return 1;
                }
                token = strtok(NULL, ",");
            }
            break;
        case 'h':
            printUsage(argv[0]);
            return 0;
        default: // Handle invalid option
            printf("Invalid option\n");
            return 1;
        }
    }

    if (num_files == 0)
    {
        printUsage(argv[0]);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_files; i++)
    {
        printf("Processing file: %s\n", file_names[i]);

        // Split the binary file into chunks.
        int total_chunks;
        uint8_t **chunks = split_file_into_chunks(file_names[i], &total_chunks);
        if (!chunks)
        {
            printf("Failed to split file: %s\n", file_names[i]);
            continue;
        }

        total_words = 0;
        int vowel_count[6] = {0};
        pthread_mutex_init(&mutex, NULL); // Initialize the mutex

        // Create worker threads.
        pthread_t worker_threads[NUM_WORKERS];
        WorkerArgs worker_args[NUM_WORKERS];

        // Assign chunks to worker threads.
        for (int j = 0; j < NUM_WORKERS; j++)
        {
            worker_args[j].id = j;
            worker_args[j].chunks = chunks;
            worker_args[j].total_chunks = total_chunks;
            worker_args[j].total_words = &total_words;
            worker_args[j].vowel_count = vowel_count;
            worker_args[j].mutex = &mutex; // Pass the mutex to the worker thread
            pthread_create(&worker_threads[j], NULL, worker, &worker_args[j]);
        }

        // Wait for worker threads to finish.
        for (int j = 0; j < NUM_WORKERS; j++)
        {
            pthread_join(worker_threads[j], NULL);
        }

        // Free the memory allocated for the array of chunks.
        free_chunks(chunks, total_chunks);

        print_results(total_words, vowel_count);

        pthread_mutex_destroy(&mutex);

        free(file_names[i]);
    }

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
                    "  -f files --- set the files to be processed\n"
                    "  -h           --- print this help\n",
            cmdName);
}