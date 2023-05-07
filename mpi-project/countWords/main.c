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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <mpi.h>
#include <stdbool.h>

// Define the number of vowels.
#define NUM_VOWELS 6

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
        uint8_t *chunk = (uint8_t *)calloc(4096, sizeof(uint8_t));
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

    printf("num chunks: %d \n", num_chunks);

    fclose(file);
    *total_chunks = num_chunks;
    return chunks;
}

void free_chunks(uint8_t **chunks, int total_chunks)
{
    for (int i = 0; i < total_chunks; i++)
    {
        free(chunks[i]);
    }
    free(chunks);
}

void worker(int rank, int size, uint8_t **chunks, int total_chunks, int *total_words, int *vowel_count)
{
    int chunk_start, chunk_end;
    int local_total_words = 0;
    int local_vowel_count[6] = {0};

    chunk_start = (rank * total_chunks) / size;
    chunk_end = ((rank + 1) * total_chunks) / size;

    for (int i = chunk_start; i < chunk_end; i++)
    {
        uint8_t *chunk = chunks[i];
        count_words_in_chunk(chunk, 4096, &local_total_words, local_vowel_count);
    }

    if (rank == 0) {
        *total_words = local_total_words;
        memcpy(vowel_count, local_vowel_count, 6 * sizeof(int));

        for (int i = 1; i < size; i++) {
            MPI_Recv(&local_total_words, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            *total_words += local_total_words;

            MPI_Recv(local_vowel_count, 6, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < NUM_VOWELS; j++) {
                vowel_count[j] += local_vowel_count[j];
            }
        }
    } else {
        MPI_Send(&local_total_words, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(local_vowel_count, 6, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}


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

#define MAX_FILENAMES 100

int main(int argc, char *argv[])
{
    int total_words;
    int rank, size;

    MPI_Init(&argc, &argv);
    //MPI_DEBUG();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2)
    {
        MPI_Finalize();
        return 1;
    }

    int opt;
    char *file_names[MAX_FILENAMES] = {NULL};
    int num_files = 0;

    if (rank == 0)
    {
        while ((opt = getopt(argc, argv, "f:h")) != -1)
        {
            switch (opt)
            {
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
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    token = strtok(NULL, ",");
                }
                break;
            case 'h':
                printf("help");
                MPI_Finalize();
                return 0;
            default: // Handle invalid option
                printf("Invalid option\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    MPI_Bcast(&num_files, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (num_files == 0)
    {
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    int total_chunks = 0;
    uint8_t **chunks;
    for (int i = 0; i < num_files; i++)
    {
        if (rank == 0)
        {
            printf("Processing file: %s\n", file_names[i]);

            // Split the binary file into chunks.
            chunks = split_file_into_chunks(file_names[i], &total_chunks);
            
            if (!chunks)
            {
                printf("Failed to split file: %s\n", file_names[i]);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }

        MPI_Bcast(&total_chunks, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank != 0)
        {
            // Allocate memory for the array of chunks for non-zero ranks
            chunks = malloc(total_chunks * sizeof(uint8_t *));
            for (int i = 0; i < total_chunks; i++)
            {
                chunks[i] = (uint8_t *)malloc(4096 * sizeof(uint8_t));
            }
        }

        // Broadcast the chunks data to all MPI processes
        for (int i = 0; i < total_chunks; i++)
        {
            MPI_Bcast(chunks[i], 4096, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
        }

        total_words = 0;
        int vowel_count[6] = {0};

        MPI_Barrier(MPI_COMM_WORLD);

        // Call worker function
        worker(rank, size, chunks, total_chunks, &total_words, vowel_count);

        if (rank == 0)
        {
            // Free the memory allocated for the array of chunks.
            free_chunks(chunks, total_chunks);

            print_results(total_words, vowel_count);

            free(file_names[i]);
        }
    }

    MPI_Finalize();

    return 0;
}
