// file_splitter.h

#ifndef FILE_SPLITTER_H
#define FILE_SPLITTER_H

#include <stdio.h>
#include <stdbool.h>

// Declare a struct to hold the chunk data and a pointer to the next chunk.
typedef struct Chunk {
    unsigned char data[4096];
    size_t size;
    struct Chunk* next;
} Chunk;

// Function to split a binary file into 4KB chunks and return a queue with the chunks.
Chunk **split_file_into_chunks(const char *file_path, int *total_chunks);

// Function to free the memory allocated for the array of chunks.
void free_chunks(Chunk **chunks, int total_chunks);

#endif // FILE_SPLITTER_H
