#include "file_splitter.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>


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
    uint8_t** chunks = (uint8_t**)malloc(num_chunks * sizeof(uint8_t*));

    // Read the chunks from the file.
    for (int i = 0; i < num_chunks; i++) {
        uint8_t* chunk = (uint8_t*)malloc(4096 * sizeof(uint8_t));
        size_t chunk_size = fread(chunk, 1, 4096, file);

        if (chunk_size < 4096 && !feof(file))
            exit(EXIT_FAILURE);

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