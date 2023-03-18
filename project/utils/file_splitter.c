// file_splitter.c

#include "file_splitter.h"
#include <stdlib.h>

// Helper function to create a new chunk.
static Chunk* create_chunk() {
    Chunk* chunk = (Chunk*)malloc(sizeof(Chunk));
    chunk->size = 0;
    chunk->next = NULL;
    return chunk;
}


// Function to split a binary file into 4KB chunks and return an array of chunks.
// The total number of chunks is stored in the `total_chunks` output parameter.
Chunk** split_file_into_chunks(const char* file_path, int* total_chunks) {
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
    Chunk** chunks = (Chunk**)malloc(num_chunks * sizeof(Chunk*));

    // Read the chunks from the file.
    for (int i = 0; i < num_chunks; i++) {
        Chunk* chunk = create_chunk();
        chunk->size = fread(chunk->data, 1, sizeof(chunk->data), file);
        chunks[i] = chunk;
    }

    fclose(file);
    *total_chunks = num_chunks;
    return chunks;
}

// Function to free the memory allocated for the array of chunks.
void free_chunks(Chunk** chunks, int total_chunks) {
    for (int i = 0; i < total_chunks; i++) {
        free(chunks[i]);
    }
    free(chunks);
}