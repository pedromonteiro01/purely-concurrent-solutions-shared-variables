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

// Declare a struct to represent the queue.
typedef struct ChunkQueue {
    Chunk* head;
    Chunk* tail;
} ChunkQueue;

// Function to split a binary file into 4KB chunks and return a queue with the chunks.
ChunkQueue* split_file_into_chunks(const char* file_path);

// Function to free the memory allocated for the ChunkQueue.
void free_chunk_queue(ChunkQueue* queue);

#endif // FILE_SPLITTER_H
