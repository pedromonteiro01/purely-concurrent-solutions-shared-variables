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

// Function to split a binary file into 4KB chunks and return a queue with the chunks.
ChunkQueue* split_file_into_chunks(const char* file_path) {
    FILE* file = fopen(file_path, "rb");
    if (!file) {
        return NULL;
    }

    ChunkQueue* queue = (ChunkQueue*)malloc(sizeof(ChunkQueue));
    queue->head = NULL;
    queue->tail = NULL;

    bool done = false;
    while (!done) {
        Chunk* chunk = create_chunk();
        chunk->size = fread(chunk->data, 1, sizeof(chunk->data), file);

        if (chunk->size > 0) {
            if (queue->head == NULL) {
                queue->head = chunk;
                queue->tail = chunk;
            } else {
                queue->tail->next = chunk;
                queue->tail = chunk;
            }
        }

        if (chunk->size < sizeof(chunk->data)) {
            done = true;
        }
    }

    fclose(file);
    return queue;
}

// Function to free the memory allocated for the ChunkQueue.
void free_chunk_queue(ChunkQueue* queue) {
    Chunk* current = queue->head;
    while (current != NULL) {
        Chunk* next = current->next;
        free(current);
        current = next;
    }
    free(queue);
}
