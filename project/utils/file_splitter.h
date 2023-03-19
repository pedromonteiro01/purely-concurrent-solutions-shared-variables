// file_splitter.h

#ifndef FILE_SPLITTER_H
#define FILE_SPLITTER_H

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

// Function to split a binary file into 4KB chunks and return a queue with the chunks.
uint8_t** split_file_into_chunks(const char *file_path, int *total_chunks);

// Function to free the memory allocated for the array of chunks.
void free_chunks(uint8_t **chunks, int total_chunks);

#endif // FILE_SPLITTER_H