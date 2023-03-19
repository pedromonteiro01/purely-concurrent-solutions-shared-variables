// countWords.h

#ifndef COUNT_WORDS_H
#define COUNT_WORDS_H

#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>

#define NUM_VOWELS 6

// Counts the total number of words and the number of words containing each vowel in the given file.
void count_words_in_chunk(uint8_t *chunk, size_t chunk_size, int *total_words, int *vowel_count);

#endif // COUNT_WORDS_H