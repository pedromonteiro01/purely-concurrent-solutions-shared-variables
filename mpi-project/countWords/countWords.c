#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#define NUM_VOWELS 6

/** \brief determine whether the given character c is a valid character in a word */
static int is_word_character(char* c, int in_word);

/** \brief normalize a character by converting certain UTF-8 encoded characters to their ASCII equivalent */
static void normalize_character(char* buffer);

/** \brief count the total number of words and the number of words containing each vowel in the given chunk */
void count_words_in_chunk(uint8_t* chunk, size_t chunk_size, int *total_words, int *vowel_count);

/**
 *  \brief Function is_word_character.
 *
 *  Its role is to determine whether the given character c is a valid character in a word.
 *
 *  \param c: pointer to a character
 *  \param in_word: integer flag that indicates whether the current character is part of a word or not
 * 
 *  \return returns 1 if the character is a word character or a single quote within a word, and 0 otherwise
 */
int is_word_character(char* c, int in_word) {
    if (*c == '\'' && in_word == 1)
        return 1;

    return isalnum(*c);
}

/**
 *  \brief Function normalize_character.
 *
 *  Its role is to normalize a character by converting certain UTF-8 encoded characters to their ASCII equivalent.
 *
 *  \param buffer: pointer to a character array
 * 
 */
void normalize_character(char* buffer) {
    /* This code shifts the first byte to the left by 8 bits and then ORs it with the second byte.
       The resulting value is an unsigned integer that represents the two-byte sequence. */
    unsigned int value = ((unsigned char)buffer[0] << 8) | (unsigned char)buffer[1];

    switch (value) {
        case 0xC3A0:  // à
        case 0xC3A1:  // á
        case 0xC3A2:  // â
        case 0xC3A3:  // ã
        case 0xC380:  // À
        case 0xC381:  // Á
        case 0xC382:  // Â
        case 0xC383:  // Ã
            buffer[0] = 0x61;  // a
            break;
        case 0xC3A8:  // è
        case 0xC3A9:  // é
        case 0xC3AA:  // ê
        case 0xC388:  // È
        case 0xC389:  // É
        case 0xC38A:  // Ê
            buffer[0] = 0x65;  // e
            break;
        case 0xC3AC:  // ì
        case 0xC3AD:  // í
        case 0xC38C:  // Ì
        case 0xC38D:  // Í
            buffer[0] = 0x69;  // i
            break;
        case 0xC3B2:  // ò
        case 0xC3B3:  // ó
        case 0xC3B4:  // ô
        case 0xC3B5:  // õ
        case 0xC392:  // Ò
        case 0xC393:  // Ó
        case 0xC394:  // Ô
        case 0xC395:  // Õ
            buffer[0] = 0x6f;  // o
            break;
        case 0xC3B9:  // ù
        case 0xC3BA:  // ú
        case 0xC399:  // Ù
        case 0xC39A:  // Ú
            buffer[0] = 0x75;  // u
            break;
        case 0xC3A7:  // ç
        case 0xC387:  // Ç
            buffer[0] = 0x63;  // c
            break;
        case 0xC2AB: // Left-Pointing Double Angle Quotation Mark
        case 0xC2BB: // Right-Pointing Double Angle Quotation Mark
            buffer[0] = 0x20; // Apostrophe
            break;
        }
}


/**
 *  \brief Function count_words_in_chunk.
 *
 *  Its role is to count the total number of words and the number of words containing each vowel in the given chunk.
 *
 *  \param chunk: pointer to a block of memory containing a chunk of text in UTF-8 format
 *  \param chunk_size: size in bytes of the chunk parameter
 *  \param total_words: pointer to an integer variable that will be used to store the total number of words counted in the chunk
 *  \param vowel_count: pointer to an integer array of size 6 that will be used to store the number of words that contain each of 6 vowels
 * 
 */
void count_words_in_chunk(uint8_t* chunk, size_t chunk_size, int *total_words, int *vowel_count) {
    size_t num_bytes = 0;
    char buffer[4]; // buffer to store the byte read from the chunk
    char word[256]; // Buffer to hold a word
    int counted_vowel[6] = {0}; // Initialize all flags to false
    int in_word = 0; // Flag indicating whether we are currently in a word
    int word_length = 0; // Length of the current word

    for (size_t chunk_pos = 0; chunk_pos < chunk_size; chunk_pos++) {
        buffer[0] = chunk[chunk_pos];
        num_bytes = 1;

        // check if the byte uses a multi-byte encoding
        if ((buffer[0] & 0x80) == 0x80) {
            // determine the number of bytes in the encoding
            if ((buffer[0] & 0xE0) == 0xC0) {
                num_bytes = 2;
            } else if ((buffer[0] & 0xF0) == 0xE0) {
                num_bytes = 3;
            } else if ((buffer[0] & 0xF8) == 0xF0) {
                num_bytes = 4;
            } else {
                printf("%x - Invalid UTF-8 sequence detected \n", buffer[0]);
                return;
            }

            // read the remaining bytes of the encoding
            for (size_t i = 1; i < num_bytes; i++) {
                if (chunk_pos + i < chunk_size) {
                    buffer[i] = chunk[++chunk_pos];
                } else {
                    printf("Error reading chunk\n");
                    return;
                }
            }
        }

        // process the character
        if (num_bytes == 1) { // no need to normalize, only convert to lower case
            buffer[0] = tolower(buffer[0]);
        }
        else if (num_bytes == 2) { // normalize character
            normalize_character(buffer);

        } else if (num_bytes > 2) {
            /* --------- handle 3 byte utf-8 char edge cases -------- */
            switch (buffer[2]) {
                case (char) 0x98:  // Left Single Quotation Mark
                case (char) 0x99:  // Right Single Quotation Mark
                    buffer[0] = 0x27;  // regular apostrophe
                    break;
                }
        }

        //printf("%c", buffer[0]);

        if (is_word_character(buffer, in_word)) {
            if (!in_word) { // If we were not already in a word
                in_word = 1; // Set the flag to indicate we are now in a word
                word_length = 0;
                memset(counted_vowel, 0, sizeof(counted_vowel)); // Reset all flags to false
                (*total_words)++;
            }

            word[word_length++] = buffer[0]; // Add the character to the current word buffer
        } else if (in_word) { // If the character is not a word character and we were in a word
            in_word = 0; // Clear the flag to indicate we are no longer in a word
            word[word_length] = '\0'; // Null-terminate the current word buffer

            // Count the first occurence of a vowel in the current word
            for (int i = 0; i < word_length; i++) {
                buffer[0] = word[i];

                if (buffer[0] == 'a' && !counted_vowel[0]) {
                    counted_vowel[0] = 1;
                    vowel_count[0]++;
                } else if (buffer[0] == 'e' && !counted_vowel[1]) {
                    counted_vowel[1] = 1;
                    vowel_count[1]++;
                } else if (buffer[0] == 'i' && !counted_vowel[2]) {
                    counted_vowel[2] = 1;
                    vowel_count[2]++;
                } else if (buffer[0] == 'o' && !counted_vowel[3]) {
                    counted_vowel[3] = 1;
                    vowel_count[3]++;
                } else if (buffer[0] == 'u' && !counted_vowel[4]) {
                    counted_vowel[4] = 1;
                    vowel_count[4]++;
                } else if (buffer[0] == 'y' && !counted_vowel[5]) {
                    counted_vowel[5] = 1;
                    vowel_count[5]++;
                }
            }
        }
    }
}