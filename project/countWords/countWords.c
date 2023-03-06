#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>

#define NUM_VOWELS 6

int is_word_character(char* c, int in_word) {
    if (*c == '\'' && in_word == 1)
        return 1;

    return isalnum(*c);
}

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


// Counts the total number of words and the number of words containing each vowel in the given file
void count_words(FILE *file, int *total_words, int *vowel_count) {
    size_t num_bytes = 0;
    char buffer[4]; // buffer to store the byte read from the file
    char word[256]; // Buffer to hold a word
    int counted_vowel[6] = {0}; // Initialize all flags to false
    int in_word = 0; // Flag indicating whether we are currently in a word
    int word_length = 0; // Length of the current word

    while (fread(buffer, 1, 1, file) == 1) {
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
                printf("Invalid UTF-8 sequence detected\n");
                return;
            }

            // read the remaining bytes of the encoding
            if (fread(buffer + 1, 1, num_bytes - 1, file) != num_bytes - 1) {
                printf("Error reading file\n");
                return;
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
        }

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

int main(int argc, char *argv[]) {
    /* --------- Check that at least one file is provided as an argument -------- */
    if (argc < 2) {
        printf("Usage: %s <file1> [<file2> ...]\n", argv[0]);
        return 1;
    }

    /* ----- Loop through all the file arguments and count the words in them ---- */
    for (int i = 1; i < argc; i++) {
        int total_words = 0;
        int vowel_count[NUM_VOWELS] = {0};
        FILE *file = fopen(argv[i], "rb"); // open the file in binary mode

        if (file == NULL) {
            printf("Could not open file: %s\n", argv[i]);
            continue;
        }

        printf("File name: %s\n", argv[i]);
        // Count the words in the file and update the total word count and vowel count arrays
        count_words(file, &total_words, vowel_count);

        fclose(file);

        printf("Total number of words = %d\nN. of words with an\n", total_words);

        // Print the count of words containing each vowel
        char vowels[NUM_VOWELS] = {'a', 'e', 'i', 'o', 'u', 'y'};

        for (int i = 0; i < NUM_VOWELS; i++) {
            printf("%5c ", toupper(vowels[i]));
        }

        printf("\n");

        for (int i = 0; i < NUM_VOWELS; i++) {
            printf("%5d ", vowel_count[i]);
        }

        printf("\n\n");

    }

    return 0;
}