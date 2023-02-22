#include <stdio.h>
#include <ctype.h>
#include <string.h>

#define NUM_VOWELS 6

int is_word_character(char c) {
    return isalnum(c) || c == '_';
}

int is_word_boundary(char c) {
    return isspace(c) || ispunct(c);
}

void utf8ToLowercase(char* buffer) {
    // Copy the 4 bytes to a temporary buffer
    char temp[4];
    memcpy(temp, buffer, 4);
    
    // Convert each byte to lowercase
    temp[0] = tolower(temp[0]);
    temp[1] = tolower(temp[1]);
    temp[2] = tolower(temp[2]);
    temp[3] = tolower(temp[3]);

    // Copy the bytes back to the original buffer
    memcpy(buffer, temp, 4);
}

void normalize_character(char* buffer) {
    utf8ToLowercase(buffer);

    /* This code shifts the first byte to the left by 8 bits and then ORs it with the second byte.
       The resulting value is an unsigned integer that represents the two-byte sequence. */
    unsigned int value = ((unsigned char)buffer[0] << 8) | (unsigned char)buffer[1];
    
    switch (value) {
        case 0xC3A0: // à
        case 0xC3A1: // á
        case 0xC3A2: // â
        case 0xC3A3: // ã
        case 0xC3A4: // ä
            buffer[0] = 0x61; // a
            break;
        case 0xC3A8: // è
        case 0xC3A9: // é
        case 0xC3AA: // ê
        case 0xC3AB: // ë
            buffer[0] = 0x65; // e
            break;
        case 0xC3AC: // ì
        case 0xC3AD: // í
        case 0xC3AE: // î
        case 0xC3AF: // ï
            buffer[0] = 0x69; // i
            break;
        case 0xC3B2: // ò
        case 0xC3B3: // ó
        case 0xC3B4: // ô
        case 0xC3B5: // õ
        case 0xC3B6: // ö
            buffer[0] = 0x6f; // o
            break;
        case 0xC3B9: // ù
        case 0xC3BA: // ú
        case 0xC3BB: // û
        case 0xC3BC: // ü
            buffer[0] = 0x75; // u
            break;
        case 0xC3BD: // ý
        case 0xC3BF: // ÿ
            buffer[0] = 0x79; // y
            break;
        case 0xC3A7: // ç
            buffer[0] = 0x63; // c
            break;
    }
}


// Counts the total number of words and the number of words containing each vowel in the given file
void count_words(FILE *file, int *total_words, int *vowel_count) {
    size_t num_bytes = 0;
    char buffer[4]; // buffer to store the byte read from the file
    char word[256]; // Buffer to hold a word
    int in_word = 0; // Flag indicating whether we are currently in a word
    int word_length = 0; // Length of the current word

    while (fread(buffer, 1, 1, file) == 1) {
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
        normalize_character(buffer);
        printf("%c", buffer[0]);

        if (is_word_character(buffer[0])) {
            if (!in_word) { // If we were not already in a word
                in_word = 1; // Set the flag to indicate we are now in a word
                word_length = 0;
            }

            word[word_length++] = buffer[0]; // Add the character to the current word buffer
        } else if (in_word) { // If the character is not a word character and we were in a word
            in_word = 0; // Clear the flag to indicate we are no longer in a word
            word[word_length] = '\0'; // Null-terminate the current word buffer

            (*total_words)++;

            // Count the vowels in the current word
            for (int i = 0; i < word_length; i++) {
                buffer[0] = word[i];

                if (buffer[0] == 'a') {
                    vowel_count[0]++;
                } else if (buffer[0] == 'e') {
                    vowel_count[1]++;
                } else if (buffer[0] == 'i') {
                    vowel_count[2]++;
                } else if (buffer[0] == 'o') {
                    vowel_count[3]++;
                } else if (buffer[0] == 'u') {
                    vowel_count[4]++;
                } else if (buffer[0] == 'y') {
                    vowel_count[5]++;
                }
            }
        }
    }

    if (in_word) { // If we were in a word at the end of the file
        (*total_words)++; // Increment the total word count
    }
}

int main(int argc, char *argv[]) {
    /* --------- Check that at least one file is provided as an argument -------- */
    if (argc < 2) {
        printf("Usage: %s <file1> [<file2> ...]\n", argv[0]);
        return 1;
    }

    int total_words = 0;
    int vowel_count[NUM_VOWELS] = {0};

    /* ----- Loop through all the file arguments and count the words in them ---- */
    for (int i = 1; i < argc; i++) {
        FILE *file = fopen(argv[i], "rb"); // open the file in binary mode

        if (file == NULL) {
            printf("Could not open file: %s\n", argv[i]);
            continue;
        }

        // Count the words in the file and update the total word count and vowel count arrays
        count_words(file, &total_words, vowel_count);

        fclose(file);
    }
    /* -------------------------------------------------------------------------- */

    printf("Total words: %d\n", total_words);

    // Print the count of words containing each vowel
    char vowels[NUM_VOWELS] = {'a', 'e', 'i', 'o', 'u', 'y'};
    for (int i = 0; i < NUM_VOWELS; i++) {
        printf("Words containing '%c': %d\n", vowels[i], vowel_count[i]);
    }

    return 0;
}