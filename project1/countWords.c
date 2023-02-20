#include <stdio.h>
#include <ctype.h>
#include <string.h>

#define NUM_VOWELS 6

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
}

// Counts the total number of words and the number of words containing each vowel in the given file
void count_words(FILE *file, int *total_words, int *vowel_count) {
    char buffer[4]; // buffer to store the byte read from the file

    // read one byte at a time until the end of the file is reached
    while (fread(buffer, 1, 1, file) == 1) {
        // check if the byte uses a 4-byte encoding
        if ((buffer[0] & 0xF8) == 0xF0) {
            // read the next 3 bytes
            if (fread(buffer + 1, 1, 3, file) != 3) {
                printf("Error reading file\n");
                return;
            }
            // print the 4-byte character
            normalize_character(buffer);
            printf("%c%c%c%c", buffer[0], buffer[1], buffer[2], buffer[3]);
        }
        else {
            // print the 1-byte character
            printf("%c", buffer[0]);
        }
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