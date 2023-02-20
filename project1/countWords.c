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

char normalize_character(char c) {
    return tolower(c);
}

// Counts the total number of words and the number of words containing each vowel in the given file
void count_words(FILE *file, int *total_words, int *vowel_count) {
    char buffer[1]; // buffer to store the byte read from the file
    char word[256]; // Buffer to hold a word
    int in_word = 0; // Flag indicating whether we are currently in a word
    int word_length = 0; // Length of the current word

    while (fread(buffer, 1, 1, file) == 1) {
        printf("%c", buffer[0]);
        continue;

        if (is_word_character(buffer[0])) {
            if (!in_word) { // If we were not already in a word
                in_word = 1; // Set the flag to indicate we are now in a word
                word_length = 0;
            }

            word[word_length++] = normalize_character(buffer[0]); // Add the normalized character to the current word buffer
        } else if (in_word) { // If the character is not a word character and we were in a word
            in_word = 0; // Clear the flag to indicate we are no longer in a word
            word[word_length] = '\0'; // Null-terminate the current word buffer

            (*total_words)++;

            // Count the vowels in the current word
            for (int i = 0; i < word_length; i++) {
                buffer[0] = normalize_character(word[i]);

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