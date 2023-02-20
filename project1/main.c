#include <stdio.h>
#include <ctype.h>

#define NUM_LETTERS 26

int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("%s <text> \n", argv[0]);
        return 1; // error, no input text
    }

    int letter_count[NUM_LETTERS] = {0};
    int bytes = 0;
    char *text = argv[1];

    // read input byte by byte  
    while (*text) {
        bytes++;

        if (isalpha(*text)) { // checks whether a character is an alphabet
            int index = tolower(*text) - 'a';
            letter_count[index]++;
        }

        text++;
    }

    printf("Total bytes: %d\n", bytes);
    printf("Letters:\n");
    for (int i = 0; i < NUM_LETTERS; i++) {
        char letter = 'a' + i;

        // only prints letters that appears at least once
        if (letter_count[i] != 0)
            printf("%c: %d\n", letter, letter_count[i]);
    }

    return 0;
}