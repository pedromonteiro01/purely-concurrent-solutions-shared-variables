#include <stdio.h>
#include <ctype.h>

#define NUM_LETTERS 26

long ParallelCharReader(char *text, int *letterFrequency)
{
    long bytes = 0;

    // read input byte by byte  
    while (*text) {
        if (isalpha(*text)) { // checks whether a character is an alphabet
            int index = tolower(*text) - 'a';
            letterFrequency[index]++;
        }

        text++;
        bytes++;
    }

    return bytes;
}

int main(int argc, char *argv[])
{
    // Validate input arguments
    if (argc < 2) {
        printf("%s <text> \n", argv[0]);
        return 1;
    }

    /* -------------------- Declare and initialize variables -------------------- */
    char *text = argv[1];
    
    //'{0}' initializes all the elements in the array to 0.
    int letterFrequency[NUM_LETTERS] = {0};
    /* -------------------------------------------------------------------------- */


    /* -------------- Calls function to read chars from input text -------------- */
    long bytes = ParallelCharReader(text, letterFrequency);


    /* -------------------------------------------------------------------------- */
    /*                               Prints results                               */
    /* -------------------------------------------------------------------------- */
    printf("Bytes read: %ld\n\nLetter Frequency:\n", bytes);

    // Print letter frequency table
    for (int i = 0; i < NUM_LETTERS; i++) {
        char letter = 'a' + i;
        printf("%c: %d, ", letter, letterFrequency[i]);
    }

    printf("\n");

    return 0;
}