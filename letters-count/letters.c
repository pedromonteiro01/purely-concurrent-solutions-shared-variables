#include <stdio.h>
#include <ctype.h>

#define NUM_LETTERS 26

int main()
{
    int letter_count[NUM_LETTERS] = {0};
    int bytes = 0;
    int c;

    // read input byte by byte  
    while ((c = getchar()) != EOF) {
        bytes++;

        if (isalpha(c)) {
            int index = tolower(c) - 'a';
            letter_count[index]++;
        }
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