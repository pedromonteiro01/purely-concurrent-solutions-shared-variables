#include <stdio.h>
#include <stdlib.h>

// set the maximum size of the array
#define MAX_SIZE 10000

int main(int argc, char *argv[]) {
    int data[MAX_SIZE], num_arr, i, j;

    // check if the program is invoked correctly
    if (argc != 2) {
        fprintf(stderr, "Usage: %s filename\n", argv[0]);
        return 1;
    }

    // check if file exists
    FILE *fp = fopen(argv[1], "rb");
    if (fp == NULL) {
        fprintf(stderr, "Error: Cannot open file %s\n", argv[1]);
        return 1;
    }

    // create array with integers from binary file
    num_arr = fread(data, sizeof(int), MAX_SIZE, fp);

    // sort array in increasing order
    for (i = 0; i < num_arr-1; i++) {
        int min_idx = i;
        for (j = i+1; j < num_arr; j++) {
            if (data[j] < data[min_idx]) {
                min_idx = j;
            }
        }
        int temp = data[i];
        data[i] = data[min_idx];
        data[min_idx] = temp;
    }

    // print sorted array
    for (i = 0; i < num_arr; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");

    fclose(fp);
    return 0;
}