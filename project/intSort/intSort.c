#include <stdio.h>
#include <stdlib.h>

// Merges two sorted subarrays
void merge(int* arr, int left, int middle, int right) {
    int i, j, k;

    // Find sizes of two subarrays
    int left_size = middle - left + 1;
    int right_size = right - middle;

    // Create temporary arrays for left and right subarrays
    int* left_arr = malloc(left_size * sizeof(int));
    int* right_arr = malloc(right_size * sizeof(int));

    // Copy data to temporary arrays
    for (i = 0; i < left_size; i++) {
        left_arr[i] = arr[left + i];
    }
    for (j = 0; j < right_size; j++) {
        right_arr[j] = arr[middle + 1 + j];
    }

    // Merge the two subarrays into arr
    i = 0;
    j = 0;
    k = left;
    while (i < left_size && j < right_size) {
        if (left_arr[i] <= right_arr[j]) {
            arr[k] = left_arr[i];
            i++;
        } else {
            arr[k] = right_arr[j];
            j++;
        }
        k++;
    }

    // Copy remaining elements of left subarray, if any
    while (i < left_size) {
        arr[k] = left_arr[i];
        i++;
        k++;
    }

    // Copy remaining elements of right subarray, if any
    while (j < right_size) {
        arr[k] = right_arr[j];
        j++;
        k++;
    }

    // Free temporary arrays
    free(left_arr);
    free(right_arr);
}

// Recursive function to sort array using merge sort
void mergeSort(int* arr, int left, int right) {
    if (left < right) {
        int middle = left + (right - left) / 2;
        mergeSort(arr, left, middle);
        mergeSort(arr, middle + 1, right);
        merge(arr, left, middle, right);
    }
}

// Function to print the sorted array
void printArray(int* arr, int size) {
    int i;
    for (i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

void validateSort(int* arr, int N) {
    int i, N;

    for (i = 0; i < N - 1; i++) {
        if (arr[i] > arr[i+1]) { 
            printf ("Error in position %d between element %d and %d\n", i, arr[i], arr[i+1]);
        break;
        }
        if (i == (N - 1))
            printf ("Everything is OK!\n");
    }
}

// Main function to test merge sort algorithm
int main() {
    FILE *file;
    char filename[50];
    int *arr, size, count;

    printf("Enter the filename to open: ");
    count = scanf("%49s", filename);

    if (count != 1) {
        printf("Error: invalid input\n");
        exit(1);
    }

    file = fopen(filename, "rb");

    if (file == NULL) {
        printf("Error: cannot open file\n");
        exit(1);
    }

    fseek(file, 0, SEEK_END);
    size = ftell(file) / sizeof(int);
    fseek(file, 0, SEEK_SET);

    arr = malloc(size * sizeof(int));

    if (arr == NULL) {
        printf("Error: cannot allocate memory\n");
        exit(1);
    }

    count = fread(arr, sizeof(int), size, file);

    if (count != size) {
        printf("Error: could not read all integers from file\n");
        exit(1);
    }

    fclose(file);

    //int arr_size = sizeof(arr) / sizeof(int);

    mergeSort(arr, 0, size - 1);
    validateSort(arr, size);

    printf("\nSorted array is:\n");
    printArray(arr, size);

        free(arr);
    return 0;
}