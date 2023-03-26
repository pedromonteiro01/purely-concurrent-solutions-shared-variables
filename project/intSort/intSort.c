#include <stdio.h>
#include <stdlib.h>

/** \brief merge two sorted subarrays */
void merge(int *arr, int left, int middle, int right);

/** \brief recursively sort array using merge sort */
void mergeSort(int *arr, int left, int right);

/** \brief print the sorted array */
void printArray(int *arr, int size);

/** \brief check if the array of integers has been sorted correctly */
int validateSort(int *arr, int N);

/**
 *  \brief Fuction merge.
 *
 *  Its role to merge two sorted subarrays
 *
 *  \param arr pointer to the integer array to be sorted
 *  \param left integer representing the starting index of the left subarray
 *  \param middle integer representing the ending index of the left subarray and the starting index of the right subarray
 *  \param right integer representing the ending index of the right subarray
 *
 */
void merge(int *arr, int left, int middle, int right)
{
    int i, j, k;

    // Find sizes of two subarrays
    int left_size = middle - left + 1;
    int right_size = right - middle;

    // Create temporary arrays for left and right subarrays
    int *left_arr = malloc(left_size * sizeof(int));
    int *right_arr = malloc(right_size * sizeof(int));

    // Copy data to temporary arrays
    for (i = 0; i < left_size; i++)
    {
        left_arr[i] = arr[left + i];
    }
    for (j = 0; j < right_size; j++)
    {
        right_arr[j] = arr[middle + 1 + j];
    }

    // Merge the two subarrays into arr
    i = 0;
    j = 0;
    k = left;
    while (i < left_size && j < right_size)
    {
        if (left_arr[i] <= right_arr[j])
        {
            arr[k] = left_arr[i];
            i++;
        }
        else
        {
            arr[k] = right_arr[j];
            j++;
        }
        k++;
    }

    // Copy remaining elements of left subarray, if any
    while (i < left_size)
    {
        arr[k] = left_arr[i];
        i++;
        k++;
    }

    // Copy remaining elements of right subarray, if any
    while (j < right_size)
    {
        arr[k] = right_arr[j];
        j++;
        k++;
    }

    // Free temporary arrays
    free(left_arr);
    free(right_arr);
}

/**
 *  \brief Fuction mergeSort.
 *
 *  Its role to recursively sort array using merge sort
 *
 *  \param arr pointer to an array of integers that needs to be sorted
 *  \param left integer representing the leftmost index of the subarray that needs to be sorted
 *  \param right integer representing the rightmost index of the subarray that needs to be sorted
 *
 */
void mergeSort(int *arr, int left, int right)
{
    if (left < right)
    {
        int middle = left + (right - left) / 2;
        mergeSort(arr, left, middle);
        mergeSort(arr, middle + 1, right);
        merge(arr, left, middle, right);
    }
}

/**
 *  \brief Fuction printArray.
 *
 *  Its role to print the sorted array
 *
 *  \param arr pointer to the first element of the array that needs to be printed
 *  \param size  integer that represents the number of elements in the array
 *
 */
void printArray(int *arr, int size)
{
    int i;
    for (i = 0; i < size; i++)
    {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

/**
 *  \brief Fuction validateSort.
 *
 *  Its role to check if the array of integers has been sorted correctly, in ascending order
 *
 *  \param arr pointer to an integer array containing the elements to be validated
 *  \param size integer representing the size of the array
 *
 */
int validateSort(int *arr, int N)
{
    int i;

    for (i = 0; i < N - 1; i++)
    {
        if (arr[i] > arr[i + 1])
        {
            printf("Error in position %d between element %d and %d\n", i, arr[i], arr[i + 1]);
            return 0;
        }
        if (i == (N - 1))
            printf("Everything is OK!\n");
    }

    return 1;
}