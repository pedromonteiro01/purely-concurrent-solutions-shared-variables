#include <stdio.h>
#include <stdlib.h>

void merge(int arr[], int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    int L[n1], R[n2]; // create temporary arrays

    // copy data to temporary arrays
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    i = 0; // initial index of first subarray
    j = 0; // initial index of second subarray
    k = l; // initial index of merged subarray

    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // copy remaining elements of L
    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }

    // copy remaining elements of R
    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(int arr[], int l, int r)
{
    if (l < r)
    {
        // same as (l+r)/2, but avoids overflow for large l and h
        int m = l + (r - l) / 2;

        // sort first and second halves
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);

        // merge the sorted halves
        merge(arr, l, m, r);
    }
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("Usage: %s <filename>\n", argv[0]);
        return 1;
    }

    // read integers from binary file
    FILE *file = fopen(argv[1], "rb");
    if (file == NULL) // check if file is empty
    {
        perror("Error opening file");
        return 1;
    }

    int n_values;
    fread(&n_values, sizeof(int), 1, file); // read number of values from the file (first int in the file)
    printf("Number of values: %d\n", n_values);

    // allocate memory with size of number of values
    int *arr = (int *)malloc(n_values * sizeof(int));
    if (arr == NULL)
    {
        perror("Error allocating memory for the array");
        return 1;
    }

    fread(arr, sizeof(int), n_values, file); // read the rest of the integers
    fclose(file);

    // apply merge sort
    mergeSort(arr, 0, n_values - 1);

    for (int i = 0; i < n_values; i++)
    {
        printf("%d ", arr[i]);
    }

    printf("\n");

    // free memory
    free(arr);

    return 0;
}
