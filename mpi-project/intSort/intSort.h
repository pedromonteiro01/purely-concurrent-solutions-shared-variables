// countWords.h

#ifndef INT_SORT_H
#define INT_SORT_H

#include <stdio.h>
#include <stdlib.h>


void printArray(int *arr, int size);

void mergeSort(int *arr, int left, int right);

void merge(int *arr, int left, int middle, int right);

int validateSort(int *arr, int N);


#endif // INT_SORT_H