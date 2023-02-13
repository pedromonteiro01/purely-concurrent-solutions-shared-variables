#include <stdio.h>
#include <string.h>

int main() {
  char word[100];
  int count;

  printf("Enter a word: ");
  scanf("%s", word);

  count = strlen(word);

  printf("The word contains %d characters.\n", count);

  return 0;
}