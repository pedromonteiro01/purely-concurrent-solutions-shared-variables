#include <stdio.h>

int main() {
    unsigned char c[10];
    printf("char: ");
    scanf("%s", c);

    printf("utf-8 code: %X%X\n", c[0], c[1]);
    return 0;
}