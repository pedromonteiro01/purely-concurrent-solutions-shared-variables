CC = gcc
CFLAGS = -Wall -Wextra -Werror -g -O3
LDFLAGS = -lpthread
TARGET = main

SRCS = main.c ../utils/file_splitter.c countWords.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET)
