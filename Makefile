

CFLAGS=-Wall -Wfatal-errors -Ofast -mtune=native -march=native
CC=gcc



build:
		$(CC) $(CFLAGS) -fPIC -shared poly_area.c -o libpoly_area.so


clean:
		rm -f *.o *.so