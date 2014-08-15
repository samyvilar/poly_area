

CFLAGS=-Wall -Wfatal-errors -O2 -mtune=native -march=native
CC=gcc



build:
		$(CC) $(CFLAGS) -fPIC -shared poly_area.c -o libpoly_area.so


clean:
		rm -f *.o *.so *.pyc