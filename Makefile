
CC=gcc
CFLAGS=-Wall -Wfatal-errors -fPIC
ifeq ($(CC),icc)
CFLAGS+=-dynamiclib -O2
else
CFLAGS+=-mtune=native -march=native -shared -O2
endif




build:
		$(CC) $(CFLAGS) poly_area.c -o libpoly_area.so


clean:
		rm -f *.o *.so *.pyc