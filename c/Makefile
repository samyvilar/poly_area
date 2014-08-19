
CC=gcc
CFLAGS=-Wall -Wfatal-errors -fPIC

cfiles=poly_area.c poly_area_sse.c poly_area_avx.c poly_area_thrd.c
ifeq ($(CC),icc)
CFLAGS+=-dynamiclib -O2
else
CFLAGS+=-mtune=native -march=native -shared -O2
endif




build:
		$(CC) $(CFLAGS) $(cfiles) -o libpoly_area.so


clean:
		rm -f *.o *.so *.pyc