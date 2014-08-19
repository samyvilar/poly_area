
build: 
	$(MAKE) -C c

clean: 
	rm *.pyc 
	$(MAKE) -C c clean


timings:
	./benchmark.py


test:
	nosetests
