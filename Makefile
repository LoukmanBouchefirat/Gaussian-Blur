all:
	gcc -o serial serial.c -lm
	gcc -o openmp openmp.c -fopenmp -lm

