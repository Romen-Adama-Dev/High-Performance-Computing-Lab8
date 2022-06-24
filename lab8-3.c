#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>

//GPU function (kernel)

__global__ void add(int *a, int *b, int *c) 
{
	*c = *a + *b;
}

int main(void) {
	int a, b, c; // host side tables
	int *d_a, *d_b, *d_c; // device side tables
	int size = sizeof(int);
	// CUDA device memory allocation
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	// Example values
	a = 2;
	b = 7;
	// Copy data to device
	cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
	// Run kernel - 1 block - 1 thread
	add<<<1,1>>>(d_a, d_b, d_c);
	// Copy data from device
	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
	// Cleaning
	printf("%d+%d=%d\n",a,b,c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	return 0;
}