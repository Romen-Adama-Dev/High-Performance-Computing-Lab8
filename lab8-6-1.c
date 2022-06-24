//Parallelization - several blocks - several threads + balance
#include <stdio.h> include <stdlib.h> include <string.h> include <cuda.h> include <iostream> include
#<cuda_runtime.h> include <chrono>

#define N 30 define M 8 //Threads per block

__global__ void add(int *a, int *b, int *c, int n) { int index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index<n)
                c[index] = a[index] + b[index];
}

void random (int *tab, int wym ) { int i; for(i=0;i<wym;i++) tab[i]=rand()%101;
}

// Print device properties
void printDevProp(cudaDeviceProp devProp) { printf("Major revision number: %d\n", devProp.major);
    printf("Minor revision number: %d\n", devProp.minor); printf("Name: %s\n", devProp.name); printf("Total
    global memory: %lu\n", devProp.totalGlobalMem); printf("Total shared memory per block: %lu\n",
    devProp.sharedMemPerBlock); printf("Total registers per block: %d\n", devProp.regsPerBlock); printf("Warp
    size: %d\n", devProp.warpSize); printf("Maximum memory pitch: %lu\n", devProp.memPitch); printf("Maximum
    threads per block: %d\n", devProp.maxThreadsPerBlock); for (int i = 0; i < 3; ++i) printf("Maximum
    dimension %d of block: %d\n", i, devProp.maxThreadsDim[i]); for (int i = 0; i < 3; ++i) printf("Maximum
    dimension %d of grid: %d\n", i, devProp.maxGridSize[i]); printf("Clock rate: %d\n", devProp.clockRate);
    printf("Total constant memory: %lu\n", devProp.totalConstMem); printf("Texture alignment: %lu\n",
    devProp.textureAlignment); printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" :
    "No")); printf("Number of multiprocessors: %d\n", devProp.multiProcessorCount); printf("Kernel execution
    timeout: %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No")); return;
}


int main(void) { int *a, *b, *c; // host copies of a, b, c int *d_a, *d_b, *d_c; // device copies of a, b, c
        int size = N * sizeof(int); int i; srand(time(NULL));
        // Allocate space for device copies of a, b, c
        cudaMalloc((void **)&d_a, size); cudaMalloc((void **)&d_b, size); cudaMalloc((void **)&d_c, size);
        // Alloc space for host copies of a, b, c and setup input values
        a = (int *)malloc(size); random(a, N); b = (int *)malloc(size); random(b, N); c = (int
        *)malloc(size);
        // Copy inputs to device
        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice); cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
        // Launch add() kernel on GPU
        add<<<(N+M-1)/M,M>>>(d_a, d_b, d_c, N);
        // Copy result back to host
        cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

        for(i=0;i<N;i++) { printf("a[%d](%d) + b[%d](%d) = c[%d](%d)\n",i,a[i],i,b[i],i,c[i]);
        }
        // Cleanup printf("%d+%d=%d\n",a,b,c);
        free(a); free(b); free(c); cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

        double sum = 0; double add = 1;

// Start measuring time
        auto begin = std::chrono::high_resolution_clock::now();

        int iterations = 1000*1000*1000; for (int i=0; i<iterations; i++) { sum += add; add /= 2.0;
        }

        // Stop measuring time and calculate the elapsed time
        auto end = std::chrono::high_resolution_clock::now(); auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

        printf("Result: %.20f\n", sum);

        printf("Time measured: %.3f seconds.\n", elapsed.count() * 1e-9);

        // Number of CUDA devices
        int devCount; cudaGetDeviceCount(&devCount); printf("CUDA Device Query...\n"); printf("There are %d
        CUDA devices.\n", devCount);

        // Iterate through devices
        for (int i = 0; i < devCount; ++i)
        {
                // Get device properties
                printf("\nCUDA Device #%d\n", i);
                cudaDeviceProp devProp;
                cudaGetDeviceProperties(&devProp, i);
                printDevProp(devProp);
        }

        printf("\nPress any key to exit...");
        char c2;
        scanf("%c2", &c2);

        return 0;


}