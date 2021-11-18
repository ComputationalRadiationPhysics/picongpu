/* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/** @file Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <iostream> //std:cout

#include <stdio.h>
// For the CUDA runtime routines (prefixed with "cupla_")
#include <cupla.hpp>
// Timer for test purpose
#include <chrono>
#include <vector>

#include <boost/lexical_cast.hpp>
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
struct vectorAdd
{
    template<typename T_Acc>
    ALPAKA_FN_HOST_ACC void operator()(
        T_Acc const& acc,
        const float* A,
        const float* B,
        float* C,
        const int numElements) const
    {
        int begin = cupla::blockDim(acc).x * cupla::blockIdx(acc).x * cupla::threadDim(acc).x
            + cupla::threadIdx(acc).x * cupla::threadDim(acc).x;
        if(begin < numElements)
        {
            int end = (begin + cupla::threadDim(acc).x < numElements) ? begin + cupla::threadDim(acc).x : numElements;
            for(int i = begin; i < end; ++i)
            {
                C[i] = A[i] + B[i];
            }
        }
    }
};

void benchmarkTest(int first, int last, int stepSize);
/**
 * Host main routine
 */
int main(int argc, char* argv[])
{
    // Error code to check return values for CUDA calls
    cuplaError_t err = cuplaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A
    float* h_A = (float*) malloc(size);

    // Allocate the host input vector B
    float* h_B = (float*) malloc(size);

    // Allocate the host output vector C
    float* h_C = (float*) malloc(size);

    // Verify that allocations succeeded
    if(h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for(int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand() / (float) RAND_MAX;
        h_B[i] = rand() / (float) RAND_MAX;
    }

    // Allocate the device input vector A
    float* d_A = NULL;
    err = cuplaMalloc((void**) &d_A, size);

    if(err != cuplaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cuplaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float* d_B = NULL;
    err = cuplaMalloc((void**) &d_B, size);

    if(err != cuplaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cuplaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float* d_C = NULL;
    err = cuplaMalloc((void**) &d_C, size);

    if(err != cuplaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cuplaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cuplaMemcpy(d_A, h_A, size, cuplaMemcpyHostToDevice);

    if(err != cuplaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cuplaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cuplaMemcpy(d_B, h_B, size, cuplaMemcpyHostToDevice);

    if(err != cuplaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cuplaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    CUPLA_KERNEL_OPTI(vectorAdd)(blocksPerGrid, threadsPerBlock, 0, 0)(d_A, d_B, d_C, numElements);
    err = cuplaGetLastError();

    if(err != cuplaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cuplaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cuplaMemcpy(h_C, d_C, size, cuplaMemcpyDeviceToHost);

    if(err != cuplaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cuplaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for(int i = 0; i < numElements; ++i)
    {
        if(fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device global memory
    err = cuplaFree(d_A);

    if(err != cuplaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cuplaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cuplaFree(d_B);

    if(err != cuplaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cuplaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cuplaFree(d_C);

    if(err != cuplaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cuplaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset the device and exit
    // cuplaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cuplaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cuplaDeviceReset();

    if(err != cuplaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cuplaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("Done\n");

    using boost::bad_lexical_cast;
    using boost::lexical_cast;
    std::vector<int> args;
    while(*++argv)
    {
        try
        {
            args.push_back(lexical_cast<int>(*argv));
        }
        catch(const bad_lexical_cast&)
        {
            args.push_back(0);
        }
    }
    // run benchmartest
    int first = 50000;
    int last = 100000;
    int stepSize = 50000;
    if(args.size() > 1)
    {
        first = args[0];
        last = args[1];
    }
    if(args.size() > 2)
    {
        stepSize = args[2];
    }
    benchmarkTest(first, last, stepSize);
    cuplaDeviceReset();
    return 0;
}

void benchmarkTest(int first, int last, int stepSize)
{
    for(int numElements = first; numElements <= last; numElements += stepSize)
    {
        std::cout << "N= " << numElements << "; ";
        size_t size = numElements * sizeof(float);
        // alloc host memory
        float* h_A = (float*) malloc(size);
        float* h_B = (float*) malloc(size);
        // init
        for(int i = 0; i < numElements; ++i)
        {
            h_A[i] = rand() / (float) RAND_MAX;
            h_B[i] = rand() / (float) RAND_MAX;
        }
        // alloc device memory
        float* d_A = NULL;
        cuplaMalloc((void**) &d_A, size);
        float* d_B = NULL;
        cuplaMalloc((void**) &d_B, size);
        float* d_C = NULL;
        cuplaMalloc((void**) &d_C, size);

        // copy host device
        cuplaMemcpy(d_A, h_A, size, cuplaMemcpyHostToDevice);
        cuplaMemcpy(d_B, h_B, size, cuplaMemcpyHostToDevice);

        int threadsPerBlock = 1024;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

        // Run Kernel
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

        CUPLA_KERNEL_OPTI(vectorAdd)(blocksPerGrid, threadsPerBlock, 0, 0)(d_A, d_B, d_C, numElements);
        cuplaDeviceSynchronize();

        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

        std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms"
                  << std::endl;
        // Free Device memory
        cuplaFree(d_A);
        cuplaFree(d_B);
        cuplaFree(d_C);
    }
}
