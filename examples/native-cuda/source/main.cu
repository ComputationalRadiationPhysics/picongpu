/*
  mallocMC: Memory Allocator for Many Core Architectures.
  https://www.hzdr.de/crp

  Copyright 2025 Institute of Radiation Physics,
                 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Julian Lenz - j.lenz ( at ) hzdr.de

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/

#include "mallocMC/span.hpp"

#include <mallocMC/mallocMC.cuh>

#include <cstdint>
#include <cstdlib>
#include <functional>

/**
 * @brief Computes the sum of squares of the first `n` natural numbers.
 *
 * This function calculates the sum of squares of the first `n` natural numbers using the formula:
 * \[
 * \text{sumOfSquares}(n) = \frac{n \times (n + 1) \times (2n + 1)}{6}
 * \]
 * It's used to check the computed value in the kernel.
 *
 * @param n The number of natural numbers to consider.
 * @return The sum of squares of the first `n` natural numbers.
 */
__device__ auto sumOfSquares(auto const n)
{
    return (n * (n + 1) * (2 * n + 1)) / 6;
}

/**
 * @brief Computes the dot product of two vectors for each thread.
 *
 * This kernel computes the dot product of two vectors, `a` and `b`, for each thread.
 * Each thread allocates memory for its own vectors, initializes them with consecutive values,
 * computes the dot product, and checks if the result matches the expected value.
 * If the result does not match, the thread prints an error message and halts execution.
 *
 * @param memoryManager A CUDA memory manager object used for memory allocation and deallocation.
 * @param numValues The number of elements in each vector.
 *
 * @note This kernnel is, of course, not very realistic as a workload but it fulfills its purpose of showcasing a
 * native CUDA application.
 */
__global__ void oneDotProductPerThread(mallocMC::CudaMemoryManager<> memoryManager, uint64_t numValues)
{
    using mallocMC::span;
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Not very realistic, all threads are doing this on their own:
    auto a
        = span<uint64_t>(reinterpret_cast<uint64_t*>(memoryManager.malloc(numValues * sizeof(uint64_t))), numValues);
    auto b
        = span<uint64_t>(reinterpret_cast<uint64_t*>(memoryManager.malloc(numValues * sizeof(uint64_t))), numValues);

    std::iota(std::begin(a), std::end(a), tid);
    std::iota(std::begin(b), std::end(b), tid);

    uint64_t result = std::transform_reduce(std::cbegin(a), std::cend(a), std::cbegin(b), 0U);

    auto expected = sumOfSquares(numValues + tid - 1) - (tid > 0 ? sumOfSquares(tid - 1) : 0);
    if(result != expected)
    {
        printf("Thread %lu: Result %lu != Expected %lu. \n", tid, result, expected);
        __trap();
    }

    memoryManager.free(a.data());
    memoryManager.free(b.data());
}

int main()
{
    size_t const heapSize = 1024U * 1024U * 1024U;
    uint64_t const numValues = 32U;
    mallocMC::CudaHostInfrastructure<> hostInfrastructure{heapSize};
    auto memoryManager = mallocMC::CudaMemoryManager{hostInfrastructure};

    std::cout << "Running native CUDA kernel." << std::endl;
    oneDotProductPerThread<<<8, 256>>>(memoryManager, numValues);
}
