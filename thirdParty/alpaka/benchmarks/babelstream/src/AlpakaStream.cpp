// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code
//
// Cupla version created by Jeff Young in 2021
// Ported from cupla to alpaka by Bernhard Manfred Gruber in 2022

#include "AlpakaStream.h"

#include <numeric>

namespace
{
    constexpr auto blockSize = 1024;
    constexpr auto dotBlockSize = 256;
} // namespace

template<typename T>
AlpakaStream<T>::AlpakaStream(Idx arraySize, Idx deviceIndex)
    : arraySize(arraySize)
    , devHost(alpaka::getDevByIdx(platformHost, 0))
    , devAcc(alpaka::getDevByIdx(platformAcc, deviceIndex))
    , sums(alpaka::allocBuf<T, Idx>(devHost, dotBlockSize))
    , d_a(alpaka::allocBuf<T, Idx>(devAcc, arraySize))
    , d_b(alpaka::allocBuf<T, Idx>(devAcc, arraySize))
    , d_c(alpaka::allocBuf<T, Idx>(devAcc, arraySize))
    , d_sum(alpaka::allocBuf<T, Idx>(devAcc, dotBlockSize))
    , queue(devAcc)
{
    if(arraySize % blockSize != 0)
        throw std::runtime_error("Array size must be a multiple of " + std::to_string(blockSize));
    std::cout << "Using alpaka device " << alpaka::getName(devAcc) << std::endl;
}

struct InitKernel
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T* a, T* b, T* c, T initA, T initB, T initC) const
    {
        auto const [i] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        a[i] = initA;
        b[i] = initB;
        c[i] = initC;
    }
};

template<typename T>
void AlpakaStream<T>::init_arrays(T initA, T initB, T initC)
{
    auto const workdiv = WorkDiv{arraySize / blockSize, blockSize, 1};
    // auto const workdiv = alpaka::getValidWorkDiv(devAcc, arraySize);
    alpaka::exec<
        Acc>(queue, workdiv, InitKernel{}, std::data(d_a), std::data(d_b), std::data(d_c), initA, initB, initC);
    alpaka::wait(queue);
}

template<typename T>
void AlpakaStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
    alpaka::memcpy(queue, alpaka::createView(devHost, a), d_a);
    alpaka::memcpy(queue, alpaka::createView(devHost, b), d_b);
    alpaka::memcpy(queue, alpaka::createView(devHost, c), d_c);
}

struct CopyKernel
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* a, T* c) const
    {
        auto const [i] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        c[i] = a[i];
    }
};

template<typename T>
void AlpakaStream<T>::copy()
{
    auto const workdiv = WorkDiv{arraySize / blockSize, blockSize, 1};
    // auto const workdiv = alpaka::getValidWorkDiv(devAcc, arraySize);
    alpaka::exec<Acc>(queue, workdiv, CopyKernel{}, std::data(d_a), std::data(d_c));
    alpaka::wait(queue);
}

struct MulKernel
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T* b, T const* c) const
    {
        const T scalar = startScalar;
        auto const [i] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        b[i] = scalar * c[i];
    }
};

template<typename T>
void AlpakaStream<T>::mul()
{
    auto const workdiv = WorkDiv{arraySize / blockSize, blockSize, 1};
    // auto const workdiv = alpaka::getValidWorkDiv(devAcc, arraySize);
    alpaka::exec<Acc>(queue, workdiv, MulKernel{}, std::data(d_b), std::data(d_c));
    alpaka::wait(queue);
}

struct AddKernel
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* a, T const* b, T* c) const
    {
        auto const [i] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        c[i] = a[i] + b[i];
    }
};

template<typename T>
void AlpakaStream<T>::add()
{
    auto const workdiv = WorkDiv{arraySize / blockSize, blockSize, 1};
    // auto const workdiv = alpaka::getValidWorkDiv(devAcc, arraySize);
    alpaka::exec<Acc>(queue, workdiv, AddKernel{}, std::data(d_a), std::data(d_b), std::data(d_c));
    alpaka::wait(queue);
}

struct TriadKernel
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T* a, T const* b, T const* c) const
    {
        const T scalar = startScalar;
        auto const [i] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        a[i] = b[i] + scalar * c[i];
    }
};

template<typename T>
void AlpakaStream<T>::triad()
{
    auto const workdiv = WorkDiv{arraySize / blockSize, blockSize, 1};
    // auto const workdiv = alpaka::getValidWorkDiv(devAcc, arraySize);
    alpaka::exec<Acc>(queue, workdiv, TriadKernel{}, std::data(d_a), std::data(d_b), std::data(d_c));
    alpaka::wait(queue);
}

struct NstreamKernel
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T* a, T const* b, T const* c) const
    {
        const T scalar = startScalar;
        auto const [i] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        a[i] += b[i] + scalar * c[i];
    }
};

template<typename T>
void AlpakaStream<T>::nstream()
{
    auto const workdiv = WorkDiv{arraySize / blockSize, blockSize, 1};
    // auto const workdiv = alpaka::getValidWorkDiv(devAcc, arraySize);
    alpaka::exec<Acc>(queue, workdiv, NstreamKernel{}, std::data(d_a), std::data(d_b), std::data(d_c));
    alpaka::wait(queue);
}

struct DotKernel
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* a, T const* b, T* sum, int arraySize) const
    {
        // TODO(Jeff Young) - test if sharedMem bug is affecting performance here
        auto& tbSum = alpaka::declareSharedVar<T[blockSize], __COUNTER__>(acc);

        auto [i] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const [local_i] = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const [totalThreads] = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        T threadSum = 0;
        for(; i < arraySize; i += totalThreads) // NOLINT(bugprone-infinite-loop)
            threadSum += a[i] * b[i];
        tbSum[local_i] = threadSum;

        auto const [blockDim] = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        for(int offset = blockDim / 2; offset > 0; offset /= 2)
        {
            alpaka::syncBlockThreads(acc);
            if(local_i < offset)
                tbSum[local_i] += tbSum[local_i + offset];
        }

        auto const [blockIdx] = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
        if(local_i == 0)
            sum[blockIdx] = tbSum[local_i];
    }
};

template<typename T>
auto AlpakaStream<T>::dot() -> T
{
    auto const workdiv = WorkDiv{dotBlockSize, blockSize, 1};
    // auto const workdiv = alpaka::getValidWorkDiv(devAcc, dotBlockSize * blockSize);
    alpaka::exec<Acc>(queue, workdiv, DotKernel{}, std::data(d_a), std::data(d_b), std::data(d_sum), arraySize);
    alpaka::wait(queue);

    alpaka::memcpy(queue, sums, d_sum);
    T const* sumPtr = std::data(sums);
    // TODO(bgruber): replace by std::reduce, when gcc 9.3 is the baseline
    return std::accumulate(sumPtr, sumPtr + dotBlockSize, T{0});
}

void listDevices()
{
    auto const platform = alpaka::Platform<Acc>{};
    auto const count = alpaka::getDevCount(platform);
    std::cout << "Devices:" << std::endl;
    for(int i = 0; i < count; i++)
        std::cout << i << ": " << getDeviceName(i) << std::endl;
}

auto getDeviceName(int deviceIndex) -> std::string
{
    auto const platform = alpaka::Platform<Acc>{};
    return alpaka::getName(alpaka::getDevByIdx(platform, deviceIndex));
}

auto getDeviceDriver([[maybe_unused]] int device) -> std::string
{
    return "Not supported";
}

template class AlpakaStream<float>;
template class AlpakaStream<double>;
