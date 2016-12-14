/**
 * Copyright 2016 Alexander Grund
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "pmacc_types.hpp"
#include "memory/buffers/HostDeviceBuffer.hpp"
#include "random/RNGProvider.hpp"
#include "random/distributions/Uniform.hpp"
#include "random/methods/Xor.hpp"
#include "random/methods/XorMin.hpp"
#include "random/methods/MRG32k3a.hpp"
#include "random/methods/MRG32k3aMin.hpp"
#include "dimensions/DataSpace.hpp"
#include "assert.hpp"
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <limits>

typedef PMacc::DataSpace<DIM2> Space2D;
typedef PMacc::DataSpace<DIM3> Space3D;

struct RandomFiller
{
    template<class T_DataBox, class T_Random>
    DINLINE void operator()(T_DataBox box, Space2D boxSize, T_Random rand, uint32_t numSamples) const
    {
        const Space3D ownIdx = Space3D(threadIdx) + Space3D(blockIdx) * Space3D(blockDim);
        rand.init(ownIdx.shrink<2>());
        for(uint32_t i=0; i<numSamples; i++)
        {
            Space2D idx = rand(boxSize);
            atomicAdd(&box(idx), 1);
        }
    }
};

template<class T_RNGProvider>
struct GetRandomIdx
{
    typedef PMacc::random::distributions::Uniform<float> Distribution;
    typedef typename T_RNGProvider::GetRandomType<Distribution>::type Random;

    HINLINE GetRandomIdx(): rand(T_RNGProvider::template createRandom<Distribution>())
    {}

    DINLINE void
    init(Space2D globalCellIdx)
    {
        rand.init(globalCellIdx);
    }

    DINLINE Space2D
    operator()(Space2D size)
    {
        using PMacc::algorithms::math::float2int_rd;
        return Space2D(float2int_rd(rand() * size.x()), float2int_rd(rand() * size.y()));
    }
private:
    PMACC_ALIGN8(rand, Random);
};

/** Write in PGM grayscale file format (easy to read/interpret) */
template<class T_Buffer>
void writePGM(const std::string& filePath, T_Buffer& buffer)
{
    const Space2D size = buffer.getDataSpace();
    uint32_t maxVal = 0;
    for(int y=0; y<size.y(); y++)
    {
        for(int x=0; x<size.x(); x++)
        {
            uint32_t val = buffer.getDataBox()(Space2D(x, y));
            if(val > maxVal)
                maxVal = val;
        }
    }

    // Standard format is single byte per value which limits the range to 0-255
    // An extension allows 2 bytes so 0-65536)
    if(maxVal > std::numeric_limits<uint16_t>::max())
        maxVal = std::numeric_limits<uint16_t>::max();
    const bool isTwoByteFormat = maxVal > std::numeric_limits<uint8_t>::max();

    std::ofstream outFile(filePath.c_str());
    // TAG
    outFile << "P5\n";
    // Size and maximum value (at most 65536 which is 2 bytes per value)
    outFile << size.x() << " " << size.y() << " " << maxVal << "\n";
    for(int y=0; y<size.y(); y++)
    {
        for(int x=0; x<size.x(); x++)
        {
            uint32_t val = buffer.getDataBox()(Space2D(x, y));
            // Clip value
            if(val > maxVal)
                val = maxVal;
            // Write first byte (higher order bits) if file is in 2 byte format
            if(isTwoByteFormat)
                outFile << uint8_t(val >> 8);
            // Write remaining bytze
            outFile << uint8_t(val);
        }
    }
}

template<class T_DeviceBuffer, class T_Random>
void generateRandomNumbers(const Space2D& rngSize, uint32_t numSamples, T_DeviceBuffer& buffer, const T_Random& rand)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    Space2D blockSize(std::min(32, rngSize.x()), std::min(16, rngSize.y()));
    Space2D gridSize(rngSize / blockSize);

    CUDA_CHECK(cudaEventRecord(start));
    PMACC_KERNEL(RandomFiller{})(gridSize, blockSize)(buffer.getDataBox(), buffer.getDataSpace(), rand, numSamples);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Done in " << milliseconds << "ms" << std::endl;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

template<class T_Method>
void runTest(uint32_t numSamples)
{
    typedef PMacc::random::RNGProvider<2, T_Method> RNGProvider;

    const std::string rngName = RNGProvider::RNGMethod::getName();
    std::cout << std::endl << "Running test for " << rngName
              << " with " << numSamples << " samples per cell"
              << std::endl;
    // Size of the detector
    const Space2D size(256, 256);
    // Size of the rng provider (= number of states used)
    const Space2D rngSize(256, 256);

    PMacc::HostDeviceBuffer<uint32_t, 2> detector(size);
    RNGProvider rngProvider(rngSize);
    rngProvider.init(0x42133742);

    generateRandomNumbers(rngSize, numSamples, detector.getDeviceBuffer(), GetRandomIdx<RNGProvider>());

    detector.deviceToHost();
    PMACC_AUTO(box, detector.getHostBuffer().getDataBox());
    // Write data to file
    std::ofstream dataFile((rngName + "_data.txt").c_str());
    for(int y=0; y<size.y(); y++)
    {
        for(int x=0; x<size.x(); x++)
            dataFile << box(Space2D(x, y)) << ",";
    }
    writePGM(rngName + "_img.pgm", detector.getHostBuffer());

    uint64_t totalNumSamples = 0;
    double mean = 0;
    uint32_t maxVal = 0;
    uint32_t minVal = static_cast<uint32_t>(-1);
    for(int y=0; y<size.y(); y++)
    {
        for(int x=0; x<size.x(); x++)
        {
            Space2D idx(x, y);
            uint32_t val = box(idx);
            if(val > maxVal)
                maxVal = val;
            if(val < minVal)
                minVal = val;
            totalNumSamples += val;
            mean += PMacc::math::linearize(size.shrink<1>(1), idx) * static_cast<uint64_t>(val);
        }
    }
    PMACC_ASSERT(totalNumSamples == uint64_t(rngSize.productOfComponents()) * uint64_t(numSamples));
    // Expected value: (n-1)/2
    double Ex = (size.productOfComponents() - 1) / 2.;
    // Variance: (n^2 - 1) / 12
    double var = (PMacc::algorithms::math::pow<double>(size.productOfComponents(), 2) - 1.) / 12.;
    // Mean value
    mean /= totalNumSamples;
    double errSq = 0;
    // Calc standard derivation
    for(int y=0; y<size.y(); y++)
    {
        for(int x=0; x<size.x(); x++)
        {
            Space2D idx(x, y);
            uint32_t val = box(idx);
            errSq += val * PMacc::algorithms::math::pow<double>(PMacc::math::linearize(size.shrink<1>(1), idx) - mean, 2);
        }
    }
    double stdDev = sqrt(errSq/(totalNumSamples - 1));

    uint64_t avg = totalNumSamples/size.productOfComponents();
    std::cout << "  Samples: " << totalNumSamples << std::endl;
    std::cout << "      Min: " << minVal << std::endl;
    std::cout << "      Max: " << maxVal << std::endl;
    std::cout << " Avg/cell: " << avg << std::endl;
    std::cout << "     E(x): " << Ex << std::endl;
    std::cout << "     mean: " << mean << std::endl;
    std::cout << "   dev(x): " << sqrt(var) << std::endl;
    std::cout << " std. dev: " << stdDev << std::endl;
}

int main(int argc, char** argv)
{
    MPI_Init( &argc, &argv );
    PMacc::Environment<2>::get().initDevices(Space2D::create(1), Space2D::create(0));

    const uint32_t numSamples = (argc > 1) ? atoi(argv[1]) : 1000;

    runTest<PMacc::random::methods::Xor>(numSamples);
    runTest<PMacc::random::methods::XorMin>(numSamples);
    runTest<PMacc::random::methods::MRG32k3a>(numSamples);
    runTest<PMacc::random::methods::MRG32k3aMin>(numSamples);

    MPI_Finalize();
}
