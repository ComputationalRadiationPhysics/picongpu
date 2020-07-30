/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/alpaka.hpp>

#include <alpaka/test/MeasureKernelRunTime.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch.hpp>

#include <iostream>
#include <typeinfo>
#include <fstream>
#include <algorithm>

//#define ALPAKA_MANDELBROT_TEST_CONTINOUS_COLOR_MAPPING  // Define this to enable the continuous color mapping.

//#############################################################################
//! Complex Number.
template<
    typename T>
class SimpleComplex
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_HOST_ACC SimpleComplex(
        T const & a,
        T const & b) :
            r(a),
            i(b)
    {}
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_INLINE
    ALPAKA_FN_HOST_ACC auto absSq() const
    -> T
    {
        return r*r + i*i;
    }
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_HOST_ACC auto operator*(SimpleComplex const & a)
    -> SimpleComplex
    {
        return SimpleComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_HOST_ACC auto operator*(float const & a)
    -> SimpleComplex
    {
        return SimpleComplex(r*a, i*a);
    }
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_HOST_ACC auto operator+(SimpleComplex const & a)
    -> SimpleComplex
    {
        return SimpleComplex(r+a.r, i+a.i);
    }
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_HOST_ACC auto operator+(float const & a)
    -> SimpleComplex
    {
        return SimpleComplex(r+a, i);
    }

public:
    T r;
    T i;
};

//#############################################################################
//! A Mandelbrot kernel.
class MandelbrotKernel
{
public:
#ifndef ALPAKA_MANDELBROT_TEST_CONTINOUS_COLOR_MAPPING
    //-----------------------------------------------------------------------------
    ALPAKA_FN_HOST_ACC MandelbrotKernel()
    {
        // Banding can be prevented by a continuous color functions.
        m_colors[0u] = convertRgbSingleToBgra(66, 30, 15);
        m_colors[1u] = convertRgbSingleToBgra(25, 7, 26);
        m_colors[2u] = convertRgbSingleToBgra(9, 1, 47);
        m_colors[3u] = convertRgbSingleToBgra(4, 4, 73);
        m_colors[4u] = convertRgbSingleToBgra(0, 7, 100);
        m_colors[5u] = convertRgbSingleToBgra(12, 44, 138);
        m_colors[6u] = convertRgbSingleToBgra(24, 82, 177);
        m_colors[7u] = convertRgbSingleToBgra(57, 125, 209);
        m_colors[8u] = convertRgbSingleToBgra(134, 181, 229);
        m_colors[9u] = convertRgbSingleToBgra(211, 236, 248);
        m_colors[10u] = convertRgbSingleToBgra(241, 233, 191);
        m_colors[11u] = convertRgbSingleToBgra(248, 201, 95);
        m_colors[12u] = convertRgbSingleToBgra(255, 170, 0);
        m_colors[13u] = convertRgbSingleToBgra(204, 128, 0);
        m_colors[14u] = convertRgbSingleToBgra(153, 87, 0);
        m_colors[15u] = convertRgbSingleToBgra(106, 52, 3);
    }
#endif

    //-----------------------------------------------------------------------------
    //! \param acc The accelerator to be executed on.
    //! \param pColors The output image.
    //! \param numRows The number of rows in the image
    //! \param numCols The number of columns in the image.
    //! \param pitchBytes The pitch in bytes.
    //! \param fMinR The left border.
    //! \param fMaxR The right border.
    //! \param fMinI The bottom border.
    //! \param fMaxI The top border.
    //! \param maxIterations The maximum number of iterations.
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        std::uint32_t * const pColors,
        std::uint32_t const & numRows,
        std::uint32_t const & numCols,
        std::uint32_t const & pitchBytes,
        float const & fMinR,
        float const & fMaxR,
        float const & fMinI,
        float const & fMaxI,
        std::uint32_t const & maxIterations) const
    -> void
    {
        static_assert(
            alpaka::dim::Dim<TAcc>::value == 2,
            "The MandelbrotKernel expects 2-dimensional indices!");

        auto const gridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc));
        auto const & gridThreadIdxX(gridThreadIdx[1u]);
        auto const & gridThreadIdxY(gridThreadIdx[0u]);

        if((gridThreadIdxY < numRows) && (gridThreadIdxX < numCols))
        {
            SimpleComplex<float> c(
                (fMinR + (static_cast<float>(gridThreadIdxX)/float(numCols-1)*(fMaxR - fMinR))),
                (fMinI + (static_cast<float>(gridThreadIdxY)/float(numRows-1)*(fMaxI - fMinI))));

            auto const iterationCount(iterateMandelbrot(c, maxIterations));

            auto const pColorsRow(pColors + ((gridThreadIdxY * pitchBytes) / sizeof(std::uint32_t)));
            pColorsRow[gridThreadIdxX] =
#ifdef ALPAKA_MANDELBROT_TEST_CONTINOUS_COLOR_MAPPING
                iterationCountToContinousColor(iterationCount, maxIterations);
#else
                iterationCountToRepeatedColor(iterationCount);
#endif
        }
    }
    //-----------------------------------------------------------------------------
    //! \return The number of iterations until the Mandelbrot iteration with the given Value reaches the absolute value of 2.
    //!     Only does maxIterations steps and returns maxIterations if the value would be higher.
    ALPAKA_FN_ACC static auto iterateMandelbrot(
        SimpleComplex<float> const & c,
        std::uint32_t const & maxIterations)
    -> std::uint32_t
    {
        SimpleComplex<float> z(0.0f, 0.0f);
        for(std::uint32_t iterations(0); iterations<maxIterations; ++iterations)
        {
            z = z*z + c;
            if(z.absSq() > 4.0f)
            {
                return iterations;
            }
        }
        return maxIterations;
    }

    //-----------------------------------------------------------------------------
    ALPAKA_FN_HOST_ACC static auto convertRgbSingleToBgra(
        std::uint32_t const & r,
        std::uint32_t const & g,
        std::uint32_t const & b)
    -> std::uint32_t
    {
        return 0xFF000000 | (r<<16) | (g<<8) | b;
    }

#ifdef ALPAKA_MANDELBROT_TEST_CONTINOUS_COLOR_MAPPING
    //-----------------------------------------------------------------------------
    //! This uses a simple mapping from iteration count to colors.
    //! This leads to banding but allows a all pixels to be colored.
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC static auto iterationCountToContinousColor(
        std::uint32_t const & iterationCount,
        std::uint32_t const & maxIterations)
    -> std::uint32_t
    {
        // Map the iteration count on the 0..1 interval.
        float const t(static_cast<float>(iterationCount)/static_cast<float>(maxIterations));
        float const oneMinusT(1.0f-t);
        // Use some modified Bernstein polynomials for r, g, b.
        std::uint32_t const r(static_cast<std::uint32_t>(9.0f*oneMinusT*t*t*t*255.0f));
        std::uint32_t const g(static_cast<std::uint32_t>(15.0f*oneMinusT*oneMinusT*t*t*255.0f));
        std::uint32_t const b(static_cast<std::uint32_t>(8.5f*oneMinusT*oneMinusT*oneMinusT*t*255.0f));
        return convertRgbSingleToBgra(r, g, b);
    }
#else
    //-----------------------------------------------------------------------------
    //! This uses a simple mapping from iteration count to colors.
    //! This leads to banding but allows a all pixels to be colored.
    ALPAKA_FN_ACC auto iterationCountToRepeatedColor(
        std::uint32_t const & iterationCount) const
    -> std::uint32_t
    {
        return m_colors[iterationCount%16];
    }

    std::uint32_t m_colors[16];
#endif
};

//-----------------------------------------------------------------------------
//! Writes the buffer color data to a file.
template<
    typename TBuf>
auto writeTgaColorImage(
    std::string const & fileName,
    TBuf const & bufRgba)
-> void
{
    static_assert(
        alpaka::dim::Dim<TBuf>::value == 2,
        "The buffer has to be 2 dimensional!");
    static_assert(
        std::is_integral<alpaka::elem::Elem<TBuf>>::value,
        "The buffer element type has to be integral!");

    // The width of the input buffer is in input elements.
    auto const bufWidthElems(alpaka::extent::getWidth(bufRgba));
    auto const bufWidthBytes(bufWidthElems * sizeof(alpaka::elem::Elem<TBuf>));
    // The row width in bytes has to be dividable by 4 Bytes (RGBA).
    ALPAKA_ASSERT(bufWidthBytes % sizeof(std::uint32_t) == 0);
    // The number of colors in a row.
    auto const bufWidthColors(bufWidthBytes / sizeof(std::uint32_t));
    ALPAKA_ASSERT(bufWidthColors >= 1);
    auto const bufHeightColors(alpaka::extent::getHeight(bufRgba));
    ALPAKA_ASSERT(bufHeightColors >= 1);
    auto const bufPitchBytes(alpaka::mem::view::getPitchBytes<alpaka::dim::Dim<TBuf>::value - 1u>(bufRgba));
    ALPAKA_ASSERT(bufPitchBytes >= bufWidthBytes);

    std::ofstream ofs(
        fileName,
        std::ofstream::out | std::ofstream::binary);
    if(!ofs.is_open())
    {
        throw std::invalid_argument("Unable to open file: "+fileName);
    }

    // Write tga image header.
    ofs.put(0x00);                      // Number of Characters in Identification Field.
    ofs.put(0x00);                      // Color Map Type.
    ofs.put(0x02);                      // Image Type Code.
    ofs.put(0x00);                      // Color Map Origin.
    ofs.put(0x00);
    ofs.put(0x00);                      // Color Map Length.
    ofs.put(0x00);
    ofs.put(0x00);                      // Color Map Entry Size.
    ofs.put(0x00);                      // X Origin of Image.
    ofs.put(0x00);
    ofs.put(0x00);                      // Y Origin of Image.
    ofs.put(0x00);
    ofs.put(static_cast<char>(bufWidthColors & 0xFFu)); // Width of Image.
    ofs.put(static_cast<char>((bufWidthColors >> 8) & 0xFFu));
    ofs.put(static_cast<char>(bufHeightColors & 0xFFu));// Height of Image.
    ofs.put(static_cast<char>((bufHeightColors >> 8) & 0xFFu));
    ofs.put(0x20);                      // Image Pixel Size.
    ofs.put(0x20);                      // Image Descriptor Byte.

    // Write the data.
    char const * pData(reinterpret_cast<char const *>(alpaka::mem::view::getPtrNative(bufRgba)));
    // If there is no padding, we can directly write the whole buffer data ...
    if(bufPitchBytes == bufWidthBytes)
    {
        ofs.write(
            pData,
            static_cast<std::streamsize>(bufWidthBytes*bufHeightColors));
    }
    // ... else we have to write row by row.
    else
    {
        for(auto row(decltype(bufHeightColors)(0)); row<bufHeightColors; ++row)
        {
            ofs.write(
                pData + bufPitchBytes*row,
                static_cast<std::streamsize>(bufWidthBytes));
        }
    }
}

using TestAccs = alpaka::test::acc::EnabledAccs<
    alpaka::dim::DimInt<2u>,
    std::uint32_t>;

TEMPLATE_LIST_TEST_CASE( "mandelbrot", "[mandelbrot]", TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::dim::Dim<Acc>;
    using Idx = alpaka::idx::Idx<Acc>;

#ifdef ALPAKA_CI
    Idx const imageSize(1u<<5u);
#else
    Idx const imageSize(1u<<10u);
#endif
    Idx const numRows(imageSize);
    Idx const numCols(imageSize);
    float const fMinR(-2.0f);
    float const fMaxR(+1.0f);
    float const fMinI(-1.2f);
    float const fMaxI(+1.2f);
    Idx const maxIterations(300u);

    using Val = std::uint32_t;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using QueueAcc = alpaka::test::queue::DefaultQueue<DevAcc>;
    using PltfHost = alpaka::pltf::PltfCpu;

    // Create the kernel function object.
    MandelbrotKernel kernel;

    // Get the host device.
    auto const devHost(
        alpaka::pltf::getDevByIdx<PltfHost>(0u));

    // Select a device to execute on.
    auto const devAcc(
        alpaka::pltf::getDevByIdx<PltfAcc>(0u));

    // Get a queue on this device.
    QueueAcc queue(
        devAcc);

    alpaka::vec::Vec<Dim, Idx> const extent(
        static_cast<Idx>(numRows),
        static_cast<Idx>(numCols));

    // Let alpaka calculate good block and grid sizes given our full problem extent.
    alpaka::workdiv::WorkDivMembers<Dim, Idx> const workDiv(
        alpaka::workdiv::getValidWorkDiv<Acc>(
            devAcc,
            extent,
            alpaka::vec::Vec<Dim, Idx>::ones(),
            false,
            alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted));

    std::cout
        << "MandelbrotKernel("
        << " numRows:" << numRows
        << ", numCols:" << numCols
        << ", maxIterations:" << maxIterations
        << ", accelerator: " << alpaka::acc::getAccName<Acc>()
        << ", kernel: " << typeid(kernel).name()
        << ", workDiv: " << workDiv
        << ")" << std::endl;

    // allocate host memory
    auto bufColorHost(
        alpaka::mem::buf::alloc<Val, Idx>(devHost, extent));

    // Allocate the buffer on the accelerator.
    auto bufColorAcc(
        alpaka::mem::buf::alloc<Val, Idx>(devAcc, extent));

    // Copy Host -> Acc.
    alpaka::mem::view::copy(queue, bufColorAcc, bufColorHost, extent);

    // Create the kernel execution task.
    auto const taskKernel(alpaka::kernel::createTaskKernel<Acc>(
        workDiv,
        kernel,
        alpaka::mem::view::getPtrNative(bufColorAcc),
        numRows,
        numCols,
        alpaka::mem::view::getPitchBytes<1u>(bufColorAcc),
        fMinR,
        fMaxR,
        fMinI,
        fMaxI,
        maxIterations));

    // Profile the kernel execution.
    std::cout << "Execution time: "
        << alpaka::test::integ::measureTaskRunTimeMs(
            queue,
            taskKernel)
        << " ms"
        << std::endl;

    // Copy back the result.
    alpaka::mem::view::copy(queue, bufColorHost, bufColorAcc, extent);

    // Wait for the queue to finish the memory operation.
    alpaka::wait::wait(queue);

    // Write the image to a file.
    std::string fileName("mandelbrot"+std::to_string(numCols)+"x"+std::to_string(numRows)+"_"+alpaka::acc::getAccName<Acc>()+".tga");
    std::replace(fileName.begin(), fileName.end(), '<', '_');
    std::replace(fileName.begin(), fileName.end(), '>', '_');
    writeTgaColorImage(
        fileName,
        bufColorHost);
}
