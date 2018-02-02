/**
 * \file
 * Copyright 2014-2015 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/test/MeasureKernelRunTime.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/stream/Stream.hpp>

#include <iostream>
#include <typeinfo>
#include <cassert>
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
    ALPAKA_FN_HOST_ACC auto absSq()
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

            auto const pColorsRow(reinterpret_cast<std::uint32_t *>(reinterpret_cast<std::uint8_t *>(pColors) + gridThreadIdxY * pitchBytes));
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
    assert(bufWidthBytes % sizeof(std::uint32_t) == 0);
    // The number of colors in a row.
    auto const bufWidthColors(bufWidthBytes / sizeof(std::uint32_t));
    assert(bufWidthColors >= 1);
    auto const bufHeightColors(alpaka::extent::getHeight(bufRgba));
    assert(bufHeightColors >= 1);
    auto const bufPitchBytes(alpaka::mem::view::getPitchBytes<alpaka::dim::Dim<TBuf>::value - 1u>(bufRgba));
    assert(bufPitchBytes >= bufWidthBytes);

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

//#############################################################################
//! Profiles the Mandelbrot kernel.
struct MandelbrotKernelTester
{
    template<
        typename TAcc,
        typename TSize>
    auto operator()(
        TSize const & numRows,
        TSize const & numCols,
        float const & fMinR,
        float const & fMaxR,
        float const & fMinI,
        float const & fMaxI,
        TSize const & maxIterations)
    -> void
    {
        std::cout << std::endl;
        std::cout << "################################################################################" << std::endl;

        using Val = std::uint32_t;
        using DevAcc = alpaka::dev::Dev<TAcc>;
        using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
        using StreamAcc = alpaka::test::stream::DefaultStream<DevAcc>;
        using PltfHost = alpaka::pltf::PltfCpu;

        // Create the kernel function object.
        MandelbrotKernel kernel;

        // Get the host device.
        auto const devHost(
            alpaka::pltf::getDevByIdx<PltfHost>(0u));

        // Select a device to execute on.
        auto const devAcc(
            alpaka::pltf::getDevByIdx<PltfAcc>(0u));

        // Get a stream on this device.
        StreamAcc stream(
            devAcc);

        alpaka::vec::Vec<alpaka::dim::DimInt<2u>, TSize> const extent(
            static_cast<TSize>(numRows),
            static_cast<TSize>(numCols));

        // Let alpaka calculate good block and grid sizes given our full problem extent.
        alpaka::workdiv::WorkDivMembers<alpaka::dim::DimInt<2u>, TSize> const workDiv(
            alpaka::workdiv::getValidWorkDiv<TAcc>(
                devAcc,
                extent,
                alpaka::vec::Vec<alpaka::dim::DimInt<2u>, TSize>::ones(),
                false,
                alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted));

        std::cout
            << "MandelbrotKernelTester("
            << " numRows:" << numRows
            << ", numCols:" << numCols
            << ", maxIterations:" << maxIterations
            << ", accelerator: " << alpaka::acc::getAccName<TAcc>()
            << ", kernel: " << typeid(kernel).name()
            << ", workDiv: " << workDiv
            << ")" << std::endl;

        // allocate host memory
        auto bufColorHost(
            alpaka::mem::buf::alloc<Val, TSize>(devHost, extent));

        // Allocate the buffer on the accelerator.
        auto bufColorAcc(
            alpaka::mem::buf::alloc<Val, TSize>(devAcc, extent));

        // Copy Host -> Acc.
        alpaka::mem::view::copy(stream, bufColorAcc, bufColorHost, extent);

        // Create the executor task.
        auto const exec(alpaka::exec::create<TAcc>(
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
            << alpaka::test::integ::measureKernelRunTimeMs(
                stream,
                exec)
            << " ms"
            << std::endl;

        // Copy back the result.
        alpaka::mem::view::copy(stream, bufColorHost, bufColorAcc, extent);

        // Wait for the stream to finish the memory operation.
        alpaka::wait::wait(stream);

        // Write the image to a file.
        std::string fileName("mandelbrot"+std::to_string(numCols)+"x"+std::to_string(numRows)+"_"+alpaka::acc::getAccName<TAcc>()+".tga");
        std::replace(fileName.begin(), fileName.end(), '<', '_');
        std::replace(fileName.begin(), fileName.end(), '>', '_');
        writeTgaColorImage(
            fileName,
            bufColorHost);

        std::cout << "################################################################################" << std::endl;
    }
};

auto main()
-> int
{
    try
    {
        std::cout << std::endl;
        std::cout << "################################################################################" << std::endl;
        std::cout << "                            alpaka mandelbrot test                              " << std::endl;
        std::cout << "################################################################################" << std::endl;
        std::cout << std::endl;

        // Logs the enabled accelerators.
        alpaka::test::acc::writeEnabledAccs<alpaka::dim::DimInt<2u>, std::uint32_t>(std::cout);

        std::cout << std::endl;

        MandelbrotKernelTester mandelbrotTester;

        // For different sizes.
        for(std::uint32_t imageSize(1u<<3u);
#ifdef ALPAKA_CI
            imageSize <= 1u<<5u;
#else
            imageSize <= 1u<<13u;
#endif
            imageSize *= 2u)
        {
            std::cout << std::endl;

            // Execute the kernel on all enabled accelerators.
            alpaka::meta::forEachType<
                alpaka::test::acc::EnabledAccs<alpaka::dim::DimInt<2u>, std::uint32_t>>(
                    mandelbrotTester,
                    imageSize,
                    imageSize,
                    -2.0f,
                    +1.0f,
                    -1.2f,
                    +1.2f,
                    300u);
        }
        return EXIT_SUCCESS;
    }
    catch(std::exception const & e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch(...)
    {
        std::cerr << "Unknown Exception" << std::endl;
        return EXIT_FAILURE;
    }
}
