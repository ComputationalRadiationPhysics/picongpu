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

#include <boost/core/ignore_unused.hpp>

#include <iostream>
#include <typeinfo>
#include <vector>
#include <functional>

//#############################################################################
//! A matrix multiplication kernel.
//! Computes C += A*B. LxM * MxN -> LxN
//! This is an adaption of the algorithm from the CUDA developers guide.
class MatMulKernel
{
public:
    //-----------------------------------------------------------------------------
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam TElem The matrix element type.
    //! \param acc The accelerator to be executed on.
    //! \param m The height of the A matrix.
    //! \param n The width of the A and height of the B matrix.
    //! \param k The width of the B matrix.
    //! \param A The pointer to the matrix A data.
    //! \param lda The pitch of the A matrix in elements.
    //! \param B The pointer to the matrix B data.
    //! \param ldb The pitch of the B matrix in elements.
    //! \param C The pointer to the matrix C data.
    //! \param ldc The pitch of the C matrix in elements.
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc,
        typename TElem,
        typename TIndex>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        TIndex const & m,
        TIndex const & n,
        TIndex const & k,
        TElem const & alpha,
        TElem const * const A,
        TIndex const & lda,
        TElem const * const B,
        TIndex const & ldb,
        TElem const & beta,
        TElem * const C,
        TIndex const & ldc) const
    -> void
    {
        static_assert(alpaka::dim::Dim<TAcc>::value == 2u,
            "The accelerator used for the GemmAlpakaKernel has to be 2 dimensional!");

        // Column and row of C to calculate.
        auto const gridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc));
        auto const & gridThreadIdxX(gridThreadIdx[1u]);
        auto const & gridThreadIdxY(gridThreadIdx[0u]);

        // Column and row inside the block of C to calculate.
        auto const blockThreadIdx(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc));
        auto const & blockThreadIdxX(blockThreadIdx[1u]);
        auto const & blockThreadIdxY(blockThreadIdx[0u]);

        // The block threads extent.
        auto const blockThreadExtent(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc));
        auto const & blockThreadExtentX(blockThreadExtent[1u]);
        auto const & blockThreadExtentY(blockThreadExtent[0u]);
        //assert(blockThreadExtentX == blockThreadExtentY);
        auto const & blockThreadExtentVal(blockThreadExtentX);

        // Shared memory used to store the current blocks of A and B.
        auto * const pBlockSharedA(alpaka::block::shared::dyn::getMem<TElem>(acc));
        auto * const pBlockSharedB(pBlockSharedA + blockThreadExtentX*blockThreadExtentY);

        auto const sharedBlockIdx1d(blockThreadIdxY*blockThreadExtentX + blockThreadIdxX);

        // If the element corresponding to the current thread is outside of the respective matrix.
        bool const insideA(gridThreadIdxY < m);
        bool const insideB(gridThreadIdxX < n);
        bool const insideC(insideA && insideB);

        TElem dotProduct(0);

        // Loop over all blocks of A and B that are required to compute the C block.
        auto const blockMulCount(static_cast<TIndex>(std::ceil(static_cast<float>(k)/static_cast<float>(blockThreadExtentVal))));
        for(TIndex k2(0u); k2 < blockMulCount; ++k2)
        {
            // Copy the current blocks of A and B into shared memory in parallel.
            // If the element of the current thread is outside of the matrix, zero is written into the shared memory.
            // This is possible because zero is a result neutral extension of the matrices regarding the dot product.
            auto const AIdxX(k2*blockThreadExtentX + blockThreadIdxX);
            auto const AIdx1d(gridThreadIdxY*lda + AIdxX);
            pBlockSharedA[sharedBlockIdx1d] = (
                ((!insideA) || (AIdxX>=k))
                ? static_cast<TElem>(0)
                : A[AIdx1d]);

            auto const BIdxY(k2*blockThreadExtentY + blockThreadIdxY);
            auto const BIdx1d(BIdxY*ldb + gridThreadIdxX);
            pBlockSharedB[sharedBlockIdx1d] = (
                ((!insideB) || (BIdxY>=k))
                ? static_cast<TElem>(0)
                : B[BIdx1d]);

            // Synchronize to make sure the complete blocks are loaded before starting the computation.
            alpaka::block::sync::syncBlockThreads(acc);

            // Not really necessary because we wrote zeros into those cells.
            //if(insideC)
            //{
                // Compute the dot products within shared memory.
                for(TIndex k3(0); k3 < blockThreadExtentVal; ++k3)
                {
                    dotProduct += pBlockSharedA[blockThreadIdxY*blockThreadExtentX + k3]
                        * pBlockSharedB[k3*blockThreadExtentY + blockThreadIdxX];
                }
            //}

            // Synchronize to make sure that the preceding computation is done before loading the next blocks of A and B.
            alpaka::block::sync::syncBlockThreads(acc);
        }

        // If the element is outside of the matrix it was only a helper thread that did not calculate any meaningful results.
        if(insideC)
        {
            auto const CIdx1d(gridThreadIdxY*ldc + gridThreadIdxX);
            C[CIdx1d] = alpha * dotProduct + beta * C[CIdx1d];
        }
    }
};

namespace alpaka
{
    namespace kernel
    {
        namespace traits
        {
            //#############################################################################
            //! The trait for getting the size of the block shared dynamic memory for a kernel.
            template<
                typename TAcc>
            struct BlockSharedMemDynSizeBytes<
                MatMulKernel,
                TAcc>
            {
                //-----------------------------------------------------------------------------
                //! \return The size of the shared memory allocated for a block.
                template<
                    typename TVec,
                    typename TIndex,
                    typename TElem>
                ALPAKA_FN_HOST static auto getBlockSharedMemDynSizeBytes(
                    MatMulKernel const & matMulKernel,
                    TVec const & blockThreadExtent,
                    TVec const & threadElemExtent,
                    TIndex const & m,
                    TIndex const & n,
                    TIndex const & k,
                    TElem const & alpha,
                    TElem const * const A,
                    TIndex const & lda,
                    TElem const * const B,
                    TIndex const & ldb,
                    TElem const & beta,
                    TElem * const C,
                    TIndex const & ldc)
                -> TIndex
                {
                    boost::ignore_unused(matMulKernel);
                    boost::ignore_unused(m);
                    boost::ignore_unused(n);
                    boost::ignore_unused(k);
                    boost::ignore_unused(alpha);
                    boost::ignore_unused(A);
                    boost::ignore_unused(lda);
                    boost::ignore_unused(B);
                    boost::ignore_unused(ldb);
                    boost::ignore_unused(beta);
                    boost::ignore_unused(C);
                    boost::ignore_unused(ldc);

                    // Reserve the buffer for the two blocks of A and B.
                    return 2u * blockThreadExtent.prod() * threadElemExtent.prod() * sizeof(TElem);
                }
            };
        }
    }
}

//#############################################################################
//! Profiles the example kernel and checks the result.
struct MatMulTester
{
    template<
        typename TAcc,
        typename TSize>
    auto operator()(
        TSize const & m,
        TSize const & n,
        TSize const & k)
    -> void
    {
        std::cout << std::endl;
        std::cout << "################################################################################" << std::endl;

        using Val = std::uint32_t;
        using Vec2 = alpaka::vec::Vec<alpaka::dim::DimInt<2u>, TSize>;
        using DevAcc = alpaka::dev::Dev<TAcc>;
        using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
        using StreamAcc = alpaka::test::stream::DefaultStream<alpaka::dev::Dev<TAcc>>;
        using PltfHost = alpaka::pltf::PltfCpu;
        using DevHost = alpaka::dev::Dev<PltfHost>;
        using StreamHost = alpaka::stream::StreamCpuAsync;

        // Create the kernel function object.
        MatMulKernel kernel;

        // Get the host device.
        DevHost const devHost(
            alpaka::pltf::getDevByIdx<PltfHost>(0u));

        // Get a stream on the host device.
        StreamHost streamHost(
            devHost);

        // Select a device to execute on.
        DevAcc const devAcc(
            alpaka::pltf::getDevByIdx<PltfAcc>(0u));

        // Get a stream on the accelerator device.
        StreamAcc streamAcc(
            devAcc);

        // Specify the input matrix extents.
        Vec2 const extentA(
            static_cast<TSize>(m),
            static_cast<TSize>(k));

        Vec2 const extentB(
            static_cast<TSize>(k),
            static_cast<TSize>(n));

        // Result matrix is MxN. We create one worker per result matrix cell.
        Vec2 const extentC(
            static_cast<TSize>(m),
            static_cast<TSize>(n));

        // Let alpaka calculate good block and grid sizes given our full problem extent.
        alpaka::workdiv::WorkDivMembers<alpaka::dim::DimInt<2u>, TSize> const workDiv(
            alpaka::workdiv::getValidWorkDiv<TAcc>(
                devAcc,
                extentC,
                alpaka::vec::Vec<alpaka::dim::DimInt<2u>, TSize>::ones(),
                false,
                alpaka::workdiv::GridBlockExtentSubDivRestrictions::EqualExtent));

        std::cout
            << "profileAcceleratedMatMulKernel("
            << "m:" << m
            << ", n:" << n
            << ", k:" << k
            << ", accelerator: " << alpaka::acc::getAccName<TAcc>()
            << ", kernel: " << typeid(kernel).name()
            << ", workDiv: " << workDiv
            << ")" << std::endl;

        // Allocate the A and B matrices as st::vectors because this allows them to be filled with uint32_t(1).
        // alpaka::mem::view::set only supports setting all bytes leading to a value of 16843009 in all elements.
        std::vector<Val> bufAHost1d(m * k, static_cast<Val>(1));
        std::vector<Val> bufBHost1d(k * n, static_cast<Val>(1));
        // Wrap the std::vectors into a memory buffer object.
        // For 1D data this would not be required because alpaka::mem::view::copy is specialized for std::vector and std::array.
        // For multi dimensional data you could directly create them using alpaka::mem::buf::alloc<Type>(devHost, extent), which is not used here.
        // Instead we use ViewPlainPtr to wrap the data.
        using BufWrapper = alpaka::mem::view::ViewPlainPtr<
            DevHost,
            Val,
            alpaka::dim::DimInt<2u>,
            TSize>;
        BufWrapper bufAHost(bufAHost1d.data(), devHost, extentA);
        BufWrapper bufBHost(bufBHost1d.data(), devHost, extentB);

        // Allocate C and set it to zero.
        auto bufCHost(alpaka::mem::buf::alloc<Val, TSize>(devHost, extentC));
        alpaka::mem::view::set(streamHost, bufCHost, 0u, extentC);

        // Allocate the buffers on the accelerator.
        auto bufAAcc(alpaka::mem::buf::alloc<Val, TSize>(devAcc, extentA));
        auto bufBAcc(alpaka::mem::buf::alloc<Val, TSize>(devAcc, extentB));
        auto bufCAcc(alpaka::mem::buf::alloc<Val, TSize>(devAcc, extentC));

        // Copy Host -> Acc.
        alpaka::mem::view::copy(streamAcc, bufAAcc, bufAHost, extentA);
        alpaka::mem::view::copy(streamAcc, bufBAcc, bufBHost, extentB);
        alpaka::wait::wait(streamHost);
        alpaka::mem::view::copy(streamAcc, bufCAcc, bufCHost, extentC);

        // Create the executor task.
        auto const exec(alpaka::exec::create<TAcc>(
            workDiv,
            kernel,
            m,
            n,
            k,
            static_cast<Val>(1),
            alpaka::mem::view::getPtrNative(bufAAcc),
            static_cast<TSize>(alpaka::mem::view::getPitchBytes<1u>(bufAAcc) / sizeof(Val)),
            alpaka::mem::view::getPtrNative(bufBAcc),
            static_cast<TSize>(alpaka::mem::view::getPitchBytes<1u>(bufBAcc) / sizeof(Val)),
            static_cast<Val>(1),
            alpaka::mem::view::getPtrNative(bufCAcc),
            static_cast<TSize>(alpaka::mem::view::getPitchBytes<1u>(bufCAcc) / sizeof(Val))));

        // Profile the kernel execution.
        std::cout << "Execution time: "
            << alpaka::test::integ::measureKernelRunTimeMs(
                streamAcc,
                exec)
            << " ms"
            << std::endl;

        // Copy back the result.
        alpaka::mem::view::copy(streamAcc, bufCHost, bufCAcc, extentC);

        // Wait for the stream to finish the memory operation.
        alpaka::wait::wait(streamAcc);

        // Assert that the results are correct.
        // When multiplying square matrices filled with ones, the result of each cell is the size of the matrix.
        auto const correctResult(static_cast<Val>(k));

        bool resultCorrect(true);
        auto const pHostData(alpaka::mem::view::getPtrNative(bufCHost));
        for(TSize i(0u);
            i < m * n;
            ++i)
        {
            auto const & val(pHostData[i]);
            if(val != correctResult)
            {
                std::cout << "C[" << i << "] == " << val << " != " << correctResult << std::endl;
                resultCorrect = false;
            }
        }

        if(resultCorrect)
        {
            std::cout << "Execution results correct!" << std::endl;
        }

        std::cout << "################################################################################" << std::endl;

        allResultsCorrect = allResultsCorrect && resultCorrect;
    }

public:
    bool allResultsCorrect = true;
};

auto main()
-> int
{
    try
    {
        {
            std::cout << std::endl;
            std::cout << "################################################################################" << std::endl;
            std::cout << "                              alpaka matMul test                                " << std::endl;
            std::cout << "################################################################################" << std::endl;
            std::cout << std::endl;

            // Logs the enabled accelerators.
            alpaka::test::acc::writeEnabledAccs<alpaka::dim::DimInt<2u>, std::uint32_t>(std::cout);

            std::cout << std::endl;

            MatMulTester matMulTester;

            // For different matrix sizes.
#ifdef ALPAKA_CI
            for(std::uint32_t m(1u); m <= 64u; m *= 8u)
            {
                for(std::uint32_t n(1u); n <= 79u; n *= 79u)
                {
                    for(std::uint32_t k(1u); k <= 23u; k *= 23u)
                    {
#else
            for(std::uint32_t m(1u); m <= 1024u; m *= 4u)
            {
                for(std::uint32_t n(1u); n <= 1024u; n *= 4u)
                {
                    for(std::uint32_t k(1u); k <= 1024u; k *= 4u)
                    {
#endif
                        std::cout << std::endl;

                        // Execute the kernel on all enabled accelerators.
                        alpaka::meta::forEachType<
                            alpaka::test::acc::EnabledAccs<alpaka::dim::DimInt<2u>, std::uint32_t>>(
                                matMulTester,
                                m, n, k);
                    }
                }
            }
            return matMulTester.allResultsCorrect ? EXIT_SUCCESS : EXIT_FAILURE;
        }
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
