/* Copyright 2023 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Jan Stephan, Bernhard Manfred Gruber,
 *                Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/test/MeasureKernelRunTime.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <functional>
#include <iostream>
#include <typeinfo>
#include <vector>

//! A matrix multiplication kernel.
//! Computes C + alpha*A*B + beta*C. LxM * MxN -> LxN
//! This is an adaption of the algorithm from the CUDA developers guide.
class MatMulKernel
{
public:
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
    template<typename TAcc, typename TElem, typename TIndex>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        TIndex const& m,
        TIndex const& n,
        TIndex const& k,
        TElem const& alpha,
        TElem const* const A,
        TIndex const& lda,
        TElem const* const B,
        TIndex const& ldb,
        TElem const& beta,
        TElem* const C,
        TIndex const& ldc) const -> void
    {
        static_assert(
            alpaka::Dim<TAcc>::value == 2u,
            "The accelerator used for the GemmAlpakaKernel has to be 2 dimensional!");

        // Column and row of C to calculate.
        auto const [gridThreadIdxY, gridThreadIdxX] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);

        // Column and row inside the block of C to calculate.
        auto const [blockThreadIdxY, blockThreadIdxX] = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);

        // The block threads extent.
        auto const [blockThreadExtentY, blockThreadExtentX] = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        // ALPAKA_ASSERT(blockThreadExtentX == blockThreadExtentY);
        auto const& blockThreadExtentVal = blockThreadExtentX;

        // Shared memory used to store the current blocks of A and B.
        auto* const pBlockSharedA = alpaka::getDynSharedMem<TElem>(acc);
        auto* const pBlockSharedB = pBlockSharedA + blockThreadExtentX * blockThreadExtentY;

        auto const sharedBlockIdx1d = blockThreadIdxY * blockThreadExtentX + blockThreadIdxX;

        // If the element corresponding to the current thread is outside of the respective matrix.
        bool const insideA(gridThreadIdxY < m);
        bool const insideB(gridThreadIdxX < n);
        bool const insideC(insideA && insideB);

        TElem dotProduct(0);

        // Loop over all blocks of A and B that are required to compute the C block.
        auto const blockMulCount = (k + blockThreadExtentVal - 1) / blockThreadExtentVal;
        for(TIndex k2(0u); k2 < blockMulCount; ++k2)
        {
            // Copy the current blocks of A and B into shared memory in parallel.
            // If the element of the current thread is outside the matrix, zero is written into the shared memory.
            // This is possible because zero is a result neutral extension of the matrices regarding the dot product.
            auto const AIdxX = k2 * blockThreadExtentX + blockThreadIdxX;
            auto const AIdx1d = gridThreadIdxY * lda + AIdxX;
            pBlockSharedA[sharedBlockIdx1d] = (!insideA || AIdxX >= k) ? TElem{0} : A[AIdx1d];

            auto const BIdxY = k2 * blockThreadExtentY + blockThreadIdxY;
            auto const BIdx1d = BIdxY * ldb + gridThreadIdxX;
            pBlockSharedB[sharedBlockIdx1d] = (!insideB || BIdxY >= k) ? TElem{0} : B[BIdx1d];

            // Synchronize to make sure the complete blocks are loaded before starting the computation.
            alpaka::syncBlockThreads(acc);

            // Compute the dot products within shared memory.
            for(TIndex k3(0); k3 < blockThreadExtentVal; ++k3)
            {
                dotProduct += pBlockSharedA[blockThreadIdxY * blockThreadExtentX + k3]
                              * pBlockSharedB[k3 * blockThreadExtentY + blockThreadIdxX];
            }

            // Synchronize to make sure that the preceding computation is done before loading the next blocks of A and
            // B.
            alpaka::syncBlockThreads(acc);
        }

        // If the element is outside of the matrix it was only a helper thread that did not calculate any meaningful
        // results.
        if(insideC)
        {
            auto const CIdx1d = gridThreadIdxY * ldc + gridThreadIdxX;
            C[CIdx1d] = alpha * dotProduct + beta * C[CIdx1d];
        }
    }
};

namespace alpaka::trait
{
    //! The trait for getting the size of the block shared dynamic memory for a kernel.
    template<typename TAcc>
    struct BlockSharedMemDynSizeBytes<MatMulKernel, TAcc>
    {
        //! \return The size of the shared memory allocated for a block.
        template<typename TVec, typename TIndex, typename TElem>
        ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
            MatMulKernel const& /* matMulKernel */,
            TVec const& blockThreadExtent,
            TVec const& threadElemExtent,
            TIndex const& /* m */,
            TIndex const& /* n */,
            TIndex const& /* k */,
            TElem const& /* alpha */,
            TElem const* const /* A */,
            TIndex const& /* lda */,
            TElem const* const /* B */,
            TIndex const& /* ldb */,
            TElem const& /* beta */,
            TElem* const /* C */,
            TIndex const& /* ldc */)
        {
            // Reserve the buffer for the two blocks of A and B.
            return static_cast<std::size_t>(2u * blockThreadExtent.prod() * threadElemExtent.prod()) * sizeof(TElem);
        }
    };
} // namespace alpaka::trait

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<2u>, std::uint32_t>;

TEMPLATE_LIST_TEST_CASE("matMul", "[matMul]", TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    Idx const m(64u);
    Idx const n(79u);
    Idx const k(23u);

    using Val = std::uint32_t;
    using Vec2 = alpaka::Vec<Dim, Idx>;
    using QueueAcc = alpaka::test::DefaultQueue<alpaka::Dev<Acc>>;
    using QueueHost = alpaka::QueueCpuNonBlocking;

    // Create the kernel function object.
    MatMulKernel kernel;

    // Get the host device.
    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    // Get a queue on the host device.
    QueueHost queueHost(devHost);

    // Select a device to execute on.
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

    // Get a queue on the accelerator device.
    QueueAcc queueAcc(devAcc);

    // Specify the input matrix extents.
    Vec2 const extentA(static_cast<Idx>(m), static_cast<Idx>(k));

    Vec2 const extentB(static_cast<Idx>(k), static_cast<Idx>(n));

    // Result matrix is MxN. We create one worker per result matrix cell.
    Vec2 const extentC(static_cast<Idx>(m), static_cast<Idx>(n));


    // Allocate the A and B matrices as std::vectors because this allows them to be filled with uint32_t(1).
    // alpaka::set only supports setting all bytes leading to a value of 16843009 in all elements.
    std::vector<Val> bufAHost1d(m * k, static_cast<Val>(1));
    std::vector<Val> bufBHost1d(k * n, static_cast<Val>(1));
    // Wrap the std::vectors into a memory buffer object.
    // For 1D data this would not be required because alpaka::copy is specialized for std::vector and std::array.
    // For multi dimensional data you could directly create them using alpaka::malloc<Type>(devHost, extent), which is
    // not used here. Instead we create a View to wrap the data.
    auto bufAHost = alpaka::createView(devHost, bufAHost1d.data(), extentA);
    auto bufBHost = alpaka::createView(devHost, bufBHost1d.data(), extentB);

    // Allocate C and set it to zero.
    auto bufCHost = alpaka::allocMappedBufIfSupported<Val, Idx>(devHost, platformAcc, extentC);
    alpaka::memset(queueHost, bufCHost, 0u);

    // Allocate the buffers on the accelerator.
    auto bufAAcc = alpaka::allocBuf<Val, Idx>(devAcc, extentA);
    auto bufBAcc = alpaka::allocBuf<Val, Idx>(devAcc, extentB);
    auto bufCAcc = alpaka::allocBuf<Val, Idx>(devAcc, extentC);

    // Copy inputs Host -> Acc.
    std::cout << "Input1 copy time: "
              << alpaka::test::integ::measureRunTimeMs(
                     [&]
                     {
                         alpaka::memcpy(queueAcc, bufAAcc, bufAHost);
                         alpaka::memcpy(queueAcc, bufBAcc, bufBHost);
                         alpaka::wait(queueAcc);
                     })
              << " ms" << std::endl;
    alpaka::wait(queueHost); // Make sure memset finished
    std::cout << "Input2 copy time: "
              << alpaka::test::integ::measureRunTimeMs([&] { alpaka::memcpy(queueAcc, bufCAcc, bufCHost); }) << " ms"
              << std::endl;

    auto const rowPitchA = alpaka::getPitchesInBytes(bufAAcc)[0];
    auto const rowPitchB = alpaka::getPitchesInBytes(bufBAcc)[0];
    auto const rowPitchC = alpaka::getPitchesInBytes(bufCAcc)[0];

    // We assume that the row pitches are divisible by the element size
    REQUIRE(rowPitchA % sizeof(Val) == 0);
    REQUIRE(rowPitchB % sizeof(Val) == 0);
    REQUIRE(rowPitchC % sizeof(Val) == 0);

    auto const lda = static_cast<Idx>(rowPitchA / sizeof(Val));
    auto const ldb = static_cast<Idx>(rowPitchB / sizeof(Val));
    auto const ldc = static_cast<Idx>(rowPitchC / sizeof(Val));

    std::cout << "pitchesA " << alpaka::getPitchesInBytes(bufAAcc) << " lda: " << lda << "\n";
    std::cout << "pitchesB " << alpaka::getPitchesInBytes(bufBAcc) << " ldb: " << ldb << "\n";
    std::cout << "pitchesC " << alpaka::getPitchesInBytes(bufCAcc) << " ldc: " << ldc << "\n";

    // Let alpaka calculate good block and grid sizes given our full problem extent
    alpaka::KernelCfg<Acc> const kernelCfg
        = {extentC, alpaka::Vec<Dim, Idx>::ones(), false, alpaka::GridBlockExtentSubDivRestrictions::EqualExtent};
    auto const workDiv = alpaka::getValidWorkDiv(
        kernelCfg,
        devAcc,
        kernel,
        m,
        n,
        k,
        static_cast<Val>(1),
        std::data(bufAAcc),
        lda,
        std::data(bufBAcc),
        ldb,
        static_cast<Val>(1),
        std::data(bufCAcc),
        ldc);


    std::cout << "MatMulKernel("
              << "m:" << m << ", n:" << n << ", k:" << k << ", accelerator: " << alpaka::getAccName<Acc>()
              << ", kernel: " << alpaka::core::demangled<decltype(kernel)> << ", workDiv: " << workDiv << ")"
              << std::endl;


    // Create the kernel execution task.
    auto const taskKernel = alpaka::createTaskKernel<Acc>(
        workDiv,
        kernel,
        m,
        n,
        k,
        static_cast<Val>(1),
        std::data(bufAAcc),
        lda,
        std::data(bufBAcc),
        ldb,
        static_cast<Val>(1),
        std::data(bufCAcc),
        ldc);

    // Profile the kernel execution.
    std::cout << "Execution time:   " << alpaka::test::integ::measureTaskRunTimeMs(queueAcc, taskKernel) << " ms"
              << std::endl;

    // Copy back the result.
    std::cout << "Output copy time: "
              << alpaka::test::integ::measureRunTimeMs(
                     [&]
                     {
                         alpaka::memcpy(queueAcc, bufCHost, bufCAcc);
                         alpaka::wait(queueAcc);
                     })
              << " ms" << std::endl;

    // Assert that the results are correct.
    // When multiplying square matrices filled with ones, the result of each cell is the size of the matrix.
    auto const correctResult = static_cast<Val>(k);

    bool resultCorrect = true;
    auto const pHostData = std::data(bufCHost);
    for(Idx i(0u); i < m * n; ++i)
    {
        auto const& val(pHostData[i]);
        if(val != correctResult)
        {
            std::cerr << "C[" << i << "] == " << val << " != " << correctResult << std::endl;
            resultCorrect = false;
        }
    }

    REQUIRE(resultCorrect);
}
