#include "babelStreamCommon.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

/**
 * Babelstream benchmarking example. Babelstream has 5 kernels. Add, Multiply, Copy, Triad and Dot. NStream is
 * optional. Init kernel is run before 5 standard kernel sequence. Babelstream is a memory-bound benchmark since the
 * main operation in the kernels has high Code Balance (bytes/FLOP) value. For example c[i] = a[i] + b[i]; has 2 reads
 * 1 writes and has one FLOP operation. For double precision each read-write is 8 bytes. Hence Code Balance (3*8 / 1) =
 * 24 bytes/FLOP.
 *
 * Some implementations and the documents are accessible through https://github.com/UoB-HPC
 *
 * Can be run with custom arguments as well as catch2 arguments
 * Run with Custom arguments and for kernels: init, copy, mul, add, triad (and dot kernel if a multi-thread acc
 * available):
 * ./babelstream --array-size=33554432 --number-runs=100
 * Run with Custom arguments and select from 3 kernel groups: all, triad, nstream
 * ./babelstream --array-size=33554432 --number-runs=100 --run-kernels=triad (only triad kernel)
 * ./babelstream --array-size=33554432 --number-runs=100 --run-kernels=nstream (only nstream kernel)
 * ./babelstream --array-size=33554432 --number-runs=100 --run-kernels=all (default case. Add, Multiply, Copy, Triad
 * and Dot) Run with default array size and num runs:
 * ./babelstream
 * Run with Catch2 arguments and default array size and num runs:
 * ./babelstream --success
 * ./babelstream -r xml
 * Run with Custom and catch2 arguments together:
 * ./babelstream  --success --array-size=1280000 --number-runs=10
 * Help to list custom and catch2 arguments
 * ./babelstream -?
 * ./babelstream --help
 *  According to tests, 2^25 or larger data size values are needed for proper benchmarking:
 *  ./babelstream --array-size=33554432 --number-runs=100
 */

// Main function that integrates Catch2 and custom argument handling
int main(int argc, char* argv[])
{
    // Handle custom arguments
    handleCustomArguments(argc, argv);

    // Initialize Catch2 and pass the command-line arguments to it
    int result = Catch::Session().run(argc, argv);

    // Return the result of the tests
    return result;
}

//! Initialization kernel
struct InitKernel
{
    //! The kernel entry point
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam T The data type
    //! \param acc The accelerator to be executed on.
    //! \param a Pointer for vector a
    //! \param initialA the value to set all items in the vector a
    //! \param initialB the value to set all items in the vector b
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T* a, T* b, T* c, T initialA, T initialB, T initialC) const
    {
        auto const [i] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        a[i] = initialA;
        b[i] = initialB;
        c[i] = initialC;
    }
};

//! Vector copying kernel
struct CopyKernel
{
    //! The kernel entry point
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam T The data type
    //! \param acc The accelerator to be executed on.
    //! \param a Pointer for vector a
    //! \param c Pointer for vector c
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* a, T* c) const
    {
        auto const [index] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        c[index] = a[index];
    }
};

//! Kernel multiplies the vector with a scalar, scaling or multiplication kernel
struct MultKernel
{
    //! The kernel entry point
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam T The data type
    //! \param acc The accelerator to be executed on.
    //! \param c Pointer for vector c
    //! \param b Pointer for result vector b
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T* b, T* const c) const
    {
        const T scalar = static_cast<T>(scalarVal);
        auto const [i] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        b[i] = scalar * c[i];
    }
};

//! Vector summation kernel
struct AddKernel
{
    //! The kernel entry point
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam T The data type
    //! \param acc The accelerator to be executed on.
    //! \param a Pointer for vector a
    //! \param b Pointer for vector b
    //! \param c Pointer for result vector c
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* a, T const* b, T* c) const
    {
        auto const [i] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        c[i] = a[i] + b[i];
    }
};

//! Kernel to find the linear combination of 2 vectors by initially scaling one of them
struct TriadKernel
{
    //! The kernel entry point
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam T The data type
    //! \param acc The accelerator to be executed on.
    //! \param a Pointer for vector a
    //! \param b Pointer for vector b
    //! \param c Pointer for result vector c
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T* a, T const* b, T const* c) const
    {
        const T scalar = static_cast<T>(scalarVal);
        auto const [i] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        a[i] = b[i] + scalar * c[i];
    }
};

//! Optional kernel, not one of the 5 standard Babelstream kernels
struct NstreamKernel
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T* a, T const* b, T const* c) const
    {
        const T scalar = static_cast<T>(scalarVal);
        auto const [i] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        a[i] += b[i] + scalar * c[i];
    }
};

//! Dot product of two vectors. The result is not a scalar but a vector of block-level dot products. For the
//! BabelStream implementation and documentation: https://github.com/UoB-HPC
struct DotKernel
{
    //! The kernel entry point
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam T The data type
    //! \param acc The accelerator to be executed on.
    //! \param a Pointer for vector a
    //! \param b Pointer for vector b
    //! \param sum Pointer for result vector consisting sums of blocks
    //! \param arraySize the size of the array
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* a, T const* b, T* sum, alpaka::Idx<TAcc> arraySize) const
    {
        using Idx = alpaka::Idx<TAcc>;
        auto& tbSum = alpaka::declareSharedVar<T[blockThreadExtentMain], __COUNTER__>(acc);

        auto i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const local_i = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];
        auto const totalThreads = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];

        T threadSum = 0;
        for(; i < arraySize; i += totalThreads)
            threadSum += a[i] * b[i];
        tbSum[local_i] = threadSum;

        auto const blockSize = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0];
        for(Idx offset = blockSize / 2; offset > 0; offset /= 2)
        {
            alpaka::syncBlockThreads(acc);
            if(local_i < offset)
            {
// Suppress warnings
#if defined(__GNUC__) && !defined(__clang__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#elif defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wuninitialized"
#endif
                // read from shared memory and sum
                tbSum[local_i] += tbSum[local_i + offset];
// Remove suppression of warnings
#if defined(__GNUC__) && !defined(__clang__)
#    pragma GCC diagnostic pop
#elif defined(__clang__)
#    pragma clang diagnostic pop
#endif
            }
        }

        auto const gridBlockIndex = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];
        if(local_i == 0)
            sum[gridBlockIndex] = tbSum[local_i];
    }
};

//! \brief The Function for testing babelstream kernels for given Acc type and data type.
//! \tparam TAcc the accelerator type
//! \tparam DataType The data type to differentiate single or double data type based tests.
template<typename TAcc, typename DataType>
void testKernels()
{
    if(kernelsToBeExecuted == KernelsToRun::All)
    {
        std::cout << "Kernels: Init, Copy, Mul, Add, Triad, Dot Kernels" << std::endl;
    }
    using Acc = TAcc;
    // Set the number of dimensions as an integral constant. Set to 1 for 1D.
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    // A MetaData class instance to keep the benchmark info and results to print later. Does not include intermediate
    // runtime data.
    BenchmarkMetaData metaData;

    // Convert data-type to string to display
    std::string dataTypeStr;
    if(std::is_same<DataType, float>::value)
    {
        dataTypeStr = "single";
    }
    else if(std::is_same<DataType, double>::value)
    {
        dataTypeStr = "double";
    }

    using QueueAcc = alpaka::Queue<Acc, alpaka::Blocking>;

    // Select a device
    auto const platform = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platform, 0);

    // Create a queue on the device
    QueueAcc queue(devAcc);

    // Get the host device for allocating memory on the host.
    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    // Create vectors
    Idx arraySize = static_cast<Idx>(arraySizeMain);

    // Acc buffers
    auto bufAccInputA = alpaka::allocBuf<DataType, Idx>(devAcc, arraySize);
    auto bufAccInputB = alpaka::allocBuf<DataType, Idx>(devAcc, arraySize);
    auto bufAccOutputC = alpaka::allocBuf<DataType, Idx>(devAcc, arraySize);

    // Host buffer as the result
    auto bufHostOutputA = alpaka::allocBuf<DataType, Idx>(devHost, arraySize);
    auto bufHostOutputB = alpaka::allocBuf<DataType, Idx>(devHost, arraySize);
    auto bufHostOutputC = alpaka::allocBuf<DataType, Idx>(devHost, arraySize);

    // Grid size and elems per thread will be used to get the work division
    using Vec = alpaka::Vec<Dim, Idx>;
    auto const elementsPerThread = Vec::all(static_cast<Idx>(1));
    auto const elementsPerGrid = Vec::all(arraySize);

    // Create pointer variables for buffer access
    auto bufAccInputAPtr = std::data(bufAccInputA);
    auto bufAccInputBPtr = std::data(bufAccInputB);
    auto bufAccOutputCPtr = std::data(bufAccOutputC);

    // Bind gridsize and elements per thread together
    alpaka::KernelCfg<Acc> const kernelCfg = {elementsPerGrid, elementsPerThread};
    // Let alpaka calculate good work division (namely the block and grid sizes) given our full problem extent
    auto const workDivInit = alpaka::getValidWorkDiv(
        kernelCfg,
        devAcc,
        InitKernel(),
        bufAccInputAPtr,
        bufAccInputBPtr,
        bufAccOutputCPtr,
        static_cast<DataType>(initA),
        static_cast<DataType>(initB),
        static_cast<DataType>(initC));
    auto const workDivCopy
        = alpaka::getValidWorkDiv(kernelCfg, devAcc, CopyKernel(), bufAccInputAPtr, bufAccInputBPtr);
    auto const workDivMult
        = alpaka::getValidWorkDiv(kernelCfg, devAcc, MultKernel(), bufAccInputAPtr, bufAccInputBPtr);
    auto const workDivAdd
        = alpaka::getValidWorkDiv(kernelCfg, devAcc, AddKernel(), bufAccInputAPtr, bufAccInputBPtr, bufAccOutputCPtr);

    auto const workDivTriad = alpaka::getValidWorkDiv(
        kernelCfg,
        devAcc,
        TriadKernel(),
        bufAccInputAPtr,
        bufAccInputBPtr,
        bufAccOutputCPtr);

    auto const workDivNStream = alpaka::getValidWorkDiv(
        kernelCfg,
        devAcc,
        NstreamKernel(),
        bufAccInputAPtr,
        bufAccInputBPtr,
        bufAccOutputCPtr);


    // Lambda to create and return work division for dot kernel
    auto getWorkDivForDotKernel = [&]<typename AccType>() -> alpaka::WorkDivMembers<Dim, Idx>
    {
        // Use babelstream benchmark standard work division for multi-threaded backends
        // For GPUs the benchmark expects a fixed workdiv: {256,1024,1}
        // For multi-threaded CPUs, which are not part of the benchmark, the code below will limit the blocksize with
        // the limit of the backend.
        if constexpr(alpaka::isMultiThreadAcc<AccType>)
        {
            auto const kernelFunctionAttributes = alpaka::getFunctionAttributes<Acc>(
                devAcc,
                DotKernel(),
                bufAccInputAPtr,
                bufAccInputBPtr,
                bufAccOutputCPtr, // this is used here a kind of dummy
                static_cast<alpaka::Idx<AccType>>(arraySize));

            // Get the maxThreadPerBlock
            auto const maxThreadsPerBlock = kernelFunctionAttributes.maxThreadsPerBlock;
            // Threads per block is 1024 for benchmark, if the system does not allow use the max value
            auto threadsPerBlock = std::min(maxThreadsPerBlock, blockThreadExtentMain);

            // Reduce operation at dot-kernel needs even block size
            threadsPerBlock = (threadsPerBlock + 1) / 2 * 2;

            // Dot kernel is only used for benchmarking of GPU backends; and Work division is fixed for benchmark:
            // 256,1024,1. https://github.com/UoB-HPC/BabelStream/blob/main/src/cuda/CUDAStream.cu Hence blocksize
            // should be 1024 for GPU backends. But for multi-threaded CPUs; this code would also run and in that case
            // the blocksize would be less.
            auto workDiv = alpaka::WorkDivMembers{
                Vec::all(static_cast<alpaka::Idx<AccType>>(dotGridBlockExtent)),
                Vec::all(static_cast<alpaka::Idx<AccType>>(threadsPerBlock)),
                Vec::all(1)};

            return workDiv;
        }
        else
        {
            // Work division for single-threaded backends
            // Since block size is 1, the elements per grid is dotGridBlockExtent * blockThreadExtentMain
            alpaka::KernelCfg<AccType> const kernelCfgDot
                = {Vec::all(dotGridBlockExtent * blockThreadExtentMain), elementsPerThread};

            return alpaka::getValidWorkDiv(
                kernelCfgDot,
                devAcc,
                DotKernel(),
                bufAccInputAPtr,
                bufAccInputBPtr,
                bufAccOutputCPtr, // this is used here a kind of dummy
                static_cast<alpaka::Idx<AccType>>(arraySize));
        }
    };

    // Work Division for Dot Kernel
    auto const workDivDot = (getWorkDivForDotKernel.template operator()<Acc>());
    // To record runtime data generated while running the kernels
    RuntimeResults runtimeResults;

    // Lambda for measuring run-time
    auto measureKernelExec = [&](auto&& kernelFunc, [[maybe_unused]] auto&& kernelLabel)
    {
        double runtime = 0.0;
        auto start = std::chrono::high_resolution_clock::now();
        kernelFunc();
        alpaka::wait(queue);
        auto end = std::chrono::high_resolution_clock::now();
        // get duration in seconds
        std::chrono::duration<double> duration = end - start;
        runtime = duration.count();
        runtimeResults.kernelToRundataMap[kernelLabel]->timingsSuccessiveRuns.push_back(runtime);
    };


    // Initialize logger before running kernels
    // Runtime result initialisation to be filled by each kernel
    runtimeResults.addKernelTimingsVec("InitKernel");
    if(kernelsToBeExecuted == KernelsToRun::All)
    {
        runtimeResults.addKernelTimingsVec("CopyKernel");
        runtimeResults.addKernelTimingsVec("AddKernel");
        runtimeResults.addKernelTimingsVec("TriadKernel");
        runtimeResults.addKernelTimingsVec("MultKernel");
        runtimeResults.addKernelTimingsVec("DotKernel");
    }
    else if(kernelsToBeExecuted == KernelsToRun::NStream)
    {
        runtimeResults.addKernelTimingsVec("NStreamKernel");
    }
    else if(kernelsToBeExecuted == KernelsToRun::Triad)
    {
        runtimeResults.addKernelTimingsVec("TriadKernel");
    }


    // Init kernel
    measureKernelExec(
        [&]()
        {
            alpaka::exec<Acc>(
                queue,
                workDivInit,
                InitKernel(),
                bufAccInputAPtr,
                bufAccInputBPtr,
                bufAccOutputCPtr,
                static_cast<DataType>(initA),
                static_cast<DataType>(initB),
                static_cast<DataType>(initC));
        },
        "InitKernel");

    // Init kernel will be run for all cases therefore add it to metadata unconditionally
    metaData.setItem(BMInfoDataType::WorkDivInit, workDivInit);

    // Dot kernel result
    DataType resultDot = static_cast<DataType>(0.0f);

    // Main for loop to run the kernel-sequence
    for(auto i = 0; i < numberOfRuns; i++)
    {
        if(kernelsToBeExecuted == KernelsToRun::All)
        {
            // Test the copy-kernel. Copy A one by one to C.
            measureKernelExec(
                [&]() { alpaka::exec<Acc>(queue, workDivCopy, CopyKernel(), bufAccInputAPtr, bufAccOutputCPtr); },
                "CopyKernel");

            // Test the scaling-kernel. Calculate B=scalar*C. Where C = A.
            measureKernelExec(
                [&]() { alpaka::exec<Acc>(queue, workDivMult, MultKernel(), bufAccInputBPtr, bufAccOutputCPtr); },
                "MultKernel");

            // Test the addition-kernel. Calculate C=A+B. Where B=scalar*C or B=scalar*A.
            measureKernelExec(
                [&]() {
                    alpaka::exec<Acc>(
                        queue,
                        workDivAdd,
                        AddKernel(),
                        bufAccInputAPtr,
                        bufAccInputBPtr,
                        bufAccOutputCPtr);
                },
                "AddKernel");
        }
        // Triad kernel is run for 2 command line arguments
        if(kernelsToBeExecuted == KernelsToRun::All || kernelsToBeExecuted == KernelsToRun::Triad)
        {
            // Test the Triad-kernel. Calculate A=B+scalar*C. Where C is A+scalar*A.
            measureKernelExec(
                [&]() {
                    alpaka::exec<Acc>(
                        queue,
                        workDivTriad,
                        TriadKernel(),
                        bufAccInputAPtr,
                        bufAccInputBPtr,
                        bufAccOutputCPtr);
                },
                "TriadKernel");
        }
        if(kernelsToBeExecuted == KernelsToRun::All)
        {
            // Vector of sums of each block
            auto bufAccSumPerBlock = alpaka::allocBuf<DataType, Idx>(devAcc, workDivDot.m_gridBlockExtent[0]);
            auto bufHostSumPerBlock = alpaka::allocBuf<DataType, Idx>(devHost, workDivDot.m_gridBlockExtent[0]);
            // Test Dot kernel with specific blocksize which is larger than one


            measureKernelExec(
                [&]()
                {
                    alpaka::exec<Acc>(
                        queue,
                        workDivDot,
                        DotKernel(), // Dot kernel
                        bufAccInputAPtr,
                        bufAccInputBPtr,
                        alpaka::getPtrNative(bufAccSumPerBlock),
                        static_cast<alpaka::Idx<Acc>>(arraySize));
                    alpaka::memcpy(queue, bufHostSumPerBlock, bufAccSumPerBlock, workDivDot.m_gridBlockExtent[0]);
                    alpaka::wait(queue);

                    DataType const* sumPtr = std::data(bufHostSumPerBlock);
                    resultDot
                        = static_cast<DataType>(std::reduce(sumPtr, sumPtr + workDivDot.m_gridBlockExtent[0], 0.0));
                },
                "DotKernel");
            // Add workdiv to the list of workdivs to print later
            metaData.setItem(BMInfoDataType::WorkDivDot, workDivDot);
        }
        // NStream kernel is run only for one command line argument
        if(kernelsToBeExecuted == KernelsToRun::NStream)
        {
            // Test the NStream-kernel. Calculate A += B + scalar * C;
            measureKernelExec(
                [&]() {
                    alpaka::exec<Acc>(
                        queue,
                        workDivNStream,
                        NstreamKernel(),
                        bufAccInputAPtr,
                        bufAccInputBPtr,
                        bufAccOutputCPtr);
                },
                "NStreamKernel");
        }
        alpaka::wait(queue);
    } // End of MAIN LOOP which runs the kernels many times


    // Copy results back to the host, measure copy time
    {
        auto start = std::chrono::high_resolution_clock::now();
        // Copy arrays back to host since the execution of kernels except dot kernel finished
        alpaka::memcpy(queue, bufHostOutputC, bufAccOutputC, arraySize);
        alpaka::memcpy(queue, bufHostOutputB, bufAccInputB, arraySize);
        alpaka::memcpy(queue, bufHostOutputA, bufAccInputA, arraySize);
        alpaka::wait(queue);
        auto end = std::chrono::high_resolution_clock::now();
        // Get duration in seconds
        std::chrono::duration<double> duration = end - start;
        double copyRuntime = duration.count();
        metaData.setItem(BMInfoDataType::CopyTimeFromAccToHost, copyRuntime);
    }

    //
    // Result Verification and BW Calculation for 3 cases
    //

    // Generated expected values by doing the same chain of operations due to floating point error
    DataType expectedA = static_cast<DataType>(initA);
    DataType expectedB = static_cast<DataType>(initB);
    DataType expectedC = static_cast<DataType>(initC);

    // To calculate expected results by applying at host the same operation sequence
    calculateBabelstreamExpectedResults(expectedA, expectedB, expectedC);

    // Verify the resulting data, if kernels are init, copy, mul, add, triad and dot kernel
    if(kernelsToBeExecuted == KernelsToRun::All)
    {
        // Find sum of the errors as sum of the differences from expected values
        constexpr DataType initVal{static_cast<DataType>(0.0)};
        DataType sumErrC{initVal}, sumErrB{initVal}, sumErrA{initVal};

        // sum of the errors for each array
        for(Idx i = 0; i < arraySize; ++i)
        {
            sumErrC += std::fabs(bufHostOutputC[static_cast<Idx>(i)] - expectedC);
            sumErrB += std::fabs(bufHostOutputB[static_cast<Idx>(i)] - expectedB);
            sumErrA += std::fabs(bufHostOutputA[static_cast<Idx>(i)] - expectedA);
        }

        // Normalize and compare sum of the errors
        // Use a different equality check if floating point errors exceed precision of FuzzyEqual function
        REQUIRE(FuzzyEqual(sumErrC / static_cast<DataType>(arraySize), static_cast<DataType>(0.0)));
        REQUIRE(FuzzyEqual(sumErrB / static_cast<DataType>(arraySize), static_cast<DataType>(0.0)));
        REQUIRE(FuzzyEqual(sumErrA / static_cast<DataType>(arraySize), static_cast<DataType>(0.0)));
        alpaka::wait(queue);

        // Verify Dot kernel
        DataType const expectedSum = static_cast<DataType>(arraySize) * expectedA * expectedB;
        //  Dot product should be identical to arraySize*valA*valB
        //  Use a different equality check if floating point errors exceed precision of FuzzyEqual function
        REQUIRE(FuzzyEqual(static_cast<float>(std::fabs(resultDot - expectedSum) / expectedSum), 0.0f));

        // Set workdivs of benchmark metadata to be displayed at the end
        metaData.setItem(BMInfoDataType::WorkDivInit, workDivInit);
        metaData.setItem(BMInfoDataType::WorkDivCopy, workDivCopy);
        metaData.setItem(BMInfoDataType::WorkDivAdd, workDivAdd);
        metaData.setItem(BMInfoDataType::WorkDivMult, workDivMult);
        metaData.setItem(BMInfoDataType::WorkDivTriad, workDivTriad);
    }
    // Verify the Triad Kernel result if "--run-kernels=triad".
    else if(kernelsToBeExecuted == KernelsToRun::Triad)
    {
        // Verify triad by summing the error
        auto sumErrA = static_cast<DataType>(0.0);
        // sum of the errors for each array
        for(Idx i = 0; i < arraySize; ++i)
        {
            sumErrA += std::fabs(bufHostOutputA[static_cast<Idx>(i)] - expectedA);
        }

        REQUIRE(FuzzyEqual(sumErrA / static_cast<DataType>(arraySize) / expectedA, static_cast<DataType>(0.0)));
        metaData.setItem(BMInfoDataType::WorkDivTriad, workDivTriad);
    }
    // Verify the NStream Kernel result if "--run-kernels=nstream".
    else if(kernelsToBeExecuted == KernelsToRun::NStream)
    {
        auto sumErrA = static_cast<DataType>(0.0);
        // sum of the errors for each array
        for(Idx i = 0; i < arraySize; ++i)
        {
            sumErrA += std::fabs(bufHostOutputA[static_cast<Idx>(i)] - expectedA);
        }
        REQUIRE(FuzzyEqual(sumErrA / static_cast<DataType>(arraySize) / expectedA, static_cast<DataType>(0.0)));

        metaData.setItem(BMInfoDataType::WorkDivNStream, workDivNStream);
    }

    // Runtime results of the benchmark: Calculate throughput and bandwidth
    // Set throuput values depending on the kernels
    runtimeResults.initializeByteReadWrite<DataType>(arraySize);
    runtimeResults.calculateBandwidthsForKernels<DataType>();

    // Set metadata to display all benchmark related information.
    //
    // All information about benchmark and results are stored in a single map
    metaData.setItem(BMInfoDataType::TimeStamp, getCurrentTimestamp());
    metaData.setItem(BMInfoDataType::NumRuns, std::to_string(numberOfRuns));
    metaData.setItem(BMInfoDataType::DataSize, std::to_string(arraySizeMain));
    metaData.setItem(BMInfoDataType::DataType, dataTypeStr);
    // Device and accelerator
    metaData.setItem(BMInfoDataType::DeviceName, alpaka::getName(devAcc));
    metaData.setItem(BMInfoDataType::AcceleratorType, alpaka::getAccName<Acc>());
    // XML reporter of catch2 always converts to Nano Seconds
    metaData.setItem(BMInfoDataType::TimeUnit, "Nano Seconds");

    // get labels from the map
    std::vector<std::string> kernelLabels;
    std::transform(
        runtimeResults.kernelToRundataMap.begin(),
        runtimeResults.kernelToRundataMap.end(),
        std::back_inserter(kernelLabels),
        [](auto const& pair) { return pair.first; });
    // Join elements and create a comma separated string and set item
    metaData.setItem(BMInfoDataType::KernelNames, joinElements(kernelLabels, ", "));
    // Join elements and create a comma separated string and set item
    std::vector<double> values(runtimeResults.getThroughputKernelArray());
    metaData.setItem(BMInfoDataType::KernelDataUsageValues, joinElements(values, ", "));
    // Join elements and create a comma separated string and set item
    std::vector<double> valuesBW(runtimeResults.getBandwidthKernelVec());
    metaData.setItem(BMInfoDataType::KernelBandwidths, joinElements(valuesBW, ", "));

    metaData.setItem(BMInfoDataType::KernelMinTimes, joinElements(runtimeResults.getMinExecTimeKernelArray(), ", "));
    metaData.setItem(BMInfoDataType::KernelMaxTimes, joinElements(runtimeResults.getMaxExecTimeKernelArray(), ", "));
    metaData.setItem(BMInfoDataType::KernelAvgTimes, joinElements(runtimeResults.getAvgExecTimeKernelArray(), ", "));
    // Print the summary as a table, if a standard serialization is needed other functions of the class can be used
    std::cout << metaData.serializeAsTable() << std::endl;
}

using TestAccs1D = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::uint32_t>;

// Run for all Accs given by the argument
TEMPLATE_LIST_TEST_CASE("TEST: Babelstream Kernels<Float>", "[benchmark-test]", TestAccs1D)
{
    using Acc = TestType;
    // Run tests for the float data type
    testKernels<Acc, float>();
}

// Run for all Accs given by the argument
TEMPLATE_LIST_TEST_CASE("TEST: Babelstream Kernels<Double>", "[benchmark-test]", TestAccs1D)
{
    using Acc = TestType;
    // Run tests for the double data type
    testKernels<Acc, double>();
}
