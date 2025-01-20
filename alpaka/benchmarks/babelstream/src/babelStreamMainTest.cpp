
#include "babelStreamCommon.hpp"
#include "catch2/catch_session.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <string>

/**
 * Babelstream benchmarking example. Babelstream has 5 kernels. Add, Multiply, Copy, Triad and Dot.
 * Babelstream is a memory-bound benchmark since the main operation in the kernels has high Code Balance (bytes/FLOP)
 * value. For example c[i] = a[i] + b[i]; has 2 reads 1 writes and has one FLOP operation. For double precision each
 * read-write is 8 bytes. Hence Code Balance (3*8 / 1) = 24 bytes/FLOP.
 *
 * Some implementations and the documents are accessible through https://github.com/UoB-HPC
 *
 * Can be run with custom arguments as well as catch2 arguments
 * Run with Custom arguments:
 * ./babelstream --array-size=33554432 --number-runs=100
 * Runt with default array size and num runs:
 * ./babelstream
 * Run with Catch2 arguments and defaul arrary size and num runs:
 * ./babelstream --success
 * ./babelstream -r a.xml
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
    //! \param initA the value to set all items in the vector
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T* a, T* b, T* c, T initA) const
    {
        auto const [i] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        a[i] = initA;
        b[i] = static_cast<T>(0.0);
        c[i] = static_cast<T>(0.0);
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
    //! \param b Pointer for vector b
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* a, T* b) const
    {
        auto const [index] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        b[index] = a[index];
    }
};

//! Kernel multiplies the vector with a scalar, scaling or multiplication kernel
struct MultKernel
{
    //! The kernel entry point
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam T The data type
    //! \param acc The accelerator to be executed on.
    //! \param a Pointer for vector a
    //! \param b Pointer for result vector b
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T* const a, T* b) const
    {
        const T scalar = static_cast<T>(scalarVal);
        auto const [i] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        b[i] = scalar * a[i];
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
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* a, T const* b, T* c) const
    {
        const T scalar = static_cast<T>(scalarVal);
        auto const [i] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        c[i] = a[i] + scalar * b[i];
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
    //! \param sum Pointer for result vector consisting sums for each block
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
                tbSum[local_i] += tbSum[local_i + offset];
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
    using Acc = TAcc;
    // Define the index domain
    // Set the number of dimensions as an integral constant. Set to 1 for 1D.
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    // Meta data
    // A MetaData class instance to keep the problem and results to print later
    MetaData metaData;
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
        static_cast<DataType>(valA));
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

    // Vector of average run-times of babelstream kernels
    std::vector<double> avgExecTimesOfKernels;
    std::vector<double> minExecTimesOfKernels;
    std::vector<double> maxExecTimesOfKernels;
    std::vector<std::string> kernelLabels;
    // Vector for collecting successive run-times of a single kernel in benchmark macro
    std::vector<double> times;

    // Lambda for measuring run-time
    auto measureKernelExec = [&](auto&& kernelFunc, [[maybe_unused]] auto&& kernelLabel)
    {
        for(auto i = 0; i < numberOfRuns; i++)
        {
            double runtime = 0.0;
            auto start = std::chrono::high_resolution_clock::now();
            kernelFunc();
            alpaka::wait(queue);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;
            runtime = duration.count();
            times.push_back(runtime);
        }

        // find the minimum of the durations array.
        // In benchmarking the first item of the runtimes array is not included in calculations.
        const auto minmaxPair = findMinMax(times);
        minExecTimesOfKernels.push_back(minmaxPair.first);
        maxExecTimesOfKernels.push_back(minmaxPair.second);
        avgExecTimesOfKernels.push_back(findAverage(times));
        kernelLabels.push_back(kernelLabel);
        times.clear();
    };

    // Run kernels one by one
    // Test the init-kernel.
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
                static_cast<DataType>(valA));
        },
        "InitKernel");

    // Test the copy-kernel. Copy A one by one to B.
    measureKernelExec(
        [&]() { alpaka::exec<Acc>(queue, workDivCopy, CopyKernel(), bufAccInputAPtr, bufAccInputBPtr); },
        "CopyKernel");

    // Test the scaling-kernel. Calculate B=scalar*A.
    measureKernelExec(
        [&]() { alpaka::exec<Acc>(queue, workDivMult, MultKernel(), bufAccInputAPtr, bufAccInputBPtr); },
        "MultKernel");

    // Test the addition-kernel. Calculate C=A+B. Where B=scalar*A.
    measureKernelExec(
        [&]()
        { alpaka::exec<Acc>(queue, workDivAdd, AddKernel(), bufAccInputAPtr, bufAccInputBPtr, bufAccOutputCPtr); },
        "AddKernel");

    // Test the Triad-kernel. Calculate C=A+scalar*B where B=scalar*A.
    measureKernelExec(
        [&]()
        { alpaka::exec<Acc>(queue, workDivTriad, TriadKernel(), bufAccInputAPtr, bufAccInputBPtr, bufAccOutputCPtr); },
        "TriadKernel");


    // Copy arrays back to host
    alpaka::memcpy(queue, bufHostOutputC, bufAccOutputC, arraySize);
    alpaka::memcpy(queue, bufHostOutputB, bufAccInputB, arraySize);
    alpaka::memcpy(queue, bufHostOutputA, bufAccInputA, arraySize);

    // Verify the results
    //
    // Find sum of the errors as sum of the differences from expected values
    DataType initVal{static_cast<DataType>(0.0)};
    DataType sumErrC{initVal}, sumErrB{initVal}, sumErrA{initVal};

    auto const expectedC = static_cast<DataType>(valA + scalarVal * scalarVal * valA);
    auto const expectedB = static_cast<DataType>(scalarVal * valA);
    auto const expectedA = static_cast<DataType>(valA);

    // sum of the errors for each array
    for(Idx i = 0; i < arraySize; ++i)
    {
        sumErrC += bufHostOutputC[static_cast<Idx>(i)] - expectedC;
        sumErrB += bufHostOutputB[static_cast<Idx>(i)] - expectedB;
        sumErrA += bufHostOutputA[static_cast<Idx>(i)] - expectedA;
    }

    // Normalize and compare sum of the errors
    REQUIRE(FuzzyEqual(sumErrC / static_cast<DataType>(arraySize) / expectedC, static_cast<DataType>(0.0)));
    REQUIRE(FuzzyEqual(sumErrB / static_cast<DataType>(arraySize) / expectedB, static_cast<DataType>(0.0)));
    REQUIRE(FuzzyEqual(sumErrA / static_cast<DataType>(arraySize) / expectedA, static_cast<DataType>(0.0)));
    alpaka::wait(queue);

    // Test Dot kernel with specific blocksize which is larger than 1
    if constexpr(alpaka::accMatchesTags<TAcc, alpaka::TagGpuCudaRt, alpaka::TagGpuHipRt, alpaka::TagGpuSyclIntel>)
    {
        using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
        // Threads per block for Dot kernel
        constexpr Idx blockThreadExtent = blockThreadExtentMain;
        // Blocks per grid for Dot kernel
        constexpr Idx gridBlockExtent = static_cast<Idx>(256);
        // Vector of sums of each block
        auto bufAccSumPerBlock = alpaka::allocBuf<DataType, Idx>(devAcc, gridBlockExtent);
        auto bufHostSumPerBlock = alpaka::allocBuf<DataType, Idx>(devHost, gridBlockExtent);
        // A specific work-division is used for dotKernel
        auto const workDivDot = WorkDiv{Vec{gridBlockExtent}, Vec{blockThreadExtent}, Vec::all(1)};

        measureKernelExec(
            [&]()
            {
                alpaka::exec<Acc>(
                    queue,
                    workDivDot,
                    DotKernel(), // Dot kernel
                    alpaka::getPtrNative(bufAccInputA),
                    alpaka::getPtrNative(bufAccInputB),
                    alpaka::getPtrNative(bufAccSumPerBlock),
                    static_cast<alpaka::Idx<Acc>>(arraySize));
            },
            "DotKernel");

        alpaka::memcpy(queue, bufHostSumPerBlock, bufAccSumPerBlock, gridBlockExtent);
        alpaka::wait(queue);

        DataType const* sumPtr = std::data(bufHostSumPerBlock);
        auto const result = std::reduce(sumPtr, sumPtr + gridBlockExtent, DataType{0});
        // Since vector values are 1, dot product should be identical to arraySize
        REQUIRE(FuzzyEqual(static_cast<DataType>(result), static_cast<DataType>(arraySize * 2)));
        // Add workdiv to the list of workdivs to print later
        metaData.setItem(BMInfoDataType::WorkDivDot, workDivDot);
    }


    //
    // Calculate and Display Benchmark Results
    //
    std::vector<double> bytesReadWriteMB = {
        getDataThroughput<DataType>(2u, static_cast<unsigned>(arraySize)),
        getDataThroughput<DataType>(2u, static_cast<unsigned>(arraySize)),
        getDataThroughput<DataType>(2u, static_cast<unsigned>(arraySize)),
        getDataThroughput<DataType>(3u, static_cast<unsigned>(arraySize)),
        getDataThroughput<DataType>(3u, static_cast<unsigned>(arraySize)),
        getDataThroughput<DataType>(2u, static_cast<unsigned>(arraySize)),
    };

    // calculate the bandwidth as throughput per seconds
    std::vector<double> bandwidthsPerKernel;
    if(minExecTimesOfKernels.size() == kernelLabels.size())
    {
        for(size_t i = 0; i < minExecTimesOfKernels.size(); ++i)
        {
            bandwidthsPerKernel.push_back(calculateBandwidth(bytesReadWriteMB.at(i), minExecTimesOfKernels.at(i)));
        }
    }

    // Setting fields of Benchmark Info map. All information about benchmark and results are stored in a single map
    metaData.setItem(BMInfoDataType::TimeStamp, getCurrentTimestamp());
    metaData.setItem(BMInfoDataType::NumRuns, std::to_string(numberOfRuns));
    metaData.setItem(BMInfoDataType::DataSize, std::to_string(arraySizeMain));
    metaData.setItem(BMInfoDataType::DataType, dataTypeStr);

    metaData.setItem(BMInfoDataType::WorkDivInit, workDivInit);
    metaData.setItem(BMInfoDataType::WorkDivCopy, workDivCopy);
    metaData.setItem(BMInfoDataType::WorkDivAdd, workDivAdd);
    metaData.setItem(BMInfoDataType::WorkDivMult, workDivMult);
    metaData.setItem(BMInfoDataType::WorkDivTriad, workDivTriad);

    // Device and accelerator
    metaData.setItem(BMInfoDataType::DeviceName, alpaka::getName(devAcc));
    metaData.setItem(BMInfoDataType::AcceleratorType, alpaka::getAccName<Acc>());
    // XML reporter of catch2 always converts to Nano Seconds
    metaData.setItem(BMInfoDataType::TimeUnit, "Nano Seconds");
    // Join elements and create a comma separated string
    metaData.setItem(BMInfoDataType::KernelNames, joinElements(kernelLabels, ", "));
    metaData.setItem(BMInfoDataType::KernelDataUsageValues, joinElements(bytesReadWriteMB, ", "));
    metaData.setItem(BMInfoDataType::KernelBandwidths, joinElements(bandwidthsPerKernel, ", "));
    metaData.setItem(BMInfoDataType::KernelMinTimes, joinElements(minExecTimesOfKernels, ", "));
    metaData.setItem(BMInfoDataType::KernelMaxTimes, joinElements(maxExecTimesOfKernels, ", "));
    metaData.setItem(BMInfoDataType::KernelAvgTimes, joinElements(avgExecTimesOfKernels, ", "));

    // Print the summary as a table, if a standard serialization is needed other functions of the class can be used
    std::cout << metaData.serializeAsTable() << std::endl;
}

using TestAccs1D = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::uint32_t>;

// Run for all Accs given by the argument
TEMPLATE_LIST_TEST_CASE("TEST: Babelstream Five Kernels<Float>", "[benchmark-test]", TestAccs1D)
{
    using Acc = TestType;
    // Run tests for the float data type
    testKernels<Acc, float>();
}

// Run for all Accs given by the argument
TEMPLATE_LIST_TEST_CASE("TEST: Babelstream Five Kernels<Double>", "[benchmark-test]", TestAccs1D)
{
    using Acc = TestType;
    // Run tests for the double data type
    testKernels<Acc, double>();
}
