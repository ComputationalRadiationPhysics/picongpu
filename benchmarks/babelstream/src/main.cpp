
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

// NOLINTBEGIN

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#define VERSION_STRING "4.0"

#include "Stream.h"

#if defined(CUDA)
#    include "CUDAStream.h"
#elif defined(STD_DATA)
#    include "STDDataStream.h"
#elif defined(STD_INDICES)
#    include "STDIndicesStream.h"
#elif defined(STD_RANGES)
#    include "STDRangesStream.hpp"
#elif defined(TBB)
#    include "TBBStream.hpp"
#elif defined(THRUST)
#    include "ThrustStream.h"
#elif defined(HIP)
#    include "HIPStream.h"
#elif defined(HC)
#    include "HCStream.h"
#elif defined(OCL)
#    include "OCLStream.h"
#elif defined(USE_RAJA)
#    include "RAJAStream.hpp"
#elif defined(KOKKOS)
#    include "KokkosStream.hpp"
#elif defined(ACC)
#    include "ACCStream.h"
#elif defined(SYCL)
#    include "SYCLStream.h"
#elif defined(SYCL2020)
#    include "SYCLStream2020.h"
#elif defined(OMP)
#    include "OMPStream.h"
#elif defined(ALPAKA)
#    include "AlpakaStream.h"
#endif

// Default size of 2^25
int ARRAY_SIZE = 33'554'432;
unsigned int num_times = 100;
unsigned int deviceIndex = 0;
bool use_float = false;
bool output_as_csv = false;
bool mibibytes = false;
std::string csv_separator = ",";

template<typename T>
void check_solution(unsigned int const ntimes, std::vector<T>& a, std::vector<T>& b, std::vector<T>& c, T& sum);

template<typename T>
void run();

// Options for running the benchmark:
// - All 5 kernels (Copy, Add, Mul, Triad, Dot).
// - Triad only.
// - Nstream only.
enum class Benchmark
{
    All,
    Triad,
    Nstream
};

// Selected run options.
Benchmark selection = Benchmark::All;

void parseArguments(int argc, char* argv[]);

int main(int argc, char* argv[])
{
    parseArguments(argc, argv);

    if(!output_as_csv)
    {
        std::cout << "BabelStream" << std::endl
                  << "Version: " << VERSION_STRING << std::endl
                  << "Implementation: " << IMPLEMENTATION_STRING << std::endl;
    }

    if(use_float)
        run<float>();
    else
        run<double>();
}

// Run the 5 main kernels
template<typename T>
std::vector<std::vector<double>> run_all(Stream<T>* stream, T& sum)
{
    // List of times
    std::vector<std::vector<double>> timings(5);

    // Declare timers
    std::chrono::high_resolution_clock::time_point t1, t2;

    // Main loop
    for(unsigned int k = 0; k < num_times; k++)
    {
        // Execute Copy
        t1 = std::chrono::high_resolution_clock::now();
        stream->copy();
        t2 = std::chrono::high_resolution_clock::now();
        timings[0].push_back(std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count());

        // Execute Mul
        t1 = std::chrono::high_resolution_clock::now();
        stream->mul();
        t2 = std::chrono::high_resolution_clock::now();
        timings[1].push_back(std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count());

        // Execute Add
        t1 = std::chrono::high_resolution_clock::now();
        stream->add();
        t2 = std::chrono::high_resolution_clock::now();
        timings[2].push_back(std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count());

        // Execute Triad
        t1 = std::chrono::high_resolution_clock::now();
        stream->triad();
        t2 = std::chrono::high_resolution_clock::now();
        timings[3].push_back(std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count());

        // Execute Dot
        t1 = std::chrono::high_resolution_clock::now();
        sum = stream->dot();
        t2 = std::chrono::high_resolution_clock::now();
        timings[4].push_back(std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count());
    }

    // Compiler should use a move
    return timings;
}

// Run the Triad kernel
template<typename T>
std::vector<std::vector<double>> run_triad(Stream<T>* stream)
{
    std::vector<std::vector<double>> timings(1);

    // Declare timers
    std::chrono::high_resolution_clock::time_point t1, t2;

    // Run triad in loop
    t1 = std::chrono::high_resolution_clock::now();
    for(unsigned int k = 0; k < num_times; k++)
    {
        stream->triad();
    }
    t2 = std::chrono::high_resolution_clock::now();

    double runtime = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    timings[0].push_back(runtime);

    return timings;
}

// Run the Nstream kernel
template<typename T>
std::vector<std::vector<double>> run_nstream(Stream<T>* stream)
{
    std::vector<std::vector<double>> timings(1);

    // Declare timers
    std::chrono::high_resolution_clock::time_point t1, t2;

    // Run nstream in loop
    for(int k = 0; k < num_times; k++)
    {
        t1 = std::chrono::high_resolution_clock::now();
        stream->nstream();
        t2 = std::chrono::high_resolution_clock::now();
        timings[0].push_back(std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count());
    }

    return timings;
}

// Generic run routine
// Runs the kernel(s) and prints output.
template<typename T>
void run()
{
    std::streamsize ss = std::cout.precision();

    if(!output_as_csv)
    {
        if(selection == Benchmark::All)
            std::cout << "Running kernels " << num_times << " times" << std::endl;
        else if(selection == Benchmark::Triad)
        {
            std::cout << "Running triad " << num_times << " times" << std::endl;
            std::cout << "Number of elements: " << ARRAY_SIZE << std::endl;
        }


        if(sizeof(T) == sizeof(float))
            std::cout << "Precision: float" << std::endl;
        else
            std::cout << "Precision: double" << std::endl;


        if(mibibytes)
        {
            // MiB = 2^20
            std::cout << std::setprecision(1) << std::fixed
                      << "Array size: " << ARRAY_SIZE * sizeof(T) * pow(2.0, -20.0) << " MiB"
                      << " (=" << ARRAY_SIZE * sizeof(T) * pow(2.0, -30.0) << " GiB)" << std::endl;
            std::cout << "Total size: " << 3.0 * ARRAY_SIZE * sizeof(T) * pow(2.0, -20.0) << " MiB"
                      << " (=" << 3.0 * ARRAY_SIZE * sizeof(T) * pow(2.0, -30.0) << " GiB)" << std::endl;
        }
        else
        {
            // MB = 10^6
            std::cout << std::setprecision(1) << std::fixed << "Array size: " << ARRAY_SIZE * sizeof(T) * 1.0E-6
                      << " MB"
                      << " (=" << ARRAY_SIZE * sizeof(T) * 1.0E-9 << " GB)" << std::endl;
            std::cout << "Total size: " << 3.0 * ARRAY_SIZE * sizeof(T) * 1.0E-6 << " MB"
                      << " (=" << 3.0 * ARRAY_SIZE * sizeof(T) * 1.0E-9 << " GB)" << std::endl;
        }
        std::cout.precision(ss);
    }

    Stream<T>* stream;

#if defined(CUDA)
    // Use the CUDA implementation
    stream = new CUDAStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(HIP)
    // Use the HIP implementation
    stream = new HIPStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(HC)
    // Use the HC implementation
    stream = new HCStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(OCL)
    // Use the OpenCL implementation
    stream = new OCLStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(USE_RAJA)
    // Use the RAJA implementation
    stream = new RAJAStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(KOKKOS)
    // Use the Kokkos implementation
    stream = new KokkosStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(STD_DATA)
    // Use the C++ STD data-oriented implementation
    stream = new STDDataStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(STD_INDICES)
    // Use the C++ STD index-oriented implementation
    stream = new STDIndicesStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(STD_RANGES)
    // Use the C++ STD ranges implementation
    stream = new STDRangesStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(TBB)
    // Use the C++20 implementation
    stream = new TBBStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(THRUST)
    // Use the Thrust implementation
    stream = new ThrustStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(ACC)
    // Use the OpenACC implementation
    stream = new ACCStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(SYCL) || defined(SYCL2020)
    // Use the SYCL implementation
    stream = new SYCLStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(OMP)
    // Use the OpenMP implementation
    stream = new OMPStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(ALPAKA)
    // Use the alpaka implementation
    stream = new AlpakaStream<T>(ARRAY_SIZE, deviceIndex);

#endif

    stream->init_arrays(startA, startB, startC);

    // Result of the Dot kernel, if used.
    T sum = 0.0;

    std::vector<std::vector<double>> timings;

    switch(selection)
    {
    case Benchmark::All:
        timings = run_all<T>(stream, sum);
        break;
    case Benchmark::Triad:
        timings = run_triad<T>(stream);
        break;
    case Benchmark::Nstream:
        timings = run_nstream<T>(stream);
        break;
    };

    // Check solutions
    // Create host vectors
    std::vector<T> a(ARRAY_SIZE);
    std::vector<T> b(ARRAY_SIZE);
    std::vector<T> c(ARRAY_SIZE);


    stream->read_arrays(a, b, c);
    check_solution<T>(num_times, a, b, c, sum);

    // Display timing results
    if(output_as_csv)
    {
        std::cout << "function" << csv_separator << "num_times" << csv_separator << "n_elements" << csv_separator
                  << "sizeof" << csv_separator << ((mibibytes) ? "max_mibytes_per_sec" : "max_mbytes_per_sec")
                  << csv_separator << "min_runtime" << csv_separator << "max_runtime" << csv_separator << "avg_runtime"
                  << std::endl;
    }
    else
    {
        std::cout << std::left << std::setw(12) << "Function" << std::left << std::setw(12)
                  << ((mibibytes) ? "MiBytes/sec" : "MBytes/sec") << std::left << std::setw(12) << "Min (sec)"
                  << std::left << std::setw(12) << "Max" << std::left << std::setw(12) << "Average" << std::endl
                  << std::fixed;
    }


    if(selection == Benchmark::All || selection == Benchmark::Nstream)
    {
        std::vector<std::string> labels;
        std::vector<size_t> sizes;

        if(selection == Benchmark::All)
        {
            labels = {"Copy", "Mul", "Add", "Triad", "Dot"};
            sizes
                = {2 * sizeof(T) * ARRAY_SIZE,
                   2 * sizeof(T) * ARRAY_SIZE,
                   3 * sizeof(T) * ARRAY_SIZE,
                   3 * sizeof(T) * ARRAY_SIZE,
                   2 * sizeof(T) * ARRAY_SIZE};
        }
        else if(selection == Benchmark::Nstream)
        {
            labels = {"Nstream"};
            sizes = {4 * sizeof(T) * ARRAY_SIZE};
        }

        for(int i = 0; i < timings.size(); ++i)
        {
            // Get min/max; ignore the first result
            auto minmax = std::minmax_element(timings[i].begin() + 1, timings[i].end());

            // Calculate average; ignore the first result
            double average = std::accumulate(timings[i].begin() + 1, timings[i].end(), 0.0) / (double) (num_times - 1);

            // Display results
            if(output_as_csv)
            {
                std::cout << labels[i] << csv_separator << num_times << csv_separator << ARRAY_SIZE << csv_separator
                          << sizeof(T) << csv_separator
                          << ((mibibytes) ? pow(2.0, -20.0) : 1.0E-6) * sizes[i] / (*minmax.first) << csv_separator
                          << *minmax.first << csv_separator << *minmax.second << csv_separator << average << std::endl;
            }
            else
            {
                std::cout << std::left << std::setw(12) << labels[i] << std::left << std::setw(12)
                          << std::setprecision(3)
                          << ((mibibytes) ? pow(2.0, -20.0) : 1.0E-6) * sizes[i] / (*minmax.first) << std::left
                          << std::setw(12) << std::setprecision(5) << *minmax.first << std::left << std::setw(12)
                          << std::setprecision(5) << *minmax.second << std::left << std::setw(12)
                          << std::setprecision(5) << average << std::endl;
            }
        }
    }
    else if(selection == Benchmark::Triad)
    {
        // Display timing results
        double total_bytes = 3 * sizeof(T) * ARRAY_SIZE * num_times;
        double bandwidth = ((mibibytes) ? pow(2.0, -30.0) : 1.0E-9) * (total_bytes / timings[0][0]);

        if(output_as_csv)
        {
            std::cout << "function" << csv_separator << "num_times" << csv_separator << "n_elements" << csv_separator
                      << "sizeof" << csv_separator << ((mibibytes) ? "gibytes_per_sec" : "gbytes_per_sec")
                      << csv_separator << "runtime" << std::endl;
            std::cout << "Triad" << csv_separator << num_times << csv_separator << ARRAY_SIZE << csv_separator
                      << sizeof(T) << csv_separator << bandwidth << csv_separator << timings[0][0] << std::endl;
        }
        else
        {
            std::cout << "--------------------------------" << std::endl
                      << std::fixed << "Runtime (seconds): " << std::left << std::setprecision(5) << timings[0][0]
                      << std::endl
                      << "Bandwidth (" << ((mibibytes) ? "GiB/s" : "GB/s") << "):  " << std::left
                      << std::setprecision(3) << bandwidth << std::endl;
        }
    }

    delete stream;
}

template<typename T>
void check_solution(unsigned int const ntimes, std::vector<T>& a, std::vector<T>& b, std::vector<T>& c, T& sum)
{
    // Generate correct solution
    T goldA = startA;
    T goldB = startB;
    T goldC = startC;
    T goldSum = 0.0;

    const T scalar = startScalar;

    for(unsigned int i = 0; i < ntimes; i++)
    {
        // Do STREAM!
        if(selection == Benchmark::All)
        {
            goldC = goldA;
            goldB = scalar * goldC;
            goldC = goldA + goldB;
            goldA = goldB + scalar * goldC;
        }
        else if(selection == Benchmark::Triad)
        {
            goldA = goldB + scalar * goldC;
        }
        else if(selection == Benchmark::Nstream)
        {
            goldA += goldB + scalar * goldC;
        }
    }

    // Do the reduction
    goldSum = goldA * goldB * ARRAY_SIZE;

    // Calculate the average error
    double errA
        = std::accumulate(a.begin(), a.end(), 0.0, [&](double sum, const T val) { return sum + fabs(val - goldA); });
    errA /= a.size();
    double errB
        = std::accumulate(b.begin(), b.end(), 0.0, [&](double sum, const T val) { return sum + fabs(val - goldB); });
    errB /= b.size();
    double errC
        = std::accumulate(c.begin(), c.end(), 0.0, [&](double sum, const T val) { return sum + fabs(val - goldC); });
    errC /= c.size();
    double errSum = fabs((sum - goldSum) / goldSum);

    double epsi = std::numeric_limits<T>::epsilon() * 100.0;

    if(errA > epsi)
        std::cerr << "Validation failed on a[]. Average error " << errA << std::endl;
    if(errB > epsi)
        std::cerr << "Validation failed on b[]. Average error " << errB << std::endl;
    if(errC > epsi)
        std::cerr << "Validation failed on c[]. Average error " << errC << std::endl;
    // Check sum to 8 decimal places
    if(selection == Benchmark::All && errSum > 1.0E-8)
        std::cerr << "Validation failed on sum. Error " << errSum << std::endl
                  << std::setprecision(15) << "Sum was " << sum << " but should be " << goldSum << std::endl;
}

int parseUInt(char const* str, unsigned int* output)
{
    char* next;
    *output = strtoul(str, &next, 10);
    return !strlen(next);
}

int parseInt(char const* str, int* output)
{
    char* next;
    *output = strtol(str, &next, 10);
    return !strlen(next);
}

void parseArguments(int argc, char* argv[])
{
    for(int i = 1; i < argc; i++)
    {
        if(!std::string("--list").compare(argv[i]))
        {
            listDevices();
            exit(EXIT_SUCCESS);
        }
        else if(!std::string("--device").compare(argv[i]))
        {
            if(++i >= argc || !parseUInt(argv[i], &deviceIndex))
            {
                std::cerr << "Invalid device index." << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        else if(!std::string("--arraysize").compare(argv[i]) || !std::string("-s").compare(argv[i]))
        {
            if(++i >= argc || !parseInt(argv[i], &ARRAY_SIZE) || ARRAY_SIZE <= 0)
            {
                std::cerr << "Invalid array size." << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        else if(!std::string("--numtimes").compare(argv[i]) || !std::string("-n").compare(argv[i]))
        {
            if(++i >= argc || !parseUInt(argv[i], &num_times))
            {
                std::cerr << "Invalid number of times." << std::endl;
                exit(EXIT_FAILURE);
            }
            if(num_times < 2)
            {
                std::cerr << "Number of times must be 2 or more" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        else if(!std::string("--float").compare(argv[i]))
        {
            use_float = true;
        }
        else if(!std::string("--triad-only").compare(argv[i]))
        {
            selection = Benchmark::Triad;
        }
        else if(!std::string("--nstream-only").compare(argv[i]))
        {
            selection = Benchmark::Nstream;
        }
        else if(!std::string("--csv").compare(argv[i]))
        {
            output_as_csv = true;
        }
        else if(!std::string("--mibibytes").compare(argv[i]))
        {
            mibibytes = true;
        }
        else if(!std::string("--help").compare(argv[i]) || !std::string("-h").compare(argv[i]))
        {
            std::cout << std::endl;
            std::cout << "Usage: " << argv[0] << " [OPTIONS]" << std::endl << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -h  --help               Print the message" << std::endl;
            std::cout << "      --list               List available devices" << std::endl;
            std::cout << "      --device     INDEX   Select device at INDEX" << std::endl;
            std::cout << "  -s  --arraysize  SIZE    Use SIZE elements in the array" << std::endl;
            std::cout << "  -n  --numtimes   NUM     Run the test NUM times (NUM >= 2)" << std::endl;
            std::cout << "      --float              Use floats (rather than doubles)" << std::endl;
            std::cout << "      --triad-only         Only run triad" << std::endl;
            std::cout << "      --nstream-only       Only run nstream" << std::endl;
            std::cout << "      --csv                Output as csv table" << std::endl;
            std::cout << "      --mibibytes          Use MiB=2^20 for bandwidth calculation (default MB=10^6)"
                      << std::endl;
            std::cout << std::endl;
            exit(EXIT_SUCCESS);
        }
        else
        {
            std::cerr << "Unrecognized argument '" << argv[i] << "' (try '--help')" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}

// NOLINTEND
