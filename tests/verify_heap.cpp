/*
  mallocMC: Memory Allocator for Many Core Architectures.
  https://www.hzdr.de/crp

  Copyright 2014 Institute of Radiation Physics,
                 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Carlchristian Eckert - c.eckert ( at ) hzdr.de

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

// each pointer in the datastructure will point to this many
// elements of type allocElem_t
constexpr auto ELEMS_PER_SLOT = 750;

#include "verify_heap_config.hpp"

#include <alpaka/alpaka.hpp>
#include <cstdio>
#include <iostream>
#include <mallocMC/mallocMC_utils.hpp>
#include <sstream>
#include <typeinfo>
#include <vector>

using Device = alpaka::Dev<Acc>;
using Queue = alpaka::Queue<Acc, alpaka::Blocking>;

// global variable for verbosity, might change due to user input '--verbose'
bool verbose = false;

// the type of the elements to allocate
using allocElem_t = unsigned long long;

auto run_heap_verification(const size_t, const unsigned, const unsigned, const bool) -> bool;
void parse_cmdline(const int, char**, size_t*, unsigned*, unsigned*, bool*);
void print_help(char**);

// used to create an empty stream for non-verbose output
struct nullstream : std::ostream
{
    nullstream() : std::ostream(0)
    {
    }
};

// uses global verbosity to switch between std::cout and a nullptr-output
auto dout() -> std::ostream&
{
    static nullstream n;
    return verbose ? std::cout : n;
}

// define some defaults
static constexpr unsigned threads_default = 128;
static constexpr unsigned blocks_default = 64;
static constexpr size_t heapInMB_default = 1024; // 1GB

/**
 * will do a basic verification of scatterAlloc.
 *
 * @param argv if -q or --quiet is supplied as a
 *        command line argument, verbosity will be reduced
 *
 * @return will return 0 if the verification was successful,
 *         otherwise returns 1
 */
auto main(int argc, char** argv) -> int
{
    bool machine_readable = false;
    size_t heapInMB = heapInMB_default;
    unsigned threads = threads_default;
    unsigned blocks = blocks_default;

    parse_cmdline(argc, argv, &heapInMB, &threads, &blocks, &machine_readable);

    const auto correct = run_heap_verification(heapInMB, threads, blocks, machine_readable);
    if(!machine_readable || verbose)
    {
        if(correct)
        {
            std::cout << "\033[0;32mverification successful ✔\033[0m\n";
            return 0;
        }
        else
        {
            std::cerr << "\033[0;31mverification failed\033[0m\n";
            return 1;
        }
    }
}

/**
 * will parse command line arguments
 *
 * for more details, see print_help()
 *
 * @param argc argc from main()
 * @param argv argv from main()
 * @param heapInMP will be filled with the heapsize, if given as a parameter
 * @param threads will be filled with number of threads, if given as a parameter
 * @param blocks will be filled with number of blocks, if given as a parameter
 */
void parse_cmdline(
    const int argc,
    char** argv,
    size_t* heapInMB,
    unsigned* threads,
    unsigned* blocks,
    bool* machine_readable)
{
    std::vector<std::pair<std::string, std::string>> parameters;

    // Parse Commandline, tokens are shaped like ARG=PARAM or ARG
    // This requires to use '=', if you want to supply a value with a parameter
    for(int i = 1; i < argc; ++i)
    {
        char* pos = strtok(argv[i], "=");
        std::pair<std::string, std::string> p(std::string(pos), std::string(""));
        pos = strtok(nullptr, "=");
        if(pos != nullptr)
        {
            p.second = std::string(pos);
        }
        parameters.push_back(p);
    }

    // go through all parameters that were found
    for(unsigned i = 0; i < parameters.size(); ++i)
    {
        std::pair<std::string, std::string> p = parameters.at(i);

        if(p.first == "-v" || p.first == "--verbose")
        {
            verbose = true;
        }

        if(p.first == "--threads")
        {
            *threads = atoi(p.second.c_str());
        }

        if(p.first == "--blocks")
        {
            *blocks = atoi(p.second.c_str());
        }

        if(p.first == "--heapsize")
        {
            *heapInMB = size_t(atoi(p.second.c_str()));
        }

        if(p.first == "-h" || p.first == "--help")
        {
            print_help(argv);
            exit(0);
        }

        if(p.first == "-m" || p.first == "--machine_readable")
        {
            *machine_readable = true;
        }
    }
}

/**
 * prints a helpful message about program use
 *
 * @param argv the argv-parameter from main, used to find the program name
 */
void print_help(char** argv)
{
    std::stringstream s;

    s << "SYNOPSIS:" << '\n';
    s << argv[0] << " [OPTIONS]" << '\n';
    s << "" << '\n';
    s << "OPTIONS:" << '\n';
    s << "  -h, --help" << '\n';
    s << "    Print this help message and exit" << '\n';
    s << "" << '\n';
    s << "  -v, --verbose" << '\n';
    s << "    Print information about parameters and progress" << '\n';
    s << "" << '\n';
    s << "  -m, --machine_readable" << '\n';
    s << "    Print all relevant parameters as CSV. This will" << '\n';
    s << "    suppress all other output unless explicitly" << '\n';
    s << "    requested with --verbose or -v" << '\n';
    s << "" << '\n';
    s << "  --threads=N" << '\n';
    s << "    Set the number of threads per block (default ";
    s << threads_default << "128)" << '\n';
    s << "" << '\n';
    s << "  --blocks=N" << '\n';
    s << "    Set the number of blocks in the grid (default ";
    s << blocks_default << ")" << '\n';
    s << "" << '\n';
    s << "  --heapsize=N" << '\n';
    s << "    Set the heapsize to N Megabyte (default ";
    s << heapInMB_default << "1024)" << '\n';

    std::cout << s.str() << std::flush;
}

/**
 * checks validity of memory for each single cell
 *
 * checks on a per thread basis, if the values written during
 * allocation are still the same. Also calculates the sum over
 * all allocated values for a more in-depth verification that
 * could be done on the host
 *
 * @param data the data to verify
 * @param counter should be initialized with 0 and will
 *        be used to count how many verifications were
 *        already done
 * @param globalSum will be filled with the sum over all
 *        allocated values in the structure
 * @param nSlots the size of the datastructure
 * @param correct should be initialized with 1.
 *        Will change to 0, if there was a value that didn't match
 */
struct Check_content
{
    ALPAKA_FN_ACC void operator()(
        const Acc& acc,
        allocElem_t** data,
        unsigned long long* counter,
        unsigned long long* globalSum,
        const size_t nSlots,
        int* correct) const
    {
        unsigned long long sum = 0;
        while(true)
        {
            const size_t pos = alpaka::atomicOp<alpaka::AtomicAdd>(acc, counter, 1ull);
            if(pos >= nSlots)
            {
                break;
            }
            const size_t offset = pos * ELEMS_PER_SLOT;
            for(size_t i = 0; i < ELEMS_PER_SLOT; ++i)
            {
                if(static_cast<allocElem_t>(data[pos][i]) != static_cast<allocElem_t>(offset + i))
                {
                    // printf("\nError in Kernel: data[%llu][%llu] is %#010x
                    // (should be %#010x)\n",
                    //    pos,i,static_cast<allocElem_t>(data[pos][i]),allocElem_t(offset+i));
                    alpaka::atomicOp<alpaka::AtomicAnd>(acc, correct, 0);
                }
                sum += static_cast<unsigned long long>(data[pos][i]);
            }
        }
        alpaka::atomicOp<alpaka::AtomicAdd>(acc, globalSum, sum);
    }
};

/**
 * checks validity of memory for each single cell
 *
 * checks on a per thread basis, if the values written during
 * allocation are still the same.
 *
 * @param data the data to verify
 * @param counter should be initialized with 0 and will
 *        be used to count how many verifications were
 *        already done
 * @param nSlots the size of the datastructure
 * @param correct should be initialized with 1.
 *        Will change to 0, if there was a value that didn't match
 */
struct Check_content_fast
{
    ALPAKA_FN_ACC void operator()(
        const Acc& acc,
        allocElem_t** data,
        unsigned long long* counter,
        const size_t nSlots,
        int* correct) const
    {
        int c = 1;
        while(true)
        {
            size_t pos = alpaka::atomicOp<alpaka::AtomicAdd>(acc, counter, 1ull);
            if(pos >= nSlots)
            {
                break;
            }
            const size_t offset = pos * ELEMS_PER_SLOT;
            for(size_t i = 0; i < ELEMS_PER_SLOT; ++i)
            {
                if(static_cast<allocElem_t>(data[pos][i]) != static_cast<allocElem_t>(offset + i))
                {
                    c = 0;
                }
            }
        }
        alpaka::atomicOp<alpaka::AtomicAnd>(acc, correct, c);
    }
};

/**
 * allocate a lot of small arrays and fill them
 *
 * Each array has the size ELEMS_PER_SLOT and the type allocElem_t.
 * Each element will be filled with a number that is related to its
 * position in the datastructure.
 *
 * @param data the datastructure to allocate
 * @param counter should be initialized with 0 and will
 *        hold, how many allocations were done
 * @param globalSum will hold the sum of all values over all
 *        allocated structures (for verification purposes)
 */
struct AllocAll
{
    ALPAKA_FN_ACC void operator()(
        const Acc& acc,
        allocElem_t** data,
        unsigned long long* counter,
        unsigned long long* globalSum,
        ScatterAllocator::AllocatorHandle mMC) const
    {
        unsigned long long sum = 0;
        while(true)
        {
            allocElem_t* p = (allocElem_t*) mMC.malloc(acc, sizeof(allocElem_t) * ELEMS_PER_SLOT);
            if(p == nullptr)
                break;

            size_t pos = alpaka::atomicOp<alpaka::AtomicAdd>(acc, counter, 1ull);
            const size_t offset = pos * ELEMS_PER_SLOT;
            for(size_t i = 0; i < ELEMS_PER_SLOT; ++i)
            {
                p[i] = static_cast<allocElem_t>(offset + i);
                sum += static_cast<unsigned long long>(p[i]);
            }
            data[pos] = p;
        }

        alpaka::atomicOp<alpaka::AtomicAdd>(acc, globalSum, sum);
    }
};

/**
 * free all the values again
 *
 * @param data the datastructure to free
 * @param counter should be an empty space on device memory,
 *        counts how many elements were freed
 * @param max the maximum number of elements to free
 */
struct DeallocAll
{
    ALPAKA_FN_ACC void operator()(
        const Acc& acc,
        allocElem_t** data,
        unsigned long long* counter,
        const size_t nSlots,
        ScatterAllocator::AllocatorHandle mMC) const
    {
        while(true)
        {
            size_t pos = alpaka::atomicOp<alpaka::AtomicAdd>(acc, counter, 1ull);
            if(pos >= nSlots)
                break;
            mMC.free(acc, data[pos]);
        }
    }
};

/**
 * damages one element in the data
 *
 * With help of this function, you can verify that
 * the checks actually work as expected and can find
 * an error, if one should exist
 *
 * @param data the datastructure to damage
 */
struct DamageElement
{
    ALPAKA_FN_ACC void operator()(const Acc& acc, allocElem_t** data) const
    {
        data[1][0] = static_cast<allocElem_t>(5 * ELEMS_PER_SLOT - 1);
    }
};

/**
 * wrapper function to allocate memory on device
 *
 * allocates memory with mallocMC. Returns the number of
 * created elements as well as the sum of these elements
 *
 * @param d_testData the datastructure which will hold
 *        pointers to the created elements
 * @param h_nSlots will be filled with the number of elements
 *        that were allocated
 * @param h_sum will be filled with the sum of all elements created
 * @param blocks the size of the CUDA grid
 * @param threads the number of CUDA threads per block
 */
void allocate(
    const Device& dev,
    Queue& queue,
    alpaka::Buf<Device, allocElem_t*, Dim, Idx>& d_testData,
    unsigned long long* nSlots,
    unsigned long long* sum,
    const unsigned blocks,
    const unsigned threads,
    ScatterAllocator& mMC)
{
    dout() << "allocating on device...";

    auto d_sum = alpaka::allocBuf<unsigned long long, Idx>(dev, Idx{1});
    auto d_nSlots = alpaka::allocBuf<unsigned long long, Idx>(dev, Idx{1});

    alpaka::memset(queue, d_sum, 0, 1);
    alpaka::memset(queue, d_nSlots, 0, 1);

    const auto workDiv = alpaka::WorkDivMembers<Dim, Idx>{Idx{blocks}, Idx{threads}, Idx{1}};
    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc>(
            workDiv,
            AllocAll{},
            alpaka::getPtrNative(d_testData),
            alpaka::getPtrNative(d_nSlots),
            alpaka::getPtrNative(d_sum),
            mMC.getAllocatorHandle()));

    const auto hostDev = alpaka::getDevByIdx<alpaka::Pltf<alpaka::DevCpu>>(0);
    auto h_sum = alpaka::allocBuf<unsigned long long, Idx>(hostDev, Idx{1});
    auto h_nSlots = alpaka::allocBuf<unsigned long long, Idx>(hostDev, Idx{1});

    alpaka::memcpy(queue, h_sum, d_sum, Idx{1});
    alpaka::memcpy(queue, h_nSlots, d_nSlots, Idx{1});
    alpaka::wait(queue);

    *sum = *alpaka::getPtrNative(h_sum);
    *nSlots = *alpaka::getPtrNative(h_nSlots);

    dout() << "done\n";
}

/**
 * Wrapper function to verify allocation on device
 *
 * Generates the same number that was written into each position of
 * the datastructure during allocation and compares the values.
 *
 * @param d_testData the datastructure which holds
 *        pointers to the elements you want to verify
 * @param nSlots the size of d_testData
 * @param blocks the size of the CUDA grid
 * @param threads the number of CUDA threads per block
 * @return true if the verification was successful, false otherwise
 */
auto verify(
    const Device& dev,
    Queue& queue,
    alpaka::Buf<Device, allocElem_t*, Dim, Idx>& d_testData,
    const unsigned long long nSlots,
    const unsigned blocks,
    const unsigned threads) -> bool
{
    dout() << "verifying on device... ";

    const auto hostDev = alpaka::getDevByIdx<alpaka::Pltf<alpaka::DevCpu>>(0);
    auto h_correct = alpaka::allocBuf<int, Idx>(hostDev, Idx{1});
    *alpaka::getPtrNative(h_correct) = 1;

    auto d_sum = alpaka::allocBuf<unsigned long long, Idx>(dev, Idx{1});
    auto d_counter = alpaka::allocBuf<unsigned long long, Idx>(dev, Idx{1});
    auto d_correct = alpaka::allocBuf<int, Idx>(dev, Idx{1});

    alpaka::memset(queue, d_sum, 0, 1);
    alpaka::memset(queue, d_counter, 0, 1);
    alpaka::memcpy(queue, d_correct, h_correct, 1);

    // can be replaced by a call to check_content_fast,
    // if the gaussian sum (see below) is not used and you
    // want to be a bit faster
    const auto workDiv = alpaka::WorkDivMembers<Dim, Idx>{Idx{blocks}, Idx{threads}, Idx{1}};
    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc>(
            workDiv,
            Check_content{},
            alpaka::getPtrNative(d_testData),
            alpaka::getPtrNative(d_counter),
            alpaka::getPtrNative(d_sum),
            static_cast<size_t>(nSlots),
            alpaka::getPtrNative(d_correct)));

    alpaka::memcpy(queue, h_correct, d_correct, 1);
    alpaka::wait(queue);

    const auto correct = *alpaka::getPtrNative(h_correct);
    dout() << (correct ? "done\n" : "failed\n");
    return correct != 0;
}

/**
 * prints all parameters machine readable
 *
 * for params, see run_heap_verification-internal parameters
 */
void print_machine_readable(
    const unsigned pagesize,
    const unsigned accessblocks,
    const unsigned regionsize,
    const unsigned wastefactor,
    const bool resetfreedpages,
    const unsigned blocks,
    const unsigned threads,
    const unsigned elemsPerSlot,
    const size_t allocElemSize,
    const size_t heapSize,
    const size_t maxSpace,
    const size_t maxSlots,
    const unsigned long long usedSlots,
    const float allocFrac,
    const size_t wasted,
    const bool correct)
{
    std::string sep = ",";
    std::stringstream h;
    std::stringstream v;

    h << "PagesizeByte" << sep;
    v << pagesize << sep;

    h << "Accessblocks" << sep;
    v << accessblocks << sep;

    h << "Regionsize" << sep;
    v << regionsize << sep;

    h << "Wastefactor" << sep;
    v << wasted << sep;

    h << "ResetFreedPage" << sep;
    v << resetfreedpages << sep;

    h << "Gridsize" << sep;
    v << blocks << sep;

    h << "Blocksize" << sep;
    v << threads << sep;

    h << "ELEMS_PER_SLOT" << sep;
    v << elemsPerSlot << sep;

    h << "allocElemByte" << sep;
    v << allocElemSize << sep;

    h << "heapsizeByte" << sep;
    v << heapSize << sep;

    h << "maxSpaceByte" << sep;
    v << maxSpace << sep;

    h << "maxSlots" << sep;
    v << maxSlots << sep;

    h << "usedSlots" << sep;
    v << usedSlots << sep;

    h << "allocFraction" << sep;
    v << allocFrac << sep;

    h << "wastedBytes" << sep;
    v << wasted << sep;

    h << "correct";
    v << correct;

    std::cout << h.str() << '\n';
    std::cout << v.str() << '\n';
}

/**
 * Verify the heap allocation of mallocMC
 *
 * Allocates as much memory as the heap allows. Make sure that allocated
 * memory actually holds the correct values without corrupting them. Will
 * fill the datastructure with values that are relative to the index and
 * later evalute, if the values inside stayed the same after allocating all
 * memory.
 * Datastructure: Array that holds up to nPointers pointers to arrays of size
 * ELEMS_PER_SLOT, each being of type allocElem_t.
 *
 * @return true if the verification was successful,
 *         false otherwise
 */
auto run_heap_verification(
    const size_t heapMB,
    const unsigned blocks,
    const unsigned threads,
    const bool machine_readable) -> bool
{
    const auto dev = alpaka::getDevByIdx<Acc>(0);
    auto queue = Queue{dev};

    const size_t heapSize = size_t(1024U * 1024U) * heapMB;
    const size_t slotSize = sizeof(allocElem_t) * ELEMS_PER_SLOT;
    const size_t nPointers = (heapSize + slotSize - 1) / slotSize;
    const size_t maxSlots = heapSize / slotSize;
    const size_t maxSpace = maxSlots * slotSize + nPointers * sizeof(allocElem_t*);
    bool correct = true;

    dout() << "CreationPolicy Arguments:\n";
    dout() << "Pagesize:              " << ScatterConfig::pagesize << '\n';
    dout() << "Accessblocks:          " << ScatterConfig::accessblocks << '\n';
    dout() << "Regionsize:            " << ScatterConfig::regionsize << '\n';
    dout() << "Wastefactor:           " << ScatterConfig::wastefactor << '\n';
    dout() << "ResetFreedPages        " << ScatterConfig::resetfreedpages << '\n';
    dout() << "\n";
    dout() << "Gridsize:              " << blocks << '\n';
    dout() << "Blocksize:             " << threads << '\n';
    dout() << "Allocated elements:    " << ELEMS_PER_SLOT << " x " << sizeof(allocElem_t);
    dout() << "    Byte (" << slotSize << " Byte)\n";
    dout() << "Heap:                  " << heapSize << " Byte";
    dout() << " (" << heapSize / pow(1024, 2) << " MByte)\n";
    dout() << "max space w/ pointers: " << maxSpace << " Byte";
    dout() << " (" << maxSpace / pow(1024, 2) << " MByte)\n";
    dout() << "maximum of elements:   " << maxSlots << '\n';

    unsigned long long usedSlots = 0;
    unsigned long long sumAllocElems = 0;
    float allocFrac = 0;
    size_t wasted = 0;

    {
        ScatterAllocator mMC(dev, queue, heapSize);

        // allocating with mallocMC
        auto d_testData = alpaka::allocBuf<allocElem_t*, Idx>(dev, Idx{nPointers});
        allocate(dev, queue, d_testData, &usedSlots, &sumAllocElems, blocks, threads, mMC);

        allocFrac = static_cast<float>(usedSlots) * 100 / maxSlots;
        wasted = heapSize - static_cast<size_t>(usedSlots) * slotSize;
        dout() << "allocated elements:    " << usedSlots;
        dout() << " (" << allocFrac << "%)\n";
        dout() << "wasted heap space:     " << wasted << " Byte";
        dout() << " (" << wasted / pow(1024, 2) << " MByte)\n";

        // verifying on device
        correct = correct && verify(dev, queue, d_testData, usedSlots, blocks, threads);

        // damaging one cell
        dout() << "damaging of element... ";
        {
            const auto workDiv = alpaka::WorkDivMembers<Dim, Idx>{Idx{1}, Idx{1}, Idx{1}};
            alpaka::enqueue(
                queue,
                alpaka::createTaskKernel<Acc>(
                    workDiv,
                    DamageElement{},
                    alpaka::getPtrNative(d_testData)));
        }
        dout() << "done\n";

        // verifying on device
        // THIS SHOULD FAIL (damage was done before!). Therefore, we must
        // inverse the logic
        correct = correct && !verify(dev, queue, d_testData, usedSlots, blocks, threads);

        // release all memory
        dout() << "deallocation...        ";
        auto d_dealloc_counter = alpaka::allocBuf<unsigned long long, Idx>(dev, Idx{1});
        alpaka::memset(queue, d_dealloc_counter, 0, 1);
        {
            const auto workDiv = alpaka::WorkDivMembers<Dim, Idx>{Idx{blocks}, Idx{threads}, Idx{1}};
            alpaka::enqueue(
                queue,
                alpaka::createTaskKernel<Acc>(
                    workDiv,
                    DeallocAll{},
                    alpaka::getPtrNative(d_testData),
                    alpaka::getPtrNative(d_dealloc_counter),
                    static_cast<size_t>(usedSlots),
                    mMC.getAllocatorHandle()));
        }
    }

    dout() << "done \n";

    if(machine_readable)
    {
        print_machine_readable(
            ScatterConfig::pagesize,
            ScatterConfig::accessblocks,
            ScatterConfig::regionsize,
            ScatterConfig::wastefactor,
            ScatterConfig::resetfreedpages,
            blocks,
            threads,
            ELEMS_PER_SLOT,
            sizeof(allocElem_t),
            heapSize,
            maxSpace,
            maxSlots,
            usedSlots,
            allocFrac,
            wasted,
            correct);
    }

    return correct;
}
