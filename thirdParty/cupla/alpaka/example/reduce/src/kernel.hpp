/* Copyright 2020 Jonas Schenke, Bernhard Manfred Gruber
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>

//! A cheap wrapper around a C-style array in heap memory.
template<typename T, uint64_t size>
struct cheapArray
{
    T data[size];

    //! Access operator.
    //!
    //! \param index The index of the element to be accessed.
    //!
    //! Returns the requested element per reference.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator[](uint64_t index) -> T&
    {
        return data[index];
    }

    //! Access operator.
    //!
    //! \param index The index of the element to be accessed.
    //!
    //! Returns the requested element per constant reference.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator[](uint64_t index) const -> T const&
    {
        return data[index];
    }
};

//! A reduction kernel.
//!
//! \tparam TBlockSize The block size.
//! \tparam T The data type.
//! \tparam TFunc The Functor type for the reduction function.
template<uint32_t TBlockSize, typename T, typename TFunc>
struct ReduceKernel
{
    ALPAKA_NO_HOST_ACC_WARNING

    //! The kernel entry point.
    //!
    //! \tparam TAcc The accelerator environment.
    //! \tparam TElem The element type.
    //! \tparam TIdx The index type.
    //!
    //! \param acc The accelerator object.
    //! \param source The source memory.
    //! \param destination The destination memory.
    //! \param n The problem size.
    //! \param func The reduction function.
    template<typename TAcc, typename TElem, typename TIdx>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        TElem const* const source,
        TElem* destination,
        TIdx const& n,
        TFunc func) const -> void
    {
        auto& sdata(alpaka::declareSharedVar<cheapArray<T, TBlockSize>, __COUNTER__>(acc));

        auto const blockIndex(static_cast<uint32_t>(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0]));
        auto const threadIndex(static_cast<uint32_t>(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0]));
        auto const gridDimension(static_cast<uint32_t>(alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0]));

        // equivalent to blockIndex * TBlockSize + threadIndex
        auto const linearizedIndex(static_cast<uint32_t>(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0]));

        typename GetIterator<T, TElem, TAcc>::Iterator it(acc, source, linearizedIndex, gridDimension * TBlockSize, n);

        T result = 0; // suppresses compiler warnings

        if(threadIndex < n)
            result = *(it++); // avoids using the
                              // neutral element of specific

        // --------
        // Level 1: grid reduce, reading from global memory
        // --------

        // reduce per thread with increased ILP by 4x unrolling sum.
        // the thread of our block reduces its 4 grid-neighbored threads and
        // advances by grid-striding loop (maybe 128bit load improve perf)

        while(it + 3 < it.end())
        {
            result = func(func(func(result, func(*it, *(it + 1))), *(it + 2)), *(it + 3));
            it += 4;
        }

        // doing the remaining blocks
        while(it < it.end())
            result = func(result, *(it++));

        if(threadIndex < n)
            sdata[threadIndex] = result;

        alpaka::syncBlockThreads(acc);

        // --------
        // Level 2: block + warp reduce, reading from shared memory
        // --------

        ALPAKA_UNROLL()
        for(uint32_t currentBlockSize = TBlockSize,
                     currentBlockSizeUp = (TBlockSize + 1) / 2; // ceil(TBlockSize/2.0)
            currentBlockSize > 1;
            currentBlockSize = currentBlockSize / 2,
                     currentBlockSizeUp = (currentBlockSize + 1) / 2) // ceil(currentBlockSize/2.0)
        {
            bool cond = threadIndex < currentBlockSizeUp // only first half of block
                                                         // is working
                        && (threadIndex + currentBlockSizeUp) < TBlockSize // index for second half must be in bounds
                        && (blockIndex * TBlockSize + threadIndex + currentBlockSizeUp) < n
                        && threadIndex < n; // if elem in second half has been initialized before

            if(cond)
                sdata[threadIndex] = func(sdata[threadIndex], sdata[threadIndex + currentBlockSizeUp]);

            alpaka::syncBlockThreads(acc);
        }

        // store block result to gmem
        if(threadIndex == 0 && threadIndex < n)
            destination[blockIndex] = sdata[0];
    }
};
