/* Copyright 2022 Benjamin Worpitz, Matthias Werner, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/Traits.hpp"
#include "alpaka/core/Assert.hpp"
#include "alpaka/core/Common.hpp"
#include "alpaka/core/Utility.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/extent/Traits.hpp"
#include "alpaka/kernel/KernelFunctionAttributes.hpp"
#include "alpaka/kernel/Traits.hpp"
#include "alpaka/vec/Vec.hpp"
#include "alpaka/workdiv/WorkDivMembers.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <set>
#include <type_traits>

#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wswitch-default"
#endif

//! The alpaka library.
namespace alpaka
{
    //! The grid block extent subdivision restrictions.
    enum class GridBlockExtentSubDivRestrictions
    {
        EqualExtent, //!< The block thread extent will be equal in all dimensions.
        CloseToEqualExtent, //!< The block thread extent will be as close to equal as possible in all dimensions.
        Unrestricted, //!< The block thread extent will not have any restrictions.
    };

    namespace detail
    {
        //! Finds the largest divisor where divident % divisor == 0
        //! \param dividend The dividend.
        //! \param maxDivisor The maximum divisor.
        //! \return The biggest number that satisfies the following conditions:
        //!     1) dividend%ret==0
        //!     2) ret<=maxDivisor
        template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
        ALPAKA_FN_HOST auto nextDivisorLowerOrEqual(T const& dividend, T const& maxDivisor) -> T
        {
            core::assertValueUnsigned(dividend);
            core::assertValueUnsigned(maxDivisor);
            ALPAKA_ASSERT(dividend >= maxDivisor);

            T divisor = maxDivisor;
            while(dividend % divisor != 0)
                --divisor;
            return divisor;
        }

        //! \param val The value to find divisors of.
        //! \param maxDivisor The maximum.
        //! \return A list of all divisors less then or equal to the given maximum.
        template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
        ALPAKA_FN_HOST auto allDivisorsLessOrEqual(T const& val, T const& maxDivisor) -> std::set<T>
        {
            std::set<T> divisorSet;

            core::assertValueUnsigned(val);
            core::assertValueUnsigned(maxDivisor);
            ALPAKA_ASSERT(maxDivisor <= val);

            for(T i(1); i <= std::min(val, maxDivisor); ++i)
            {
                if(val % i == 0)
                {
                    divisorSet.insert(static_cast<T>(val / i));
                }
            }

            return divisorSet;
        }
    } // namespace detail

    //! \tparam TDim The dimensionality of the accelerator device properties.
    //! \tparam TIdx The idx type of the accelerator device properties.
    //! \param accDevProps The maxima for the work division.
    //! \return If the accelerator device properties are valid.
    template<typename TDim, typename TIdx>
    ALPAKA_FN_HOST auto isValidAccDevProps(AccDevProps<TDim, TIdx> const& accDevProps) -> bool
    {
        // Check that the maximum counts are greater or equal 1.
        if((accDevProps.m_gridBlockCountMax < 1) || (accDevProps.m_blockThreadCountMax < 1)
           || (accDevProps.m_threadElemCountMax < 1))
        {
            return false;
        }

        // Store the maxima allowed for extents of grid, blocks and threads.
        auto const gridBlockExtentMax = subVecEnd<TDim>(accDevProps.m_gridBlockExtentMax);
        auto const blockThreadExtentMax = subVecEnd<TDim>(accDevProps.m_blockThreadExtentMax);
        auto const threadElemExtentMax = subVecEnd<TDim>(accDevProps.m_threadElemExtentMax);

        // Check that the extents for all dimensions are correct.
        for(typename TDim::value_type i(0); i < TDim::value; ++i)
        {
            // Check that the maximum extents are greater or equal 1.
            if((gridBlockExtentMax[i] < 1) || (blockThreadExtentMax[i] < 1) || (threadElemExtentMax[i] < 1))
            {
                return false;
            }
        }

        return true;
    }

    //! Subdivides the given grid thread extent into blocks restricted by the maxima allowed.
    //! 1. The the maxima block, thread and element extent and counts
    //! 2. The requirement of the block thread extent to divide the grid thread extent without remainder
    //! 3. The requirement of the block extent.
    //!
    //! \param gridElemExtent The full extent of elements in the grid.
    //! \param threadElemExtent the number of elements computed per thread.
    //! \param accDevProps The maxima for the work division.
    //! \param kernelBlockThreadCountMax The maximum number of threads per block. If it is zero this argument is not
    //! used, device hard limits are used.
    //! \param blockThreadMustDivideGridThreadExtent If this is true, the grid thread extent will be multiples of the
    //! corresponding block thread extent.
    //!     NOTE: If this is true and gridThreadExtent is prime (or otherwise bad chosen) in a dimension, the block
    //!     thread extent will be one in this dimension.
    //! \param gridBlockExtentSubDivRestrictions The grid block extent subdivision restrictions.
    template<typename TDim, typename TIdx>
    ALPAKA_FN_HOST auto subDivideGridElems(
        Vec<TDim, TIdx> const& gridElemExtent,
        Vec<TDim, TIdx> const& threadElemExtent,
        AccDevProps<TDim, TIdx> const& accDevProps,
        TIdx kernelBlockThreadCountMax = static_cast<TIdx>(0u),
        bool blockThreadMustDivideGridThreadExtent = true,
        GridBlockExtentSubDivRestrictions gridBlockExtentSubDivRestrictions
        = GridBlockExtentSubDivRestrictions::Unrestricted) -> WorkDivMembers<TDim, TIdx>
    {
        using Vec = Vec<TDim, TIdx>;
        using DimLoopInd = typename TDim::value_type;

        for(DimLoopInd i(0); i < TDim::value; ++i)
        {
            ALPAKA_ASSERT(gridElemExtent[i] >= 1);
            ALPAKA_ASSERT(threadElemExtent[i] >= 1);
            ALPAKA_ASSERT(threadElemExtent[i] <= accDevProps.m_threadElemExtentMax[i]);
        }
        ALPAKA_ASSERT(threadElemExtent.prod() <= accDevProps.m_threadElemCountMax);
        ALPAKA_ASSERT(isValidAccDevProps(accDevProps));

        // Handle threadElemExtent and compute gridThreadExtent. Afterwards, only the blockThreadExtent has to be
        // optimized.
        auto clippedThreadElemExtent = elementwise_min(threadElemExtent, gridElemExtent);
        auto const gridThreadExtent = [&]
        {
            Vec r;
            for(DimLoopInd i(0u); i < TDim::value; ++i)
                r[i] = core::divCeil(gridElemExtent[i], clippedThreadElemExtent[i]);
            return r;
        }();

        ///////////////////////////////////////////////////////////////////
        // Try to calculate an optimal blockThreadExtent.

        // Restrict the max block thread extent from the maximum possible to the grid thread extent.
        // This removes dimensions not required in the grid thread extent.
        // This has to be done before the blockThreadCountMax clipping to get the maximum correctly.
        auto blockThreadExtent = elementwise_min(accDevProps.m_blockThreadExtentMax, gridThreadExtent);

        // For equal block thread extent, restrict it to its minimum component.
        // For example (512, 256, 1024) will get (256, 256, 256).
        if(gridBlockExtentSubDivRestrictions == GridBlockExtentSubDivRestrictions::EqualExtent)
            blockThreadExtent = Vec::all(blockThreadExtent.min() != TIdx(0) ? blockThreadExtent.min() : TIdx(1));

        // Choose kernelBlockThreadCountMax if it is not zero. It is less than the accelerator properties.
        TIdx const& blockThreadCountMax
            = (kernelBlockThreadCountMax != 0) ? kernelBlockThreadCountMax : accDevProps.m_blockThreadCountMax;

        // Block thread extent could be {1024,1024,1024} although max threads per block is 1024. Block thread extent
        // shows the max number of threads along each axis, it is not a measure to get max number of threads per block.
        // It must be further limited (clipped above) by the kernel limit along each axis, using device limits is not
        // enough.
        for(typename TDim::value_type i(0); i < TDim::value; ++i)
        {
            blockThreadExtent[i] = std::min(blockThreadExtent[i], blockThreadCountMax);
        }

        // Make the blockThreadExtent product smaller or equal to the accelerator's limit.
        if(blockThreadCountMax == 1)
        {
            blockThreadExtent = Vec::all(core::nthRootFloor(blockThreadCountMax, TIdx{TDim::value}));
        }
        else if(blockThreadExtent.prod() > blockThreadCountMax)
        {
            switch(gridBlockExtentSubDivRestrictions)
            {
            case GridBlockExtentSubDivRestrictions::EqualExtent:
                blockThreadExtent = Vec::all(core::nthRootFloor(blockThreadCountMax, TIdx{TDim::value}));
                break;
            case GridBlockExtentSubDivRestrictions::CloseToEqualExtent:
                // Very primitive clipping. Just halve the largest value until it fits.
                while(blockThreadExtent.prod() > blockThreadCountMax)
                    blockThreadExtent[blockThreadExtent.maxElem()] /= TIdx{2};
                break;
            case GridBlockExtentSubDivRestrictions::Unrestricted:
                // Very primitive clipping. Just halve the smallest value (which is not 1) until it fits.
                while(blockThreadExtent.prod() > blockThreadCountMax)
                {
                    auto const it = std::min_element(
                        blockThreadExtent.begin(),
                        blockThreadExtent.end() - 1, //! \todo why omit the last element?
                        [](TIdx const& a, TIdx const& b)
                        {
                            if(a == TIdx{1})
                                return false;
                            if(b == TIdx{1})
                                return true;
                            return a < b;
                        });
                    *it /= TIdx{2};
                }
                break;
            }
        }


        // Make the block thread extent divide the grid thread extent.
        if(blockThreadMustDivideGridThreadExtent)
        {
            switch(gridBlockExtentSubDivRestrictions)
            {
            case GridBlockExtentSubDivRestrictions::EqualExtent:
                {
                    // For equal size block extent we have to compute the gcd of all grid thread extent that is less
                    // then the current maximal block thread extent. For this we compute the divisors of all grid
                    // thread extent less then the current maximal block thread extent.
                    std::array<std::set<TIdx>, TDim::value> gridThreadExtentDivisors;
                    for(DimLoopInd i(0u); i < TDim::value; ++i)
                    {
                        gridThreadExtentDivisors[i]
                            = detail::allDivisorsLessOrEqual(gridThreadExtent[i], blockThreadExtent[i]);
                    }
                    // The maximal common divisor of all block thread extent is the optimal solution.
                    std::set<TIdx> intersects[2u];
                    for(DimLoopInd i(1u); i < TDim::value; ++i)
                    {
                        intersects[(i - 1u) % 2u] = gridThreadExtentDivisors[0];
                        intersects[(i) % 2u].clear();
                        set_intersection(
                            std::begin(intersects[(i - 1u) % 2u]),
                            std::end(intersects[(i - 1u) % 2u]),
                            std::begin(gridThreadExtentDivisors[i]),
                            std::end(gridThreadExtentDivisors[i]),
                            std::inserter(intersects[i % 2], std::begin(intersects[i % 2u])));
                    }
                    TIdx const maxCommonDivisor = *(--std::end(intersects[(TDim::value - 1) % 2u]));
                    blockThreadExtent = Vec::all(maxCommonDivisor);
                    break;
                }
            case GridBlockExtentSubDivRestrictions::CloseToEqualExtent:
                [[fallthrough]];
            case GridBlockExtentSubDivRestrictions::Unrestricted:
                for(DimLoopInd i(0u); i < TDim::value; ++i)
                {
                    blockThreadExtent[i] = detail::nextDivisorLowerOrEqual(gridThreadExtent[i], blockThreadExtent[i]);
                }
                break;
            }
        }

        // grid blocks extent = grid thread / block thread extent. quotient is rounded up.
        auto gridBlockExtent = [&]
        {
            Vec r;
            for(DimLoopInd i = 0; i < TDim::value; ++i)
                r[i] = core::divCeil(gridThreadExtent[i], blockThreadExtent[i]);
            return r;
        }();


        // Store the maxima allowed for extents of grid, blocks and threads.
        auto const gridBlockExtentMax = subVecEnd<TDim>(accDevProps.m_gridBlockExtentMax);
        auto const blockThreadExtentMax = subVecEnd<TDim>(accDevProps.m_blockThreadExtentMax);
        auto const threadElemExtentMax = subVecEnd<TDim>(accDevProps.m_threadElemExtentMax);

        // Check that the extents for all dimensions are correct.
        for(typename TDim::value_type i(0); i < TDim::value; ++i)
        {
            // Check that the maximum extents are greater or equal 1.
            if(gridBlockExtentMax[i] < gridBlockExtent[i])
            {
                gridBlockExtent[i] = gridBlockExtentMax[i];
            }
            if(blockThreadExtentMax[i] < blockThreadExtent[i])
            {
                blockThreadExtent[i] = blockThreadExtentMax[i];
            }
            if(threadElemExtentMax[i] < threadElemExtent[i])
            {
                clippedThreadElemExtent[i] = threadElemExtentMax[i];
            }
        }

        return WorkDivMembers<TDim, TIdx>(gridBlockExtent, blockThreadExtent, clippedThreadElemExtent);
    }

    //! Kernel start configuration to determine a valid work division
    //!
    //! \tparam TGridElemExtent The type of the grid element extent.
    //! \tparam TThreadElemExtent The type of the thread element extent.
    template<
        typename TAcc,
        typename TGridElemExtent = alpaka::Vec<Dim<TAcc>, Idx<TAcc>>,
        typename TThreadElemExtent = alpaka::Vec<Dim<TAcc>, Idx<TAcc>>>
    struct KernelCfg
    {
        //! The full extent of elements in the grid.
        TGridElemExtent const gridElemExtent = alpaka::Vec<Dim<TAcc>, Idx<TAcc>>::ones();
        //! The number of elements computed per thread.
        TThreadElemExtent const threadElemExtent = alpaka::Vec<Dim<TAcc>, Idx<TAcc>>::ones();
        //! If this is true, the grid thread extent will be multiples of
        //! the corresponding block thread extent.
        //!     NOTE: If this is true and gridThreadExtent is prime (or otherwise bad chosen) in a dimension, the block
        //!     thread extent will be one in this dimension.
        bool blockThreadMustDivideGridThreadExtent = true;
        //! The grid block extent subdivision restrictions.
        GridBlockExtentSubDivRestrictions gridBlockExtentSubDivRestrictions
            = GridBlockExtentSubDivRestrictions::Unrestricted;

        static_assert(
            Dim<TGridElemExtent>::value == Dim<TAcc>::value,
            "The dimension of Acc and the dimension of TGridElemExtent have to be identical!");
        static_assert(
            Dim<TGridElemExtent>::value == Dim<TAcc>::value,
            "The dimension of Acc and the dimension of TThreadElemExtent have to be identical!");
        static_assert(
            std::is_same_v<Idx<TGridElemExtent>, Idx<TAcc>>,
            "The idx type of Acc and the idx type of TGridElemExtent have to be identical!");
        static_assert(
            std::is_same_v<Idx<TThreadElemExtent>, Idx<TAcc>>,
            "The idx type of Acc and the idx type of TThreadElemExtent have to be identical!");
    };

    //! \tparam TDev The type of the device.
    //! \tparam TGridElemExtent The type of the grid element extent.
    //! \tparam TThreadElemExtent The type of the thread element extent.
    //! \param dev The device the work division should be valid for.
    //! \param kernelFnObj The kernel function object which should be executed.
    //! \param args The kernel invocation arguments.
    //! \return The work division for the accelerator based on the kernel and argument types
    template<
        typename TAcc,
        typename TDev,
        typename TGridElemExtent,
        typename TThreadElemExtent,
        typename TKernelFnObj,
        typename... TArgs>
    ALPAKA_FN_HOST auto getValidWorkDiv(
        KernelCfg<TAcc, TGridElemExtent, TThreadElemExtent> const& kernelCfg,
        [[maybe_unused]] TDev const& dev,
        TKernelFnObj const& kernelFnObj,
        TArgs&&... args) -> WorkDivMembers<Dim<TAcc>, Idx<TAcc>>
    {
        using Acc = TAcc;

        // Get max number of threads per block depending on the kernel function attributes.
        // For GPU backend; number of registers used by the kernel, local and shared memory usage of the kernel
        // determines the max number of threads per block. This number could be equal or less than the max number of
        // threads per block defined by device properties.
        auto const kernelFunctionAttributes
            = getFunctionAttributes<Acc>(dev, kernelFnObj, std::forward<TArgs>(args)...);
        auto const threadsPerBlock = kernelFunctionAttributes.maxThreadsPerBlock;

        if constexpr(Dim<TGridElemExtent>::value == 0)
        {
            auto const zero = Vec<DimInt<0>, Idx<Acc>>{};
            ALPAKA_ASSERT(kernelCfg.gridElemExtent == zero);
            ALPAKA_ASSERT(kernelCfg.threadElemExtent == zero);
            return WorkDivMembers<DimInt<0>, Idx<Acc>>{zero, zero, zero};
        }
        else
            return subDivideGridElems(
                getExtents(kernelCfg.gridElemExtent),
                getExtents(kernelCfg.threadElemExtent),
                getAccDevProps<Acc>(dev),
                static_cast<Idx<Acc>>(threadsPerBlock),
                kernelCfg.blockThreadMustDivideGridThreadExtent,
                kernelCfg.gridBlockExtentSubDivRestrictions);

        using V [[maybe_unused]] = Vec<Dim<TGridElemExtent>, Idx<TGridElemExtent>>;
        ALPAKA_UNREACHABLE(WorkDivMembers<Dim<TGridElemExtent>, Idx<TGridElemExtent>>{V{}, V{}, V{}});
    }

    //! Checks if the work division is supported
    //!
    //! \tparam TWorkDiv The type of the work division.
    //! \tparam TDim The dimensionality of the accelerator device properties.
    //! \tparam TIdx The idx type of the accelerator device properties.
    //! \param workDiv The work division to test for validity.
    //! \param accDevProps The maxima for the work division.
    //! \return If the work division is valid for the given accelerator device properties.
    template<typename TWorkDiv, typename TDim, typename TIdx>
    ALPAKA_FN_HOST auto isValidWorkDiv(TWorkDiv const& workDiv, AccDevProps<TDim, TIdx> const& accDevProps) -> bool
    {
        // Get the extents of grid, blocks and threads of the work division to check.
        auto const gridBlockExtent = getWorkDiv<Grid, Blocks>(workDiv);
        auto const blockThreadExtent = getWorkDiv<Block, Threads>(workDiv);
        auto const threadElemExtent = getWorkDiv<Thread, Elems>(workDiv);

        // Check that the maximal counts are satisfied.
        if(accDevProps.m_gridBlockCountMax < gridBlockExtent.prod())
        {
            return false;
        }
        if(accDevProps.m_blockThreadCountMax < blockThreadExtent.prod())
        {
            return false;
        }
        if(accDevProps.m_threadElemCountMax < threadElemExtent.prod())
        {
            return false;
        }

        // Check that the extents for all dimensions are correct.
        if constexpr(Dim<TWorkDiv>::value > 0)
        {
            // Store the maxima allowed for extents of grid, blocks and threads.
            auto const gridBlockExtentMax = subVecEnd<Dim<TWorkDiv>>(accDevProps.m_gridBlockExtentMax);
            auto const blockThreadExtentMax = subVecEnd<Dim<TWorkDiv>>(accDevProps.m_blockThreadExtentMax);
            auto const threadElemExtentMax = subVecEnd<Dim<TWorkDiv>>(accDevProps.m_threadElemExtentMax);

            for(typename Dim<TWorkDiv>::value_type i(0); i < Dim<TWorkDiv>::value; ++i)
            {
                // No extent is allowed to be zero or greater then the allowed maximum.
                if((gridBlockExtent[i] < 1) || (blockThreadExtent[i] < 1) || (threadElemExtent[i] < 1)
                   || (gridBlockExtentMax[i] < gridBlockExtent[i]) || (blockThreadExtentMax[i] < blockThreadExtent[i])
                   || (threadElemExtentMax[i] < threadElemExtent[i]))
                {
                    return false;
                }
            }
        }

        return true;
    }

    //! Checks if the work division is supported
    //!
    //! \tparam TWorkDiv The type of the work division.
    //! \tparam TDim The dimensionality of the accelerator device properties.
    //! \tparam TIdx The idx type of the accelerator device properties.
    //! \param workDiv The work division to test for validity.
    //! \param accDevProps The maxima for the work division.
    //! \param kernelFunctionAttributes Kernel attributes, including the maximum number of threads per block that can
    //! be used by this kernel on the given device. This number can be equal to or smaller than the the number of
    //! threads per block supported by the device.
    //! \return Returns true if the work division is valid for the given accelerator device properties and for the
    //! given kernel. Otherwise returns false.
    template<typename TAcc, typename TWorkDiv, typename TDim, typename TIdx>
    ALPAKA_FN_HOST auto isValidWorkDiv(
        TWorkDiv const& workDiv,
        AccDevProps<TDim, TIdx> const& accDevProps,
        KernelFunctionAttributes const& kernelFunctionAttributes) -> bool
    {
        // Get the extents of grid, blocks and threads of the work division to check.
        auto const gridBlockExtent = getWorkDiv<Grid, Blocks>(workDiv);
        auto const blockThreadExtent = getWorkDiv<Block, Threads>(workDiv);
        auto const threadElemExtent = getWorkDiv<Thread, Elems>(workDiv);
        // Use kernel properties to find the max threads per block for the kernel
        auto const threadsPerBlockForKernel = kernelFunctionAttributes.maxThreadsPerBlock;
        // Select the minimum to find the upper bound for the threads per block
        auto const allowedThreadsPerBlock = std::min(
            static_cast<TIdx>(threadsPerBlockForKernel),
            static_cast<TIdx>(accDevProps.m_blockThreadCountMax));
        // Check that the maximal counts are satisfied.
        if(accDevProps.m_gridBlockCountMax < gridBlockExtent.prod())
        {
            return false;
        }
        if(allowedThreadsPerBlock < blockThreadExtent.prod())
        {
            return false;
        }
        if(accDevProps.m_threadElemCountMax < threadElemExtent.prod())
        {
            return false;
        }

        // Check that the extents for all dimensions are correct.
        if constexpr(Dim<TWorkDiv>::value > 0)
        {
            // Store the maxima allowed for extents of grid, blocks and threads.
            auto const gridBlockExtentMax = subVecEnd<Dim<TWorkDiv>>(accDevProps.m_gridBlockExtentMax);
            auto const blockThreadExtentMax = subVecEnd<Dim<TWorkDiv>>(accDevProps.m_blockThreadExtentMax);
            auto const threadElemExtentMax = subVecEnd<Dim<TWorkDiv>>(accDevProps.m_threadElemExtentMax);

            for(typename Dim<TWorkDiv>::value_type i(0); i < Dim<TWorkDiv>::value; ++i)
            {
                // No extent is allowed to be zero or greater then the allowed maximum.
                if((gridBlockExtent[i] < 1) || (blockThreadExtent[i] < 1) || (threadElemExtent[i] < 1)
                   || (gridBlockExtentMax[i] < gridBlockExtent[i]) || (blockThreadExtentMax[i] < blockThreadExtent[i])
                   || (threadElemExtentMax[i] < threadElemExtent[i]))
                {
                    return false;
                }
            }
        }

        return true;
    }

    //! Checks if the work division is supported for the kernel on the device
    //!
    //! \tparam TAcc The accelerator to test the validity on.
    //! \tparam TDev The type of the device.
    //! \tparam TWorkDiv The type of work division to test for validity.
    //! \param workDiv The work division to test for validity.
    //! \param dev The device to test the work division for validity on.
    //! \param kernelFnObj The kernel function object which should be executed.
    //! \param args The kernel invocation arguments.
    //! \return Returns the value of isValidWorkDiv function.
    template<typename TAcc, typename TWorkDiv, typename TDev, typename TKernelFnObj, typename... TArgs>
    ALPAKA_FN_HOST auto isValidWorkDiv(
        TWorkDiv const& workDiv,
        TDev const& dev,
        TKernelFnObj const& kernelFnObj,
        TArgs&&... args) -> bool
    {
        return isValidWorkDiv<TAcc>(
            workDiv,
            getAccDevProps<TAcc>(dev),
            getFunctionAttributes<TAcc>(dev, kernelFnObj, std::forward<TArgs>(args)...));
    }

    //! Checks if the work division is supported by the device
    //!
    //! \tparam TAcc The accelerator to test the validity on.
    //! \param workDiv The work division to test for validity.
    //! \param dev The device to test the work division for validity on.
    //! \return If the work division is valid on this accelerator.
    template<typename TAcc, typename TWorkDiv, typename TDev>
    ALPAKA_FN_HOST auto isValidWorkDiv(TWorkDiv const& workDiv, TDev const& dev) -> bool
    {
        return isValidWorkDiv(workDiv, getAccDevProps<TAcc>(dev));
    }
} // namespace alpaka

#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
