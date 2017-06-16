/*
  mallocMC: Memory Allocator for Many Core Architectures.
  https://www.hzdr.de/crp

  Copyright 2014 - 2015 Institute of Radiation Physics,
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

#pragma once

#include "mallocMC_utils.hpp"
#include "mallocMC_constraints.hpp"
#include "mallocMC_prefixes.hpp"
#include "mallocMC_traits.hpp"

#include <boost/cstdint.hpp>
#include <stdio.h>

namespace mallocMC{

namespace detail{

    /**
     * @brief Template class to call getAvailableSlots[Host|Accelerator] if the CreationPolicy provides it.
     *
     * Returns 0 else.
     *
     * @tparam T_Allocator The type of the Allocator to be used
     * @tparam T_isHost True for the host call, false for the accelerator call
     * @tparam T_providesAvailableSlots If the CreationPolicy provides getAvailableSlots[Host|Accelerator] (auto filled, do not set)
     */
    template<
        typename T_Allocator,
        bool T_providesAvailableSlots
    >
    struct GetAvailableSlotsIfAvailAcc
    {
        MAMC_ACCELERATOR static
        unsigned
        getAvailableSlots(
            size_t,
            T_Allocator &
        )
        {
            return 0;
        }

    };

    template<
        typename T_Allocator
    >
    struct GetAvailableSlotsIfAvailAcc<
        T_Allocator,
        true
    >{
        MAMC_ACCELERATOR static
        unsigned
        getAvailableSlots(
            size_t slotSize,
            T_Allocator& alloc
        )
        {
            return alloc.T_Allocator::CreationPolicy
                ::getAvailableSlotsAccelerator( slotSize );
        }

    };

} // namespace detail


    /**
     * @brief "HostClass" that combines all policies to a useful allocator
     *
     * This class implements the necessary glue-logic to form an actual allocator
     * from the provided policies. It implements the public interface and
     * executes some constraint checking based on an instance of the class
     * PolicyConstraints.
     *
     * @tparam T_CreationPolicy The desired type of a CreationPolicy
     * @tparam T_DistributionPolicy The desired type of a DistributionPolicy
     * @tparam T_OOMPolicy The desired type of a OOMPolicy
     * @tparam T_ReservePoolPolicy The desired type of a ReservePoolPolicy
     * @tparam T_AlignmentPolicy The desired type of a AlignmentPolicy
     */
    template<
        typename T_CreationPolicy,
        typename T_DistributionPolicy,
        typename T_OOMPolicy,
        typename T_AlignmentPolicy
    >
    class DeviceAllocator :
        public T_CreationPolicy
    {
        typedef boost::uint32_t uint32;
    public:
        typedef T_CreationPolicy CreationPolicy;
        typedef T_DistributionPolicy DistributionPolicy;
        typedef T_OOMPolicy OOMPolicy;
        typedef T_AlignmentPolicy AlignmentPolicy;

        void* pool;

        MAMC_ACCELERATOR
        void*
        malloc(
            size_t bytes
        )
        {
            DistributionPolicy distributionPolicy;
            bytes = AlignmentPolicy::applyPadding( bytes );
            uint32 req_size = distributionPolicy.collect( bytes );
            void* memBlock = CreationPolicy::create( req_size );
            const bool oom = CreationPolicy::isOOM( memBlock, req_size );
            if( oom )
                memBlock = OOMPolicy::handleOOM( memBlock );
            void* myPart = distributionPolicy.distribute( memBlock );
            return myPart;
        }

        MAMC_ACCELERATOR
        void
        free(
            void* p
        )
        {
            CreationPolicy::destroy( p );
        }


        /* polymorphism over the availability of getAvailableSlots for calling
         * from the accelerator
         */
        MAMC_ACCELERATOR
        unsigned
        getAvailableSlots(
            size_t slotSize
        )
        {
            slotSize = AlignmentPolicy::applyPadding( slotSize );
            return detail::GetAvailableSlotsIfAvailAcc<
                DeviceAllocator,
                Traits< DeviceAllocator >::providesAvailableSlots
            >::getAvailableSlots( slotSize, *this );
        }

    };

} // namespace mallocMC
