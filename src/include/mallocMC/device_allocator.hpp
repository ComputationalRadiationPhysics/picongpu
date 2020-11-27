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

#include "mallocMC_constraints.hpp"
#include "mallocMC_traits.hpp"
#include "mallocMC_utils.hpp"

#include <alpaka/core/Common.hpp>
#include <cstdint>
#include <cstdio>

namespace mallocMC
{
    namespace detail
    {
        /**
         * @brief Template class to call getAvailableSlots[Host|Accelerator] if
         * the CreationPolicy provides it.
         *
         * Returns 0 else.
         *
         * @tparam T_Allocator The type of the Allocator to be used
         * @tparam T_isHost True for the host call, false for the accelerator
         * call
         * @tparam T_providesAvailableSlots If the CreationPolicy provides
         * getAvailableSlots[Host|Accelerator] (auto filled, do not set)
         */
        template<typename AlpakaAcc, typename T_Allocator, bool T_providesAvailableSlots>
        struct GetAvailableSlotsIfAvailAcc
        {
            ALPAKA_FN_ACC static auto getAvailableSlots(const AlpakaAcc&, size_t, T_Allocator&) -> unsigned
            {
                return 0;
            }
        };

        template<typename AlpakaAcc, typename T_Allocator>
        struct GetAvailableSlotsIfAvailAcc<AlpakaAcc, T_Allocator, true>
        {
            ALPAKA_FN_ACC static auto getAvailableSlots(const AlpakaAcc& acc, size_t slotSize, T_Allocator& alloc)
                -> unsigned
            {
                return alloc.T_Allocator::CreationPolicy ::getAvailableSlotsAccelerator(acc, slotSize);
            }
        };

    } // namespace detail

    /**
     * @brief "HostClass" that combines all policies to a useful allocator
     *
     * This class implements the necessary glue-logic to form an actual
     * allocator from the provided policies. It implements the public interface
     * and executes some constraint checking based on an instance of the class
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
        typename T_AlignmentPolicy>
    class DeviceAllocator : public T_CreationPolicy
    {
        using uint32 = std::uint32_t;

    public:
        using CreationPolicy = T_CreationPolicy;
        using DistributionPolicy = T_DistributionPolicy;
        using OOMPolicy = T_OOMPolicy;
        using AlignmentPolicy = T_AlignmentPolicy;

        void* pool;

        template<typename AlpakaAcc>
        ALPAKA_FN_ACC auto malloc(const AlpakaAcc& acc, size_t bytes) -> void*
        {
            bytes = AlignmentPolicy::applyPadding(bytes);
            DistributionPolicy distributionPolicy(acc);
            const uint32 req_size = distributionPolicy.collect(acc, bytes);
            void* memBlock = CreationPolicy::create(acc, req_size);
            if(CreationPolicy::isOOM(memBlock, req_size))
                memBlock = OOMPolicy::handleOOM(memBlock);
            return distributionPolicy.distribute(acc, memBlock);
        }

        template<typename AlpakaAcc>
        ALPAKA_FN_ACC void free(const AlpakaAcc& acc, void* p)
        {
            CreationPolicy::destroy(acc, p);
        }

        /* polymorphism over the availability of getAvailableSlots for calling
         * from the accelerator
         */
        template<typename AlpakaAcc>
        ALPAKA_FN_ACC auto getAvailableSlots(const AlpakaAcc& acc, size_t slotSize) -> unsigned
        {
            slotSize = AlignmentPolicy::applyPadding(slotSize);
            return detail::GetAvailableSlotsIfAvailAcc<
                AlpakaAcc,
                DeviceAllocator,
                Traits<DeviceAllocator>::providesAvailableSlots>::getAvailableSlots(acc, slotSize, *this);
        }
    };

} // namespace mallocMC
