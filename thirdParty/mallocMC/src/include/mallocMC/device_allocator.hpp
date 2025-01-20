/*
  mallocMC: Memory Allocator for Many Core Architectures.
  https://www.hzdr.de/crp

  Copyright 2014 - 2024 Institute of Radiation Physics,
                        Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Carlchristian Eckert - c.eckert ( at ) hzdr.de
              Julian J. Lenz - j.lenz ( at ) hzdr.de

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

#include "mallocMC_traits.hpp"

#include <alpaka/core/Common.hpp>

#include <cstdint>
#include <cstdio>

namespace mallocMC
{
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
    class DeviceAllocator : public T_CreationPolicy::template AlignmentAwarePolicy<T_AlignmentPolicy>
    {
        using uint32 = std::uint32_t;

    public:
        using CreationPolicy = T_CreationPolicy;
        using DistributionPolicy = T_DistributionPolicy;
        using OOMPolicy = T_OOMPolicy;
        using AlignmentPolicy = T_AlignmentPolicy;

        template<typename AlpakaAcc>
        ALPAKA_FN_ACC auto malloc(AlpakaAcc const& acc, size_t bytes) -> void*
        {
            if(bytes == 0U)
            {
                return nullptr;
            }
            bytes = AlignmentPolicy::applyPadding(bytes);
            DistributionPolicy distributionPolicy(acc);
            uint32 const req_size = distributionPolicy.collect(acc, bytes);
            void* memBlock = CreationPolicy::template AlignmentAwarePolicy<T_AlignmentPolicy>::create(acc, req_size);
            if(CreationPolicy::isOOM(memBlock, req_size))
            {
                memBlock = OOMPolicy::handleOOM(memBlock);
            }
            return distributionPolicy.distribute(acc, memBlock);
        }

        template<typename AlpakaAcc>
        ALPAKA_FN_ACC void free(AlpakaAcc const& acc, void* pointer)
        {
            if(pointer != nullptr)
            {
                CreationPolicy::template AlignmentAwarePolicy<T_AlignmentPolicy>::destroy(acc, pointer);
            }
        }

        /** Provide the number of available free slots.
         *
         * @tparam AlpakaAcc The type of the Allocator to be used
         * @param acc alpaka accelerator
         * @param slotSize assumed allocation size in bytes
         * @return number of free slots of the given size, if creation policy is not providing the information on the
         * device side 0 will be returned.
         */
        template<typename AlpakaAcc>
        ALPAKA_FN_ACC auto getAvailableSlots(AlpakaAcc const& acc, size_t slotSize) -> unsigned
        {
            slotSize = AlignmentPolicy::applyPadding(slotSize);
            if constexpr(Traits<DeviceAllocator>::providesAvailableSlots)
            {
                return CreationPolicy::template AlignmentAwarePolicy<T_AlignmentPolicy>::getAvailableSlotsAccelerator(
                    acc,
                    slotSize);
            }
            else
            {
                return 0U;
            }
        }
    };

} // namespace mallocMC
