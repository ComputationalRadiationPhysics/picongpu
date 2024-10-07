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

#include "device_allocator.hpp"
#include "mallocMC_allocator_handle.hpp"
#include "mallocMC_constraints.hpp"
#include "mallocMC_traits.hpp"
#include "mallocMC_utils.hpp"

#include <alpaka/alpaka.hpp>
#include <cstdint>
#include <memory>
#include <sstream>
#include <tuple>
#include <vector>

namespace mallocMC
{
    namespace detail
    {
        template<typename T_Allocator, bool T_providesAvailableSlots>
        struct GetAvailableSlotsIfAvailHost
        {
            template<typename AlpakaAcc, typename AlpakaDevice, typename AlpakaQueue>
            ALPAKA_FN_HOST static auto getAvailableSlots(AlpakaDevice&, AlpakaQueue&, size_t, T_Allocator&) -> unsigned
            {
                return 0;
            }
        };

        template<class T_Allocator>
        struct GetAvailableSlotsIfAvailHost<T_Allocator, true>
        {
            template<typename AlpakaAcc, typename AlpakaDevice, typename AlpakaQueue>
            ALPAKA_FN_HOST static auto getAvailableSlots(
                AlpakaDevice& dev,
                AlpakaQueue& queue,
                size_t slotSize,
                T_Allocator& alloc) -> unsigned
            {
                return T_Allocator::CreationPolicy::template getAvailableSlotsHost<AlpakaAcc>(
                    dev,
                    queue,
                    slotSize,
                    alloc.getAllocatorHandle().devAllocator);
            }
        };
    } // namespace detail

    struct HeapInfo
    {
        void* p;
        size_t size;
    };

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
        typename AlpakaAcc,
        typename T_CreationPolicy,
        typename T_DistributionPolicy,
        typename T_OOMPolicy,
        typename T_ReservePoolPolicy,
        typename T_AlignmentPolicy>
    class Allocator
        : public PolicyConstraints<
              T_CreationPolicy,
              T_DistributionPolicy,
              T_OOMPolicy,
              T_ReservePoolPolicy,
              T_AlignmentPolicy>
    {
        using uint32 = std::uint32_t;

    public:
        using CreationPolicy = T_CreationPolicy;
        using DistributionPolicy = T_DistributionPolicy;
        using OOMPolicy = T_OOMPolicy;
        using ReservePoolPolicy = T_ReservePoolPolicy;
        using AlignmentPolicy = T_AlignmentPolicy;
        using HeapInfoVector = std::vector<HeapInfo>;
        using DevAllocator = DeviceAllocator<CreationPolicy, DistributionPolicy, OOMPolicy, AlignmentPolicy>;
        using AllocatorHandle = AllocatorHandleImpl<Allocator>;

    private:
        ReservePoolPolicy reservePolicy;
        using DevAllocatorStorageBufferType
            = alpaka::Buf<alpaka::Dev<AlpakaAcc>, DevAllocator, alpaka::DimInt<1>, int>;
        std::unique_ptr<DevAllocatorStorageBufferType>
            devAllocatorBuffer; // FIXME(bgruber): replace by std::optional<>
        HeapInfo heapInfos;

        /** allocate heap memory
         *
         * @param size number of bytes
         */
        template<typename AlpakaDevice, typename AlpakaQueue>
        ALPAKA_FN_HOST void
        /* `volatile size_t size` is required to break clang optimizations which
         * results into runtime errors. Observed in PIConGPU if size is known at
         * compile time. The volatile workaround has no negative effects on the
         * register usage in CUDA.
         */
        alloc(AlpakaDevice& dev, AlpakaQueue& queue, volatile size_t size)
        {
            void* pool = reservePolicy.setMemPool(dev, size);
            std::tie(pool, size) = AlignmentPolicy::alignPool(pool, size);

            devAllocatorBuffer
                = std::make_unique<DevAllocatorStorageBufferType>(alpaka::allocBuf<DevAllocator, int>(dev, 1));
            CreationPolicy::template initHeap<AlpakaAcc>(
                dev,
                queue,
                alpaka::getPtrNative(*devAllocatorBuffer),
                pool,
                size);

            heapInfos.p = pool;
            heapInfos.size = size;
        }

        /** free all data structures
         *
         * Free all allocated memory.
         * After this call the instance is an in invalid state
         */
        ALPAKA_FN_HOST void free()
        {
            devAllocatorBuffer = {};
            reservePolicy.resetMemPool();
            heapInfos.size = 0;
            heapInfos.p = nullptr;
        }

        /* forbid to copy the allocator */
        ALPAKA_FN_HOST
        Allocator(const Allocator&) = delete;

    public:
        template<typename AlpakaDevice, typename AlpakaQueue>
        ALPAKA_FN_HOST Allocator(AlpakaDevice& dev, AlpakaQueue& queue, size_t size = 8U * 1024U * 1024U)
        {
            alloc(dev, queue, size);
        }

        ALPAKA_FN_HOST
        ~Allocator()
        {
            free();
        }

        /** destroy current heap data and resize the heap
         *
         * @param size number of bytes
         */
        template<typename AlpakaDevice, typename AlpakaQueue>
        ALPAKA_FN_HOST void destructiveResize(AlpakaDevice& dev, AlpakaQueue& queue, size_t size)
        {
            free();
            alloc(dev, queue, size);
        }

        ALPAKA_FN_HOST
        auto getAllocatorHandle() -> AllocatorHandle
        {
            return AllocatorHandle{alpaka::getPtrNative(*devAllocatorBuffer)};
        }

        ALPAKA_FN_HOST
        operator AllocatorHandle()
        {
            return getAllocatorHandle();
        }

        ALPAKA_FN_HOST static auto info(std::string linebreak = " ") -> std::string
        {
            std::stringstream ss;
            ss << "CreationPolicy:      " << CreationPolicy::classname() << "    " << linebreak;
            ss << "DistributionPolicy:  " << DistributionPolicy::classname() << "" << linebreak;
            ss << "OOMPolicy:           " << OOMPolicy::classname() << "         " << linebreak;
            ss << "ReservePoolPolicy:   " << ReservePoolPolicy::classname() << " " << linebreak;
            ss << "AlignmentPolicy:     " << AlignmentPolicy::classname() << "   " << linebreak;
            return ss.str();
        }

        // polymorphism over the availability of getAvailableSlots for calling
        // from the host
        template<typename AlpakaDevice, typename AlpakaQueue>
        ALPAKA_FN_HOST auto getAvailableSlots(AlpakaDevice& dev, AlpakaQueue& queue, size_t slotSize) -> unsigned
        {
            slotSize = AlignmentPolicy::applyPadding(slotSize);
            return detail::GetAvailableSlotsIfAvailHost<Allocator, Traits<Allocator>::providesAvailableSlots>::
                template getAvailableSlots<AlpakaAcc>(dev, queue, slotSize, *this);
        }

        ALPAKA_FN_HOST
        auto getHeapLocations() -> HeapInfoVector
        {
            HeapInfoVector v;
            v.push_back(heapInfos);
            return v;
        }
    };

} // namespace mallocMC
