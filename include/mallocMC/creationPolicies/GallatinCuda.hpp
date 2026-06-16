/*
  mallocMC: Memory Allocator for Many Core Architectures.

  Copyright 2014-2024 Institute of Radiation Physics,
                 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Carlchristian Eckert - c.eckert ( at ) hzdr.de
              Julian Lenz - j.lenz ( at ) hzdr.de

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

#include <alpaka/alpaka.hpp>

#ifdef mallocMC_HAS_Gallatin_AVAILABLE
#    include <gallatin/allocators/gallatin.cuh>
#else

// Construct a fake, so we get a nice error message when we try to use is
// and it's not in the way when we don't.
namespace gallatin::allocators
{
    template<size_t...>
    struct Gallatin
    {
        static auto generate_on_device(auto...)
        {
            return nullptr;
        }

        template<typename... T>
        auto malloc(T... /*unused*/) -> void*
        {
            // This always triggers but it depends on the template parameter, so it's only instantiated if we actually
            // use it.
            static_assert(sizeof...(T) < 0, "Attempt to use malloc of unavailable gallatin prototype.");
            return nullptr;
        }

        template<typename... T>
        auto free(T... /*unused*/)
        {
            // This always triggers but it depends on the template parameter, so it's only instantiated if we actually
            // use it.
            static_assert(sizeof...(T) < 0, "Attempt to use free of unavailable gallatin prototype.");
        }
    };
} // namespace gallatin::allocators

#endif

namespace mallocMC
{
    namespace CreationPolicies
    {
        /**
         * @brief Prototype integration of Gallatin (https://dl.acm.org/doi/10.1145/3627535.3638499)
         *
         * This CreationPolicy integrates the CUDA code for the Gallatin prototype into mallocMC
         * as a thin wrapper. Its intended for proof-of-principle tests and benchmarks only and
         * obviously only works with on CUDA devices.
         *
         * It also only works with the reservePoolPolicies::Noop beccause it does what CudaSetLimits
         * does internally on its own.
         *
         * If we should ever see the need for it, we'd re-implement it in alpaka for a fully-fletched
         * and well-maintained version of this.
         * Experience has been mixed so far: While we could reproduce good performance in some cases,
         * fragmentation was found to be unusably high (to the point of single-digit utilisaton of
         * available memory) in PIConGPU. That's why there's currently no plan to lift the prototype
         * status in the near future.
         */
        template<
            typename T_AlignmentPolicy,
            size_t bytes_per_segment = 16ULL * 1024 * 1024,
            size_t smallest_slice = 16,
            size_t largest_slice = 4096>
        class GallatinCudaImpl
        {
            using Gallatin = gallatin::allocators::Gallatin<bytes_per_segment, smallest_slice, largest_slice>;

        public:
            template<typename T_AlignmentPolicyLocal>
            using AlignmentAwarePolicy
                = GallatinCudaImpl<T_AlignmentPolicyLocal, bytes_per_segment, smallest_slice, largest_slice>;
            Gallatin* heap{nullptr};

            static constexpr auto providesAvailableSlots = false;

            template<typename AlpakaAcc>
            ALPAKA_FN_ACC auto create(AlpakaAcc const& /*acc*/, uint32_t bytes) const -> void*
            {
                return heap->malloc(static_cast<size_t>(bytes));
            }

            template<typename AlpakaAcc>
            ALPAKA_FN_ACC void destroy(AlpakaAcc const& /*acc*/, void* mem) const
            {
                heap->free(mem);
            }

            ALPAKA_FN_ACC auto isOOM(void* p, size_t s) const -> bool
            {
                return s != 0 && (p == nullptr);
            }

            template<typename AlpakaAcc, typename AlpakaDevice, typename AlpakaQueue, typename T_DeviceAllocator>
            static void initHeap(
                AlpakaDevice& /*dev*/,
                AlpakaQueue& queue,
                T_DeviceAllocator* devAllocator,
                void*,
                size_t memsize)
            {
                static_assert(
                    std::is_same_v<alpaka::AccToTag<AlpakaAcc>, alpaka::TagGpuCudaRt>,
                    "The GallatinCuda creation policy is only available on CUDA architectures. Please choose a "
                    "different one.");

                // This is an extremely hot fix:
                // PIConGPU initialises its allocator with 0 bytes to be able to distribute the pointer.
                // Only afterwards it can find out its actual memory requirements and uses destructiveResize to set
                // the correct heap size. Gallatin runs into issues with this approach.
                // Instead, we simply don't believe the request if it's 0.
                if(memsize == 0)
                    return;

                auto devHost = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);
                using Dim = typename alpaka::trait::DimType<AlpakaAcc>::type;
                using Idx = typename alpaka::trait::IdxType<AlpakaAcc>::type;
                using VecType = alpaka::Vec<Dim, Idx>;

                auto tmp = Gallatin::generate_on_device(memsize, 42, true);
                auto workDivSingleThread
                    = alpaka::WorkDivMembers<Dim, Idx>{VecType::ones(), VecType::ones(), VecType::ones()};
                alpaka::exec<AlpakaAcc>(
                    queue,
                    workDivSingleThread,
                    [tmp, devAllocator] ALPAKA_FN_ACC(AlpakaAcc const&) { devAllocator->heap = tmp; });
            }

            static auto classname() -> std::string
            {
                return "GallatinCuda";
            }
        };

        template<
            size_t bytes_per_segment = 16ULL * 1024 * 1024,
            size_t smallest_slice = 16,
            size_t largest_slice = 4096>
        struct GallatinCuda
        {
            template<typename T_AlignmentPolicy>
            using AlignmentAwarePolicy
                = GallatinCudaImpl<T_AlignmentPolicy, bytes_per_segment, smallest_slice, largest_slice>;
        };

    } // namespace CreationPolicies
} // namespace mallocMC
