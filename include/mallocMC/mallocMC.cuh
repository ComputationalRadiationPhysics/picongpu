/*
  mallocMC: Memory Allocator for Many Core Architectures.
  https://www.hzdr.de/crp

  Copyright 2025 Institute of Radiation Physics,
                 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Julian Lenz - j.lenz ( at ) hzdr.de

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

#include "mallocMC/alignmentPolicies/Shrink.hpp"
#include "mallocMC/creationPolicies/FlatterScatter.hpp"
#include "mallocMC/reservePoolPolicies/AlpakaBuf.hpp"

#include <mallocMC/mallocMC.hpp>
#include <sys/types.h>

#include <cstdint>

namespace mallocMC
{
    // This namespace implements an alpaka-agnostic interface by choosing some reasonable defaults working fine for
    // CUDA devices. Further below, we export the necessary names to the global mallocMC:: namespace. See below if
    // you're only interested in usage. Look inside if you want to understand what we've done here or want to port this
    // to other architectures.
    namespace detail
    {
        using Dim = alpaka::DimInt<1>;
        using Idx = std::uint32_t;
        using Acc = alpaka::AccGpuCudaRt<Dim, Idx>;

        // Hide the alpaka-specific Acc argument of `ReservePoolPolicies::AlpakaBuf`.
        using CudaAlpakaBuf = ReservePoolPolicies::AlpakaBuf<Acc>;

        /**
         * @brief Allocator template with hidden alpaka-specifics.
         */
        template<
            typename T_CreationPolicy = CreationPolicies::FlatterScatter<>,
            typename T_DistributionPolicy = DistributionPolicies::Noop,
            typename T_OOMPolicy = OOMPolicies::ReturnNull,
            typename T_ReservePoolPolicy = CudaAlpakaBuf,
            typename T_AlignmentPolicy = AlignmentPolicies::Shrink<>>
        using CudaAllocator = Allocator<
            alpaka::AccToTag<Acc>,
            T_CreationPolicy,
            T_DistributionPolicy,
            T_OOMPolicy,
            T_ReservePoolPolicy,
            T_AlignmentPolicy>;

        /**
         * @brief Host-side infrastructure needed for setting up everything.
         *
         * You need to create an instance of this on the host. It provides the alpaka infrastructure and sets up
         * everything on the device side, so you can get started allocating stuff.
         */
        template<
            typename T_CreationPolicy = CreationPolicies::FlatterScatter<>,
            typename T_DistributionPolicy = DistributionPolicies::Noop,
            typename T_OOMPolicy = OOMPolicies::ReturnNull,
            typename T_ReservePoolPolicy = ReservePoolPolicies::AlpakaBuf<Acc>,
            typename T_AlignmentPolicy = AlignmentPolicies::Shrink<>>
        struct CudaHostInfrastructure
        {
            using MyAllocatorType = CudaAllocator<
                T_CreationPolicy,
                T_DistributionPolicy,
                T_OOMPolicy,
                T_ReservePoolPolicy,
                T_AlignmentPolicy>;

            // Keep this first, so compiler-generated constructors can be called as just
            // CudaHostInfrastructure<>{heapSize};
            size_t heapSize{};

            // All of this is necessary alpaka infrastructure.
            alpaka::Platform<Acc> const platform{};
            std::remove_cv_t<decltype(alpaka::getDevByIdx(platform, 0))> const dev{alpaka::getDevByIdx(platform, 0)};
            alpaka::Queue<Acc, alpaka::NonBlocking> queue{dev};

            // This is our actual host-side instance of the allocator. It sets up everything on the device and provides
            // the handle that we can pass to kernels.
            MyAllocatorType hostInstance{dev, queue, heapSize};
        };

        /**
         * @brief Memory manager to pass to kernels.
         *
         * Create this on the host and pass it to your kernels. It's a lightweight object barely more than a pointer,
         * so you can just copy it around as needed. Its main purpose is to provide an alpaka-agnostic interface by
         * adding an accelerator internally before forwarding malloc/free calls to mallocMC.
         */
        template<
            typename T_CreationPolicy = CreationPolicies::FlatterScatter<>,
            typename T_DistributionPolicy = DistributionPolicies::Noop,
            typename T_OOMPolicy = OOMPolicies::ReturnNull,
            typename T_ReservePoolPolicy = ReservePoolPolicies::AlpakaBuf<Acc>,
            typename T_AlignmentPolicy = AlignmentPolicies::Shrink<>>
        struct CudaMemoryManager
        {
            using MyHostInfrastructure = CudaHostInfrastructure<
                T_CreationPolicy,
                T_DistributionPolicy,
                T_OOMPolicy,
                T_ReservePoolPolicy,
                T_AlignmentPolicy>;

            /**
             * @brief Construct the memory manager from the host infrastructure.
             *
             * @param hostInfrastructure Reference to the host infrastructure.
             */
            explicit CudaMemoryManager(MyHostInfrastructure const& hostInfrastructure)
                : deviceHandle(hostInfrastructure.hostInstance.getAllocatorHandle())
            {
            }

            /**
             * @brief Allocates memory on the device.
             *
             * @param size Size of the memory to allocate.
             * @return Pointer to the allocated memory.
             */
            __device__ __forceinline__ void* malloc(size_t size)
            {
                // This is cheating a tiny little bit. The accelerator could, in general, be a stateful object but
                // concretely for CUDA and HIP it just forwards to the corresponding API calls, so it doesn't actually
                // carry any information by itself. We're rather using it as a tag here.
                std::array<std::byte, sizeof(Acc)> fakeAccMemory{};
                return deviceHandle.malloc(*reinterpret_cast<Acc*>(fakeAccMemory.data()), size);
            }

            /**
             * @brief Frees memory on the device.
             *
             * @param ptr Pointer to the memory to free.
             */
            __device__ __forceinline__ void free(void* ptr)
            {
                std::array<std::byte, sizeof(Acc)> fakeAccMemory{};
                deviceHandle.free(*reinterpret_cast<Acc*>(fakeAccMemory.data()), ptr);
            }

            /**
             * @brief Handle to the device allocator.
             *
             * This is what actually does the work in mallocMC. We forward all our calls to this.
             */
            MyHostInfrastructure::MyAllocatorType::AllocatorHandle deviceHandle;
        };
    } // namespace detail

    // Use the following in your native CUDA code and you are good to go! All alpaka-specific interfaces are patched
    // away.
    using detail::CudaAllocator;
    using detail::CudaHostInfrastructure;
    using detail::CudaMemoryManager;

    namespace ReservePoolPolicies
    {
        // This is provided because the original ReservePoolPolicies::AlpakaBuf takes an alpaka::Acc tag as template
        // argument. In contrast, this is alpaka-agnostic.
        using detail::CudaAlpakaBuf;
    } // namespace ReservePoolPolicies
} // namespace mallocMC
