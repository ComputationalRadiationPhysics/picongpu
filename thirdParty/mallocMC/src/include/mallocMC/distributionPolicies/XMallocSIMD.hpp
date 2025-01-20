/*
  mallocMC: Memory Allocator for Many Core Architectures.
  http://www.icg.tugraz.at/project/mvp

  Copyright (C) 2012 Institute for Computer Graphics and Vision,
                     Graz University of Technology
  Copyright (C) 2014-2024 Institute of Radiation Physics,
                     Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Markus Steinberger - steinberger ( at ) icg.tugraz.at
              Rene Widera - r.widera ( at ) hzdr.de
              Axel Huebl - a.huebl ( at ) hzdr.de
              Carlchristian Eckert - c.eckert ( at ) hzdr.de
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

#include "../mallocMC_utils.hpp"
#include "XMallocSIMD.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/warp/Traits.hpp>

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>

namespace mallocMC
{
    namespace DistributionPolicies
    {
        namespace XMallocSIMDConf
        {
            struct DefaultXMallocConfig
            {
                static constexpr auto pagesize = 4096;
            };
        } // namespace XMallocSIMDConf

        /**
         * @brief SIMD optimized chunk resizing in the style of XMalloc
         *
         * This DistributionPolicy can take the memory requests from a group of
         * worker threads and combine them, so that only one of the workers will
         * allocate the whole request. Later, each worker gets an appropriate
         * offset into the allocated chunk. This is beneficial for SIMD
         * architectures since only one of the workers has to compete for the
         * resource.  This algorithm is inspired by the XMalloc memory allocator
         * (http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=5577907&tag=1)
         * and its implementation in ScatterAlloc
         * (http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6339604)
         * XMallocSIMD is inteded to be used with Nvidia CUDA capable
         * accelerators that support at least compute capability 2.0
         *
         * @tparam T_Config (optional) The configuration struct to overwrite
         *        default configuration. The default can be obtained through
         *        XMallocSIMD<>::Properties
         */
        template<typename T_Config = XMallocSIMDConf::DefaultXMallocConfig>
        class XMallocSIMD
        {
        private:
            using uint32 = std::uint32_t;
            bool can_use_coalescing;
            uint32 warpid;
            uint32 myoffset;
            uint32 threadcount;
            uint32 req_size;

        public:
            using Properties = T_Config;

            template<typename AlpakaAcc>
            ALPAKA_FN_ACC XMallocSIMD(AlpakaAcc const& acc)
                : can_use_coalescing(false)
                , warpid(warpid_withinblock(acc))
                , myoffset(0)
                , threadcount(0)
                , req_size(0)
            {
            }

        private:
/** Allow for a hierarchical validation of parameters:
 *
 * shipped default-parameters (in the inherited struct) have lowest precedence.
 * They will be overridden by a given configuration struct. However, even the
 * given configuration struct can be overridden by compile-time command line
 * parameters (e.g. -D MALLOCMC_DP_XMALLOCSIMD_PAGESIZE 1024)
 *
 * default-struct < template-struct < command-line parameter
 */
#ifndef MALLOCMC_DP_XMALLOCSIMD_PAGESIZE
#    define MALLOCMC_DP_XMALLOCSIMD_PAGESIZE (Properties::pagesize)
#endif
            static constexpr uint32 pagesize = MALLOCMC_DP_XMALLOCSIMD_PAGESIZE;

        public:
            static constexpr uint32 _pagesize = pagesize;

            template<typename AlpakaAcc>
            ALPAKA_FN_ACC auto collect(AlpakaAcc const& acc, uint32 bytes) -> uint32
            {
                can_use_coalescing = false;
                myoffset = 0;
                threadcount = 0;

                // init with initial counter
                auto& warp_sizecounter
                    = alpaka::declareSharedVar<std::uint32_t[maxThreadsPerBlock / warpSize<AlpakaAcc>()], __COUNTER__>(
                        acc);
                warp_sizecounter[warpid] = 16;

                // second half: make sure that all coalesced allocations can fit
                // within one page necessary for offset calculation
                bool const coalescible = bytes > 0 && bytes < (pagesize / 32);

#if(MALLOCMC_DEVICE_COMPILE)
                threadcount = alpaka::popcount(alpaka::warp::ballot(acc, coalescible));
#else
                threadcount = 1; // TODO
#endif
                if(coalescible && threadcount > 1)
                {
                    myoffset = alpaka::atomicOp<alpaka::AtomicAdd>(acc, &warp_sizecounter[warpid], bytes);
                    can_use_coalescing = true;
                }

                req_size = bytes;
                if(can_use_coalescing)
                    req_size = (myoffset == 16) ? warp_sizecounter[warpid] : 0;

                return req_size;
            }

            template<typename AlpakaAcc>
            ALPAKA_FN_ACC auto distribute(AlpakaAcc const& acc, void* allocatedMem) -> void*
            {
                auto& warp_res
                    = alpaka::declareSharedVar<char * [maxThreadsPerBlock / warpSize<AlpakaAcc>()], __COUNTER__>(acc);

                char* myalloc = (char*) allocatedMem;
                if(req_size && can_use_coalescing)
                {
                    warp_res[warpid] = myalloc;
                    if(myalloc != 0)
                        *(uint32*) myalloc = threadcount;
                }

                threadfenceBlock(acc);

                void* myres = myalloc;
                if(can_use_coalescing)
                {
                    if(warp_res[warpid] != 0)
                        myres = warp_res[warpid] + myoffset;
                    else
                        myres = 0;
                }
                return myres;
            }

            ALPAKA_FN_HOST
            static auto classname() -> std::string
            {
                std::stringstream ss;
                ss << "XMallocSIMD[" << pagesize << "]";
                return ss.str();
            }
        };

    } // namespace DistributionPolicies

} // namespace mallocMC
