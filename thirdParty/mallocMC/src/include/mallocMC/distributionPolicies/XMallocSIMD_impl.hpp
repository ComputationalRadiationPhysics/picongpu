/*
  mallocMC: Memory Allocator for Many Core Architectures.
  http://www.icg.tugraz.at/project/mvp

  Copyright (C) 2012 Institute for Computer Graphics and Vision,
                     Graz University of Technology
  Copyright (C) 2014 Institute of Radiation Physics,
                     Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Markus Steinberger - steinberger ( at ) icg.tugraz.at
              Rene Widera - r.widera ( at ) hzdr.de
              Axel Huebl - a.huebl ( at ) hzdr.de
              Carlchristian Eckert - c.eckert ( at ) hzdr.de

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

#include <boost/cstdint.hpp>
#include <boost/static_assert.hpp>
#include <limits>
#include <string>
#include <sstream>

#include "../mallocMC_utils.hpp"
#include "../mallocMC_prefixes.hpp"
#include "XMallocSIMD.hpp"

namespace mallocMC{
namespace DistributionPolicies{

  template<class T_Config>
  class XMallocSIMD
  {
    private:

      typedef boost::uint32_t uint32;
      bool can_use_coalescing;
      uint32 warpid;
      uint32 myoffset;
      uint32 threadcount;
      uint32 req_size;
    public:
      typedef T_Config Properties;

      MAMC_ACCELERATOR
      XMallocSIMD() : can_use_coalescing(false), warpid(warpid_withinblock()),
        myoffset(0), threadcount(0), req_size(0)
      {}

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
#define MALLOCMC_DP_XMALLOCSIMD_PAGESIZE Properties::pagesize::value
#endif
      BOOST_STATIC_CONSTEXPR uint32 pagesize      = MALLOCMC_DP_XMALLOCSIMD_PAGESIZE;

      //all the properties must be unsigned integers > 0
      BOOST_STATIC_ASSERT(!std::numeric_limits<typename Properties::pagesize::type>::is_signed);

      // \TODO: The static_cast can be removed once the minimal dependencies of
      //        this project is are at least CUDA 7.0 and gcc 4.8.2
      BOOST_STATIC_ASSERT(static_cast<uint32>(pagesize) > 0);

    public:
      BOOST_STATIC_CONSTEXPR uint32 _pagesize = pagesize;

      MAMC_ACCELERATOR
      uint32 collect(uint32 bytes){

        can_use_coalescing = false;
        myoffset = 0;
        threadcount = 0;

        //init with initial counter
        __shared__ uint32 warp_sizecounter[MaxThreadsPerBlock::value / WarpSize::value];
        warp_sizecounter[warpid] = 16;

        //second half: make sure that all coalesced allocations can fit within one page
        //necessary for offset calculation
        bool coalescible = bytes > 0 && bytes < (pagesize / 32);
#if(__CUDACC_VER_MAJOR__ >= 9)
        threadcount = __popc(__ballot_sync(__activemask(), coalescible));
#else
        threadcount = __popc(__ballot(coalescible));
#endif
        if (coalescible && threadcount > 1)
        {
          myoffset = atomicAdd(&warp_sizecounter[warpid], bytes);
          can_use_coalescing = true;
        }

        req_size = bytes;
        if (can_use_coalescing)
          req_size = (myoffset == 16) ? warp_sizecounter[warpid] : 0;

        return req_size;
      }


      MAMC_ACCELERATOR
      void* distribute(void* allocatedMem){
        __shared__ char* warp_res[MaxThreadsPerBlock::value / WarpSize::value];

        char* myalloc = (char*) allocatedMem;
        if (req_size && can_use_coalescing)
        {
          warp_res[warpid] = myalloc;
          if (myalloc != 0)
            *(uint32*)myalloc = threadcount;
        }
        __threadfence_block();

        void *myres = myalloc;
        if(can_use_coalescing)
        {
          if(warp_res[warpid] != 0)
            myres = warp_res[warpid] + myoffset;
          else
            myres = 0;
        }
        return myres;
      }

      MAMC_HOST
      static std::string classname(){
        std::stringstream ss;
        ss << "XMallocSIMD[" << pagesize << "]";
        return ss.str();
      }

  };

} //namespace DistributionPolicies

} //namespace mallocMC
