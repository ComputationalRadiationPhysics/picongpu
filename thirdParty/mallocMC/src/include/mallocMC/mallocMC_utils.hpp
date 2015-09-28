/*
  mallocMC: Memory Allocator for Many Core Architectures.
  http://www.icg.tugraz.at/project/mvp
  https://www.hzdr.de/crp

  Copyright (C) 2012 Institute for Computer Graphics and Vision,
                     Graz University of Technology
  Copyright (C) 2014 Institute of Radiation Physics,
                     Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Markus Steinberger - steinberger ( at ) icg.tugraz.at
              Michael Kenzel - kenzel ( at ) icg.tugraz.at
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

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <string>
#include <sstream>
#include <stdexcept>
#include <boost/cstdint.hpp>

#include "mallocMC_prefixes.hpp"


namespace CUDA
{
  class error : public std::runtime_error
  {
  private:
    static std::string genErrorString(cudaError errorValue, const char* file, int line)
    {
      std::ostringstream msg;
      msg << file << '(' << line << "): error: " << cudaGetErrorString(errorValue);
      return msg.str();
    }
  public:
    error(cudaError errorValue, const char* file, int line)
      : runtime_error(genErrorString(errorValue, file, line))
    {
    }

    error(cudaError errorValue)
      : runtime_error(cudaGetErrorString(errorValue))
    {
    }

    error(const std::string& msg)
      : runtime_error(msg)
    {
    }
  };

  inline void checkError(cudaError errorValue, const char* file, int line)
  {
#ifdef _DEBUG
    if (errorValue != cudaSuccess)
      throw CUDA::error(errorValue, file, line);
#endif
  }

  inline void checkError(const char* file, int line)
  {
    checkError(cudaGetLastError(), file, line);
  }

  inline void checkError()
  {
#ifdef _DEBUG
    cudaError errorValue = cudaGetLastError();
    if (errorValue != cudaSuccess)
      throw CUDA::error(errorValue);
#endif
  }

#define MALLOCMC_CUDA_CHECKED_CALL(call) CUDA::checkError(call, __FILE__, __LINE__)
#define MALLOCMC_CUDA_CHECK_ERROR() CUDA::checkError(__FILE__, __LINE__)
}


#define warp_serial                                    \
  for (unsigned int __mask = __ballot(1),              \
            __num = __popc(__mask),                    \
            __lanemask = mallocMC::lanemask_lt(),      \
            __local_id = __popc(__lanemask & __mask),  \
            __active = 0;                              \
       __active < __num;                               \
       ++__active)                                     \
    if (__active == __local_id)


namespace mallocMC 
{

  template<int PSIZE>
  class __PointerEquivalent
  {
  public:
    typedef unsigned int type;
  };
  template<>
  class __PointerEquivalent<8>
  {
  public:
    typedef unsigned long long int type;
  };

  typedef mallocMC::__PointerEquivalent<sizeof(char*)>::type PointerEquivalent;


  MAMC_ACCELERATOR inline uint32_t laneid()
  {
    uint32_t mylaneid;
    asm("mov.u32 %0, %laneid;" : "=r" (mylaneid));
    return mylaneid;
  }

  MAMC_ACCELERATOR inline uint32_t warpid()
  {
    uint32_t mywarpid;
    asm("mov.u32 %0, %warpid;" : "=r" (mywarpid));
    return mywarpid;
  }
  MAMC_ACCELERATOR inline uint32_t nwarpid()
  {
    uint32_t mynwarpid;
    asm("mov.u32 %0, %nwarpid;" : "=r" (mynwarpid));
    return mynwarpid;
  }

  MAMC_ACCELERATOR inline uint32_t smid()
  {
    uint32_t mysmid;
    asm("mov.u32 %0, %smid;" : "=r" (mysmid));
    return mysmid;
  }

  MAMC_ACCELERATOR inline uint32_t nsmid()
  {
    uint32_t mynsmid;
    asm("mov.u32 %0, %nsmid;" : "=r" (mynsmid));
    return mynsmid;
  }
  MAMC_ACCELERATOR inline uint32_t lanemask()
  {
    uint32_t lanemask;
    asm("mov.u32 %0, %lanemask_eq;" : "=r" (lanemask));
    return lanemask;
  }

  MAMC_ACCELERATOR inline uint32_t lanemask_le()
  {
    uint32_t lanemask;
    asm("mov.u32 %0, %lanemask_le;" : "=r" (lanemask));
    return lanemask;
  }

  MAMC_ACCELERATOR inline uint32_t lanemask_lt()
  {
    uint32_t lanemask;
    asm("mov.u32 %0, %lanemask_lt;" : "=r" (lanemask));
    return lanemask;
  }

  MAMC_ACCELERATOR inline uint32_t lanemask_ge()
  {
    uint32_t lanemask;
    asm("mov.u32 %0, %lanemask_ge;" : "=r" (lanemask));
    return lanemask;
  }

  MAMC_ACCELERATOR inline uint32_t lanemask_gt()
  {
    uint32_t lanemask;
    asm("mov.u32 %0, %lanemask_gt;" : "=r" (lanemask));
    return lanemask;
  }

  template<class T>
  MAMC_HOST MAMC_ACCELERATOR inline T divup(T a, T b) { return (a + b - 1)/b; }

}
