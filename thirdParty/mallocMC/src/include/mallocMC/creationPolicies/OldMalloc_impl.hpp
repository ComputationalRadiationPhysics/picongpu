/*
  mallocMC: Memory Allocator for Many Core Architectures.

  Copyright 2014 Institute of Radiation Physics,
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

#include <boost/cstdint.hpp>
#include <boost/mpl/bool.hpp>

#include "OldMalloc.hpp"

namespace mallocMC{
namespace CreationPolicies{
    
  class OldMalloc
  {
    typedef boost::uint32_t uint32;

    public:
    typedef boost::mpl::bool_<false> providesAvailableSlots;

    __device__ void* create(uint32 bytes)
    {
      return ::malloc(static_cast<size_t>(bytes));
    }

    __device__ void destroy(void* mem)
    {
      free(mem);
    }

    __device__ bool isOOM(void* p, size_t s){
      return s && (p == NULL);
    }

    template < typename T>
    static void* initHeap(const T& obj, void* pool, size_t memsize){
      T* dAlloc;
      MALLOCMC_CUDA_CHECKED_CALL(cudaGetSymbolAddress((void**)&dAlloc,obj));
      return dAlloc;
    }   

    template < typename T>
    static void finalizeHeap(const T& obj, void* pool){
      return;
    }

    static std::string classname(){
      return "OldMalloc";
    }

  };

} //namespace CreationPolicies
} //namespace mallocMC
