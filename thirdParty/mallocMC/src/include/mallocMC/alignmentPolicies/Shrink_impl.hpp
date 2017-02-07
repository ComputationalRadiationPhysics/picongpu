/*
  mallocMC: Memory Allocator for Many Core Architectures.
  http://www.icg.tugraz.at/project/mvp

  Copyright (C) 2012 Institute for Computer Graphics and Vision,
                     Graz University of Technology
  Copyright (C) 2014 Institute of Radiation Physics,
                     Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Markus Steinberger - steinberger ( at ) icg.tugraz.at
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
#include <iostream>
#include <string>
#include <sstream>
#include <boost/tuple/tuple.hpp>

#include "Shrink.hpp"
#include "../mallocMC_prefixes.hpp"

namespace mallocMC{
namespace AlignmentPolicies{

namespace Shrink2NS{
    
  template<int PSIZE> struct __PointerEquivalent{ typedef unsigned int type;};
  template<>
  struct __PointerEquivalent<8>{ typedef unsigned long long int type; };
}// namespace ShrinkNS

  template<typename T_Config>
  class Shrink{
    public:
    typedef T_Config Properties;

    private:
    typedef boost::uint32_t uint32;
    typedef Shrink2NS::__PointerEquivalent<sizeof(char*)>::type PointerEquivalent;

/** Allow for a hierarchical validation of parameters:
 *
 * shipped default-parameters (in the inherited struct) have lowest precedence.
 * They will be overridden by a given configuration struct. However, even the
 * given configuration struct can be overridden by compile-time command line
 * parameters (e.g. -D MALLOCMC_AP_SHRINK_DATAALIGNMENT 128)
 *
 * default-struct < template-struct < command-line parameter
 */
#ifndef MALLOCMC_AP_SHRINK_DATAALIGNMENT
#define MALLOCMC_AP_SHRINK_DATAALIGNMENT Properties::dataAlignment::value
#endif
    BOOST_STATIC_CONSTEXPR uint32 dataAlignment = MALLOCMC_AP_SHRINK_DATAALIGNMENT;

    // \TODO: The static_cast can be removed once the minimal dependencies of
    //        this project is are at least CUDA 7.0 and gcc 4.8.2
    BOOST_STATIC_ASSERT(static_cast<uint32>(dataAlignment) > 0);
    //dataAlignment must also be a power of 2!
    BOOST_STATIC_ASSERT(dataAlignment && !(dataAlignment & (dataAlignment-1)) ); 

    public:
    static boost::tuple<void*,size_t> alignPool(void* memory, size_t memsize){
      PointerEquivalent alignmentstatus = ((PointerEquivalent)memory) & (dataAlignment -1);
      if(alignmentstatus != 0)
      {
        std::cout << "Heap Warning: memory to use not ";
        std::cout << dataAlignment << " byte aligned..."        << std::endl;
        std::cout << "Before:"                                  << std::endl;
        std::cout << "dataAlignment:   " << dataAlignment       << std::endl;
        std::cout << "Alignmentstatus: " << alignmentstatus     << std::endl;
        std::cout << "size_t memsize   " << memsize << " byte"  << std::endl;
        std::cout << "void *memory     " << memory              << std::endl;

        memory   = (void*)(((PointerEquivalent)memory) + dataAlignment - alignmentstatus);
        memsize -= (size_t)dataAlignment + (size_t)alignmentstatus;

        std::cout << "Was shrunk automatically to: "            << std::endl;
        std::cout << "size_t memsize   " << memsize << " byte"  << std::endl;
        std::cout << "void *memory     " << memory              << std::endl;
      }

      return boost::make_tuple(memory,memsize);
    }

    MAMC_HOST
    MAMC_ACCELERATOR
    static uint32 applyPadding(uint32 bytes){
      return (bytes + dataAlignment - 1) & ~(dataAlignment-1);
    }

    MAMC_HOST
    static std::string classname(){
      std::stringstream ss;
      ss << "Shrink[" << dataAlignment << "]";
      return ss.str();
    }

  };

} //namespace AlignmentPolicies
} //namespace mallocMC
