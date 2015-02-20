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

#include <boost/mpl/int.hpp>

namespace mallocMC{
namespace DistributionPolicies{
    
  namespace XMallocSIMDConf{
    struct DefaultXMallocConfig{
      typedef boost::mpl::int_<4096>     pagesize;
    };  
  }

  /**
   * @brief SIMD optimized chunk resizing in the style of XMalloc
   *
   * This DistributionPolicy can take the memory requests from a group of
   * worker threads and combine them, so that only one of the workers will
   * allocate the whole request. Later, each worker gets an appropriate offset
   * into the allocated chunk. This is beneficial for SIMD architectures since
   * only one of the workers has to compete for the resource.  This algorithm
   * is inspired by the XMalloc memory allocator
   * (http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=5577907&tag=1) and
   * its implementation in ScatterAlloc
   * (http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6339604)
   * XMallocSIMD is inteded to be used with Nvidia CUDA capable accelerators
   * that support at least compute capability 2.0
   *
   * @tparam T_Config (optional) The configuration struct to overwrite
   *        default configuration. The default can be obtained through
   *        XMallocSIMD<>::Properties
   */
  template<class T_Config=XMallocSIMDConf::DefaultXMallocConfig>
  class XMallocSIMD;


} //namespace DistributionPolicies
} //namespace mallocMC
