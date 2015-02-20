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

#include <boost/mpl/bool.hpp>
#include <boost/mpl/int.hpp>

namespace mallocMC{
namespace CreationPolicies{
namespace ScatterConf{
  struct DefaultScatterConfig{
    typedef boost::mpl::int_<4096>  pagesize;
    typedef boost::mpl::int_<8>     accessblocks;
    typedef boost::mpl::int_<16>    regionsize;
    typedef boost::mpl::int_<2>     wastefactor;
    typedef boost::mpl::bool_<false> resetfreedpages;
  };

  struct DefaultScatterHashingParams{
    typedef boost::mpl::int_<38183> hashingK;
    typedef boost::mpl::int_<17497> hashingDistMP;
    typedef boost::mpl::int_<1>     hashingDistWP;
    typedef boost::mpl::int_<1>     hashingDistWPRel;
  };  
}

  /**
   * @brief fast memory allocation based on ScatterAlloc
   *
   * This CreationPolicy implements a fast memory allocator that trades speed
   * for fragmentation of memory. This is based on the memory allocator
   * "ScatterAlloc"
   * (http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6339604), and
   * is extended to report free memory slots of a given size (both on host and
   * accelerator).
   * To work properly, this policy class requires a pre-allocated heap on the
   * accelerator and works only with Nvidia CUDA capable accelerators that have
   * at least compute capability 2.0.
   *
   * @tparam T_Config (optional) configure the heap layout. The
   *        default can be obtained through Scatter<>::HeapProperties
   * @tparam T_Hashing (optional) configure the parameters for
   *        the hashing formula. The default can be obtained through
   *        Scatter<>::HashingProperties
   */
  template<
  class T_Config = ScatterConf::DefaultScatterConfig,
  class T_Hashing = ScatterConf::DefaultScatterHashingParams
  >
  class Scatter;

}// namespace CreationPolicies
}// namespace mallocMC
