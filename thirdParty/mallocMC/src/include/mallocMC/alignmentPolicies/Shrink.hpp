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

#include <boost/mpl/int.hpp>

namespace mallocMC{
namespace AlignmentPolicies{

namespace ShrinkConfig{
  struct DefaultShrinkConfig{
    typedef boost::mpl::int_<16> dataAlignment;
  };
}

  /**
   * @brief Provides proper alignment of pool and pads memory requests
   *
   * This AlignmentPolicy is based on ideas from ScatterAlloc
   * (http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6339604). It
   * performs alignment operations on big memory pools and requests to allocate
   * memory. Memory pools are truncated at the beginning until the pointer to
   * the memory fits the alignment. Requests to allocate memory are padded
   * until their size is a multiple of the alignment.
   *
   * @tparam T_Config (optional) The alignment to use
   */
  template<typename T_Config = ShrinkConfig::DefaultShrinkConfig>
  class Shrink;

} //namespace AlignmentPolicies
} //namespace mallocMC
