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

#include "BadAllocException.hpp"

#include <alpaka/core/Common.hpp>

#include <cassert>
#include <string>

namespace mallocMC
{
    namespace OOMPolicies
    {
        /**
         * @brief Throws a std::bad_alloc exception on OutOfMemory
         *
         * This OOMPolicy will throw a std::bad_alloc exception, if the
         * accelerator supports it. Currently, Nvidia CUDA does not support any
         * form of exception handling, therefore handleOOM() does not have any
         * effect on these accelerators. Using this policy on other types of
         * accelerators that do not support exceptions results in undefined
         * behaviour.
         */
        struct BadAllocException
        {
            ALPAKA_FN_ACC
            static auto handleOOM(void* mem) -> void*
            {
#if BOOST_LANG_CUDA || BOOST_COMP_HIP
// #if __CUDA_ARCH__ < 350
#    define PM_EXCEPTIONS_NOT_SUPPORTED_HERE
// #endif
#endif

#ifdef PM_EXCEPTIONS_NOT_SUPPORTED_HERE
#    undef PM_EXCEPTIONS_NOT_SUPPORTED_HERE
                assert(false);
#else
                throw std::bad_alloc{};
#endif
                return mem;
            }

            static auto classname() -> std::string
            {
                return "BadAllocException";
            }
        };

    } // namespace OOMPolicies
} // namespace mallocMC
