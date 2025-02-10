/*
  mallocMC: Memory Allocator for Many Core Architectures.
  https://www.hzdr.de/crp

  Copyright 2014 - 2015 Institute of Radiation Physics,
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

#include <alpaka/core/Common.hpp>

namespace mallocMC
{
    template<typename T_HostAllocator>
    struct AllocatorHandleImpl
    {
        using DevAllocator = typename T_HostAllocator::DevAllocator;

        DevAllocator* devAllocator;

        explicit AllocatorHandleImpl(DevAllocator* p) : devAllocator(p)
        {
        }

        template<typename AlpakaAcc>
        ALPAKA_FN_ACC auto malloc(AlpakaAcc const& acc, size_t size) -> void*
        {
            return devAllocator->malloc(acc, size);
        }

        template<typename AlpakaAcc>
        ALPAKA_FN_ACC void free(AlpakaAcc const& acc, void* p)
        {
            devAllocator->free(acc, p);
        }

        template<typename AlpakaAcc>
        ALPAKA_FN_ACC auto getAvailableSlots(AlpakaAcc const& acc, size_t slotSize) -> unsigned
        {
            return devAllocator->getAvailableSlots(acc, slotSize);
        }
    };

} // namespace mallocMC
