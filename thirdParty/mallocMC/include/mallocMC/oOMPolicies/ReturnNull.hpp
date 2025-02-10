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

#include "ReturnNull.hpp"

#include <alpaka/core/Common.hpp>

#include <string>

namespace mallocMC
{
    namespace OOMPolicies
    {
        /**
         * @brief Returns a nullptr pointer on OutOfMemory conditions
         *
         * This OOMPolicy will return nullptr, if handleOOM() is called.
         */
        class ReturnNull
        {
        public:
            ALPAKA_FN_ACC
            static auto handleOOM([[maybe_unused]] void* mem) -> void*
            {
                return nullptr;
            }

            static auto classname() -> std::string
            {
                return "ReturnNull";
            }
        };

    } // namespace OOMPolicies
} // namespace mallocMC
