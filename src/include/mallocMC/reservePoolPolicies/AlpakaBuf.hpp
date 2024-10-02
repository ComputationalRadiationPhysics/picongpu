/*
  mallocMC: Memory Allocator for Many Core Architectures.

  Copyright 2020 Helmholtz-Zentrum Dresden - Rossendorf,
                 CERN

  Author(s):  Bernhard Manfred Gruber

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

#include <alpaka/alpaka.hpp>
#include <memory>
#include <string>

namespace mallocMC
{
    namespace ReservePoolPolicies
    {
        template<typename AlpakaAcc>
        struct AlpakaBuf
        {
            template<typename AlpakaDev>
            auto setMemPool(const AlpakaDev& dev, size_t memsize) -> void*
            {
                poolBuffer = std::make_unique<PoolBufferType>(alpaka::allocBuf<unsigned char, size_t>(dev, memsize));
                return alpaka::getPtrNative(*poolBuffer);
            }

            void resetMemPool()
            {
                poolBuffer = {};
            }

            static auto classname() -> std::string
            {
                return "AlpakaBuf";
            }

        private:
            using PoolBufferType = alpaka::Buf<alpaka::Dev<AlpakaAcc>, unsigned char, alpaka::DimInt<1>, size_t>;
            std::unique_ptr<PoolBufferType> poolBuffer; // FIXME(bgruber): replace by std::optional<>
        };
    } // namespace ReservePoolPolicies
} // namespace mallocMC
