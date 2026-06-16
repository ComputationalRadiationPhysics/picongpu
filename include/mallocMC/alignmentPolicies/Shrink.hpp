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

#include "Shrink.hpp"

#include <alpaka/core/Common.hpp>

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>

namespace mallocMC
{
    namespace AlignmentPolicies
    {
        namespace Shrink2NS
        {
            template<int PSIZE>
            struct __PointerEquivalent
            {
                using type = unsigned int;
            };

            template<>
            struct __PointerEquivalent<8>
            {
                using type = unsigned long long;
            };
        } // namespace Shrink2NS

        namespace ShrinkConfig
        {
            struct DefaultShrinkConfig
            {
                static constexpr auto dataAlignment = 16;
            };
        } // namespace ShrinkConfig

        /**
         * @brief Provides proper alignment of pool and pads memory requests
         *
         * This AlignmentPolicy is based on ideas from ScatterAlloc
         * (http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6339604).
         * It performs alignment operations on big memory pools and requests to
         * allocate memory. Memory pools are truncated at the beginning until
         * the pointer to the memory fits the alignment. Requests to allocate
         * memory are padded until their size is a multiple of the alignment.
         *
         * @tparam T_Config (optional) The alignment to use
         */
        template<typename T_Config = ShrinkConfig::DefaultShrinkConfig>
        class Shrink
        {
        public:
            using Properties = T_Config;

        private:
            using PointerEquivalent = Shrink2NS::__PointerEquivalent<sizeof(char*)>::type;

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
#    define MALLOCMC_AP_SHRINK_DATAALIGNMENT (Properties::dataAlignment)
#endif
            static constexpr size_t dataAlignment = MALLOCMC_AP_SHRINK_DATAALIGNMENT;

            // dataAlignment must be a power of 2!
            static_assert(
                dataAlignment != 0 && (dataAlignment & (dataAlignment - 1)) == 0,
                "dataAlignment must also be a power of 2");

        public:
            static auto alignPool(void* memory, size_t memsize) -> std::tuple<void*, size_t>
            {
                PointerEquivalent alignmentstatus = ((PointerEquivalent) memory) & (dataAlignment - 1);
                if(alignmentstatus != 0)
                {
                    std::cout << "Heap Warning: memory to use not " << dataAlignment << " byte aligned...\n"
                              << "Before:\n"
                              << "dataAlignment:   " << dataAlignment << '\n'
                              << "Alignmentstatus: " << alignmentstatus << '\n'
                              << "size_t memsize   " << memsize << " byte" << '\n'
                              << "void *memory     " << memory << '\n';

                    memory = (void*) (((PointerEquivalent) memory) + dataAlignment - alignmentstatus);
                    memsize -= dataAlignment + (size_t) alignmentstatus;

                    std::cout << "Was shrunk automatically to: " << '\n'
                              << "size_t memsize   " << memsize << " byte" << '\n'
                              << "void *memory     " << memory << '\n';
                }

                return std::make_tuple(memory, memsize);
            }

            ALPAKA_FN_HOST_ACC
            static auto applyPadding(uint32_t bytes) -> uint32_t
            {
                constexpr uint32_t bitsToClear = dataAlignment - 1;
                return (bytes + bitsToClear) & ~bitsToClear;
            }

            ALPAKA_FN_HOST
            static auto classname() -> std::string
            {
                std::stringstream ss;
                ss << "Shrink[" << dataAlignment << "]";
                return ss.str();
            }
        };

    } // namespace AlignmentPolicies
} // namespace mallocMC
