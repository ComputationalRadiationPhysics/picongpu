/*
  mallocMC: Memory Allocator for Many Core Architectures.
  http://www.icg.tugraz.at/project/mvp
  https://www.hzdr.de/crp

  Copyright (C) 2012 Institute for Computer Graphics and Vision,
                     Graz University of Technology
  Copyright (C) 2014-2026 Institute of Radiation Physics,
                     Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Markus Steinberger - steinberger ( at ) icg.tugraz.at
              Bernhard Kainz - kainz ( at ) icg.tugraz.at
              Michael Kenzel - kenzel ( at ) icg.tugraz.at
              Rene Widera - r.widera ( at ) hzdr.de
              Axel Huebl - a.huebl ( at ) hzdr.de
              Carlchristian Eckert - c.eckert ( at ) hzdr.de
              Julian Lenz - j.lenz ( at ) hzdr.de

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

// This is a workaround for the following issue:
// https://github.com/llvm/llvm-project/pull/136133
// If clang or clang-based compilers like hipcc try to compile device code
// with a too recent version of libstdc++ from GCC,
// they run into issues like
// error: reference to __host__ function '__glibcxx_assert_fail' in __host__ __device__ function

#pragma once
#include <cstddef>

namespace mallocMC
{
    template<typename TData>
    struct span
    {
        TData* ptr_;
        size_t size_;

        constexpr span(TData* ptr, size_t size) : ptr_(ptr), size_(size) {};

        // This is explicitly NOT `explcit` because we want to be able to
        // silently wrap an array into a span within other constructor calls.
        template<size_t N>
        constexpr span(TData (&arr)[N]) : ptr_(arr)
                                        , size_(N)
        {
        }

        [[nodiscard]] constexpr auto size() const -> size_t
        {
            return size_;
        }

        [[nodiscard]] constexpr auto operator[](size_t index) const -> decltype(auto)
        {
            return ptr_[index];
        }

        [[nodiscard]] constexpr auto begin() const -> decltype(auto)
        {
            return ptr_;
        }

        [[nodiscard]] constexpr auto end() const -> decltype(auto)
        {
            return &(ptr_[size_]);
        }
    };
} // namespace mallocMC
