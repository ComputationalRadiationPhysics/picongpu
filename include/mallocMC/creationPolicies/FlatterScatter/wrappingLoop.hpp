/*
  mallocMC: Memory Allocator for Many Core Architectures.

  Copyright 2024 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Julian Johannes Lenz, Rene Widera

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


#include <alpaka/core/Common.hpp>

#include <cstdint>

/**
 * @brief Abstraction of a short-circuiting loop that wraps around from an arbitrary starting point within the range.
 *
 * This implements a re-occuring pattern in the code: Due to the scattering approach taken, we're often in a position
 * where we want to run a simple loop except for the fact that we start in an arbitrary position within the range and
 * complete it by wrapping around to the start of the range continuing from there. Furthermore, these loops are all
 * searches, so it's advantageous to implement short-circuiting by early exit in case of finding another value than the
 * provided failureValue.
 *
 * @tparam T_size Type of size-like arguments. This function is used in various contexts where this can either be
 * size_t or uint32_t.
 * @tparam TFunctor Type of the function representing the loop body (typically a lambda function).
 * @tparam TArgs Types of additional arguments provided to the function.
 * @param startIndex Index to start the loop at.
 * @param size Size of the range which equals the number of iterations to be performed in total.
 * @param failureValue Return value of the function indicating a failure of the current iteration and triggering the
 * next iteration.
 * @param func Function of type TFunctor representing the loop body. It is supposed to return a value of
 * decltype(failureValue) and indicate failure by returning the latter. Any other value is interpreted as success
 * triggering early exit of the loop.
 * @param args Additional arguments to be provided to the function on each iteration.
 * @return The return value of func which might be failureValue in case all iterations failed.
 */
template<typename TAcc, typename T_size, typename TFunctor, typename... TArgs>
ALPAKA_FN_INLINE ALPAKA_FN_ACC auto wrappingLoop(
    TAcc const& acc,
    T_size const startIndex,
    T_size const size,
    auto failureValue,
    TFunctor func,
    TArgs... args)
{
    for(uint32_t i = 0; i < size; ++i)
    {
        auto result = func(acc, (i + startIndex) % size, args...);
        if(result != failureValue)
        {
            return result;
        }
    }
    return failureValue;
}
