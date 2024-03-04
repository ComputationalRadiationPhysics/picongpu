/* Copyright 2022 Jiri Vyskocil, Bernhard Manfred Gruber, Jeffrey Kelling
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/rand/Philox/PhiloxStateless.hpp"

#include <utility>

namespace alpaka::rand::engine
{
    /** Common class for Philox family engines
     *
     * Relies on `PhiloxStateless` to provide the PRNG and adds state to handling the counting.
     *
     * @tparam TBackend device-dependent backend, specifies the array types
     * @tparam TParams Philox algorithm parameters \sa PhiloxParams
     * @tparam TImpl engine type implementation (CRTP)
     *
     * static const data members are transformed into functions, because GCC
     * assumes types with static data members to be not mappable and makes not
     * exception for constexpr ones. This is a valid interpretation of the
     * OpenMP <= 4.5 standard. In OpenMP >= 5.0 types with any kind of static
     * data member are mappable.
     */
    template<typename TBackend, typename TParams, typename TImpl>
    class PhiloxBaseCommon
        : public TBackend
        , public PhiloxStateless<TBackend, TParams>
    {
    public:
        using Counter = typename PhiloxStateless<TBackend, TParams>::Counter;
        using Key = typename PhiloxStateless<TBackend, TParams>::Key;

    protected:
        /** Advance the \a counter to the next state
         *
         * Increments the passed-in \a counter by one with a 128-bit carry.
         *
         * @param counter reference to the counter which is to be advanced
         */
        ALPAKA_FN_HOST_ACC void advanceCounter(Counter& counter)
        {
            counter[0]++;
            /* 128-bit carry */
            if(counter[0] == 0)
            {
                counter[1]++;
                if(counter[1] == 0)
                {
                    counter[2]++;
                    if(counter[2] == 0)
                    {
                        counter[3]++;
                    }
                }
            }
        }

        /** Advance the internal state counter by \a offset N-vectors (N = counter size)
         *
         * Advances the internal value of this->state.counter
         *
         * @param offset number of N-vectors to skip
         */
        ALPAKA_FN_HOST_ACC void skip4(uint64_t offset)
        {
            Counter& counter = static_cast<TImpl*>(this)->state.counter;
            Counter temp = counter;
            counter[0] += low32Bits(offset);
            counter[1] += high32Bits(offset) + (counter[0] < temp[0] ? 1 : 0);
            counter[2] += (counter[0] < temp[1] ? 1u : 0u);
            counter[3] += (counter[0] < temp[2] ? 1u : 0u);
        }

        /** Advance the counter by the length of \a subsequence
         *
         * Advances the internal value of this->state.counter
         *
         * @param subsequence number of subsequences to skip
         */
        ALPAKA_FN_HOST_ACC void skipSubsequence(uint64_t subsequence)
        {
            Counter& counter = static_cast<TImpl*>(this)->state.counter;
            Counter temp = counter;
            counter[2] += low32Bits(subsequence);
            counter[3] += high32Bits(subsequence) + (counter[2] < temp[2] ? 1 : 0);
        }
    };
} // namespace alpaka::rand::engine
