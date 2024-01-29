/* Copyright 2022 Jiri Vyskocil, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/rand/Philox/MultiplyAndSplit64to32.hpp"
#include "alpaka/rand/Philox/PhiloxBaseTraits.hpp"

#include <utility>

namespace alpaka::rand::engine
{
    /** Philox state for vector generator
     *
     * @tparam TCounter Type of the Counter array
     * @tparam TKey Type of the Key array
     */
    template<typename TCounter, typename TKey>
    struct PhiloxStateVector
    {
        using Counter = TCounter;
        using Key = TKey;

        Counter counter; ///< Counter array
        Key key; ///< Key array
    };

    /** Philox engine generating a vector of numbers
     *
     * This engine's operator() will return a vector of numbers corresponding to the full size of its counter.
     * This is a convenience vs. memory size tradeoff since the user has to deal with the output array
     * themselves, but the internal state comprises only of a single counter and a key.
     *
     * @tparam TAcc Accelerator type as defined in alpaka/acc
     * @tparam TParams Basic parameters for the Philox algorithm
     */
    template<typename TAcc, typename TParams>
    class PhiloxVector : public trait::PhiloxBaseTraits<TAcc, TParams, PhiloxVector<TAcc, TParams>>::Base
    {
    public:
        /// Specialization for different TAcc backends
        using Traits = trait::PhiloxBaseTraits<TAcc, TParams, PhiloxVector<TAcc, TParams>>;

        using Counter = typename Traits::Counter; ///< Backend-dependent Counter type
        using Key = typename Traits::Key; ///< Backend-dependent Key type
        using State = PhiloxStateVector<Counter, Key>; ///< Backend-dependent State type
        template<typename TDistributionResultScalar>
        using ResultContainer = typename Traits::template ResultContainer<TDistributionResultScalar>;

        State state;

    protected:
        /** Get the next array of random numbers and advance internal state
         *
         * @return The next array of random numbers
         */
        ALPAKA_FN_HOST_ACC auto nextVector()
        {
            this->advanceCounter(state.counter);
            return this->nRounds(state.counter, state.key);
        }

        /** Skips the next \a offset vectors
         *
         * Unlike its counterpart in \a PhiloxSingle, this function advances the state in multiples of the
         * counter size thus skipping the entire array of numbers.
         */
        ALPAKA_FN_HOST_ACC void skip(uint64_t offset)
        {
            this->skip4(offset);
        }

    public:
        /** Construct a new Philox engine with vector output
         *
         * @param seed Set the Philox generator key
         * @param subsequence Select a subsequence of size 2^64
         * @param offset Skip \a offset numbers form the start of the subsequence
         */
        ALPAKA_FN_HOST_ACC PhiloxVector(uint64_t seed = 0, uint64_t subsequence = 0, uint64_t offset = 0)
            : state{{0, 0, 0, 0}, {low32Bits(seed), high32Bits(seed)}}
        {
            this->skipSubsequence(subsequence);
            skip(offset);
            nextVector();
        }

        /** Get the next vector of random numbers
         *
         * @return The next vector of random numbers
         */
        ALPAKA_FN_HOST_ACC auto operator()()
        {
            return nextVector();
        }
    };
} // namespace alpaka::rand::engine
