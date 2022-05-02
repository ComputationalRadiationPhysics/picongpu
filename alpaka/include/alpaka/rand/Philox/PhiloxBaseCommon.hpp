/* Copyright 2022 Jiri Vyskocil, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/rand/Philox/MultiplyAndSplit64to32.hpp>
#include <alpaka/rand/Philox/PhiloxConstants.hpp>

#include <utility>


namespace alpaka::rand::engine
{
    /** Philox algorithm parameters
     *
     * @tparam TCounterSize number of elements in the counter
     * @tparam TWidth width of one counter element (in bits)
     * @tparam TRounds number of S-box rounds
     */
    template<unsigned TCounterSize, unsigned TWidth, unsigned TRounds>
    struct PhiloxParams
    {
        static unsigned constexpr counterSize = TCounterSize;
        static unsigned constexpr width = TWidth;
        static unsigned constexpr rounds = TRounds;
    };

    /** Common class for Philox family engines
     *
     * Checks the validity of passed-in parameters and calls the \a TBackend methods to perform N rounds of the
     * Philox shuffle.
     *
     * @tparam TBackend device-dependent backend, specifies the array types
     * @tparam TParams Philox algorithm parameters \sa PhiloxParams
     * @tparam TImpl engine type implementation (CRTP)
     */
    template<typename TBackend, typename TParams, typename TImpl>
    class PhiloxBaseCommon
        : public TBackend
        , public PhiloxConstants<TParams>
    {
        static unsigned const numRounds = TParams::rounds;
        static unsigned const vectorSize = TParams::counterSize;
        static unsigned const numberWidth = TParams::width;

        static_assert(numRounds > 0, "Number of Philox rounds must be > 0.");
        static_assert(vectorSize % 2 == 0, "Philox counter size must be an even number.");
        static_assert(vectorSize <= 16, "Philox SP network is not specified for sizes > 16.");
        static_assert(numberWidth % 8 == 0, "Philox number width in bits must be a multiple of 8.");

        // static_assert(TWidth == 32 || TWidth == 64, "Philox implemented only for 32 and 64 bit numbers.");
        static_assert(numberWidth == 32, "Philox implemented only for 32 bit numbers.");

    public:
        using Counter = typename TBackend::Counter;
        using Key = typename TBackend::Key;

    protected:
        /** Single round of the Philox shuffle
         *
         * @param counter state of the counter
         * @param key value of the key
         * @return shuffled counter
         */
        ALPAKA_FN_HOST_ACC auto singleRound(Counter const& counter, Key const& key)
        {
            std::uint32_t H0, L0, H1, L1;
            multiplyAndSplit64to32(counter[0], this->MULTIPLITER_4x32_0, H0, L0);
            multiplyAndSplit64to32(counter[2], this->MULTIPLITER_4x32_1, H1, L1);
            return Counter{H1 ^ counter[1] ^ key[0], L1, H0 ^ counter[3] ^ key[1], L0};
        }

        /** Bump the \a key by the Weyl sequence step parameter
         *
         * @param key the key to be bumped
         * @return the bumped key
         */
        ALPAKA_FN_HOST_ACC auto bumpKey(Key const& key)
        {
            return Key{key[0] + this->WEYL_32_0, key[1] + this->WEYL_32_1};
        }

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

        /** Performs N rounds of the Philox shuffle
         *
         * @param counter_in initial state of the counter
         * @param key_in initial state of the key
         * @return result of the PRNG shuffle; has the same size as the counter
         */
        ALPAKA_FN_HOST_ACC auto nRounds(Counter const& counter_in, Key const& key_in) -> Counter
        {
            Key key{key_in};
            Counter counter = singleRound(counter_in, key);

            // TODO: Consider unrolling the loop for performance
            for(unsigned int n = 0; n < numRounds; ++n)
            {
                key = bumpKey(key);
                counter = singleRound(counter, key);
            }
            // TODO: Should the key be returned as well??? i.e. should the original key be bumped?

            return counter;
        }
    };
} // namespace alpaka::rand::engine
