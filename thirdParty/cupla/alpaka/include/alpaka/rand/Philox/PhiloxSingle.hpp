/* Copyright 2021 Jiri Vyskocil
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/rand/Philox/MultiplyAndSplit64to32.hpp>
#include <alpaka/rand/Philox/PhiloxBaseTraits.hpp>

#include <utility>

namespace alpaka
{
    namespace rand
    {
        namespace engine
        {
            /** Philox state for single value engine
             *
             * @tparam TCounter Type of the Counter array
             * @tparam TKey Type of the Key array
             */
            template<typename TCounter, typename TKey>
            struct PhiloxStateSingle
            {
                using Counter = TCounter;
                using Key = TKey;

                Counter counter; ///< Counter array
                Key key; ///< Key array
                Counter result; ///< Intermediate result array
                std::uint32_t position; ///< Pointer to the active intermediate result element
                // TODO: Box-Muller states
            };

            /** Philox engine generating a single number
             *
             * This engine's operator() will return a single number. Since the result is the same size as the counter,
             * and so it contains more than one number, it has to be stored between individual invocations of
             * operator(). Additionally a pointer has to be stored indicating which part of the result array is to be
             * returned next.
             *
             * @tparam TAcc Accelerator type as defined in alpaka/acc
             * @tparam TParams Basic parameters for the Philox algorithm
             */
            template<typename TAcc, typename TParams>
            class PhiloxSingle : public traits::PhiloxBaseTraits<TAcc, TParams, PhiloxSingle<TAcc, TParams>>::Base
            {
            public:
                /// Specialization for different TAcc backends
                using Traits = typename traits::PhiloxBaseTraits<TAcc, TParams, PhiloxSingle<TAcc, TParams>>;

                using Counter = typename Traits::Counter; ///< Backend-dependent Counter type
                using Key = typename Traits::Key; ///< Backend-dependent Key type
                using State = PhiloxStateSingle<Counter, Key>; ///< Backend-dependent State type

                State state; ///< Internal engine state

            protected:
                /** Advance internal counter to the next value
                 *
                 * Advances the full internal counter array, resets the position pointer and stores the intermediate
                 * result to be recalled when the user requests a number.
                 */
                ALPAKA_FN_HOST_ACC void advanceState()
                {
                    this->advanceCounter(state.counter);
                    state.result = this->nRounds(state.counter, state.key);
                    state.position = 0;
                }

                /** Get the next random number and advance internal state
                 *
                 * The intermediate result stores N = TParams::counterSize numbers. Check if we've already given out
                 * all of them. If so, generate a new intermediate result (this also resets the pointer to the position
                 * of the actual number). Finally, we return the actual number.
                 *
                 * @return The next random number
                 */
                ALPAKA_FN_HOST_ACC auto nextNumber()
                {
                    state.position++;
                    if(state.position == TParams::counterSize)
                    {
                        advanceState();
                    }
                    return state.result[state.position];
                }

                /// Skips the next \a offset numbers
                ALPAKA_FN_HOST_ACC void skip(uint64_t offset)
                {
                    state.position += offset & 3;
                    offset += state.position < 4 ? 0 : 4;
                    state.position -= state.position < 4 ? 0 : 4u;
                    this->skip4(offset / 4);
                }

            public:
                /** Construct a new Philox engine with single-value output
                 *
                 * @param seed Set the Philox generator key
                 * @param subsequence Select a subsequence of size 2^64
                 * @param offset Skip \a offset numbers form the start of the subsequence
                 */
                ALPAKA_FN_HOST_ACC PhiloxSingle(uint64_t seed = 0, uint64_t subsequence = 0, uint64_t offset = 0)
                    : state{{0, 0, 0, 0}, {low32Bits(seed), high32Bits(seed)}, {0, 0, 0, 0}, 0}
                {
                    this->skipSubsequence(subsequence);
                    skip(offset);
                    advanceState();
                }

                ALPAKA_FN_HOST_ACC PhiloxSingle(PhiloxSingle const& other) : state{other.state}
                {
                }

                /** Get the next random number
                 *
                 * @return The next random number
                 */
                ALPAKA_FN_HOST_ACC auto operator()()
                {
                    return nextNumber();
                }
            };
        } // namespace engine
    } // namespace rand
} // namespace alpaka
