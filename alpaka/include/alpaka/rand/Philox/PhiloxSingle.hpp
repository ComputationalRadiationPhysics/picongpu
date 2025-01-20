/* Copyright 2022 Jiri Vyskocil, Rene Widera, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/rand/Philox/MultiplyAndSplit64to32.hpp"
#include "alpaka/rand/Philox/PhiloxBaseCommon.hpp"

#include <utility>

namespace alpaka::rand::engine
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

        /// Counter array
        Counter counter;
        /// Key array
        Key key;
        /// Intermediate result array
        Counter result;
        /// Pointer to the active intermediate result element
        std::uint32_t position;
        // TODO: Box-Muller states
    };

    /** Philox engine generating a single number
     *
     * This engine's operator() will return a single number. Since the result is the same size as the counter,
     * and so it contains more than one number, it has to be stored between individual invocations of
     * operator(). Additionally a pointer has to be stored indicating which part of the result array is to be
     * returned next.
     *
     * @tparam TParams Basic parameters for the Philox algorithm
     */
    template<typename TParams>
    class PhiloxSingle : public PhiloxBaseCommon<TParams, PhiloxSingle<TParams>>
    {
    public:
        using Base = PhiloxBaseCommon<TParams, PhiloxSingle<TParams>>;

        /// Counter type
        using Counter = typename Base::Counter;
        /// Key type
        using Key = typename Base::Key;
        /// State type
        using State = PhiloxStateSingle<Counter, Key>;

        /// Internal engine state
        State state;

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
            // Element zero will always contain the next valid random number.
            auto result = state.result[0];
            state.position++;
            if(state.position == TParams::counterSize)
            {
                advanceState();
            }
            else
            {
                // Shift state results to allow hard coded access to element zero.
                // This will avoid high register usage on NVIDIA devices.
                // \todo Check if this shifting of the result vector is decreasing CPU performance.
                //       If so this optimization for GPUs (mostly NVIDIA) should be moved into
                //       PhiloxBaseCudaArray.
                state.result[0] = state.result[1];
                state.result[1] = state.result[2];
                state.result[2] = state.result[3];
            }

            return result;
        }

        /// Skips the next \a offset numbers
        ALPAKA_FN_HOST_ACC void skip(uint64_t offset)
        {
            static_assert(TParams::counterSize == 4, "Only counterSize is supported.");
            state.position = static_cast<decltype(state.position)>(state.position + (offset & 3));
            offset += state.position < 4 ? 0 : 4;
            state.position -= state.position < 4 ? 0 : 4u;
            for(auto numShifts = state.position; numShifts > 0; --numShifts)
            {
                // Shift state results to allow hard coded access to element zero.
                // This will avoid high register usage on NVIDIA devices.
                state.result[0] = state.result[1];
                state.result[1] = state.result[2];
                state.result[2] = state.result[3];
            }
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

        /** Get the next random number
         *
         * @return The next random number
         */
        ALPAKA_FN_HOST_ACC auto operator()()
        {
            return nextNumber();
        }
    };
} // namespace alpaka::rand::engine
