/** Copyright 2019 Jakob Krude, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "Defines.hpp"

#include <cassert>
#include <limits>
#include <random>

namespace alpaka
{
    namespace test
    {
        namespace unit
        {
            namespace math
            {
                /**
                 * Fills buffer with random numbers (host-only).
                 *
                 * @tparam TData The used data-type (float || double).
                 * @tparam TArgs The args-buffer to be filled.
                 * @tparam TFunctor The used Functor-type.
                 * @param args The buffer that should be filled.
                 * @param functor The Functor, needed for ranges.
                 * @param seed The used seed.
                 */
                template<typename TData, typename TArgs, typename TFunctor>
                auto fillWithRndArgs(TArgs& args, TFunctor functor, unsigned int const& seed) -> void
                {
                    /*
                     * Each "sub-buffer" is filled with zero and/or max and/or lowest,
                     * depending on the specified range (at [0] - [2]).
                     *
                     * Every switch case needs to return!
                     * If no switch case was matched an assert(false) will be triggered.
                     *
                     * This function is easily extendable. It is only necessary to add extra
                     * definitions in the switch case, for more Range-types.
                     */
                    static_assert(
                        TArgs::value_type::arity == TFunctor::arity,
                        "Buffer properties must match TFunctor::arity");
                    static_assert(TArgs::capacity > 2, "Set of args must provide > 2 entries.");
                    constexpr auto max = std::numeric_limits<TData>::max();
                    constexpr auto low = std::numeric_limits<TData>::lowest();
                    std::default_random_engine eng{static_cast<std::default_random_engine::result_type>(seed)};

                    // These pseudo-random numbers are implementation/platform specific!
                    std::uniform_real_distribution<TData> dist(0, 1000);
                    std::uniform_real_distribution<TData> distOne(-1, 1);
                    for(size_t k = 0; k < TFunctor::arity_nr; ++k)
                    {
                        bool matchedSwitch = false;
                        switch(functor.ranges[k])
                        {
                        case Range::OneNeighbourhood:
                            matchedSwitch = true;
                            for(size_t i = 0; i < TArgs::capacity; ++i)
                            {
                                args(i).arg[k] = distOne(eng);
                            }
                            break;

                        case Range::PositiveOnly:
                            matchedSwitch = true;
                            args(0).arg[k] = max;
                            for(size_t i = 1; i < TArgs::capacity; ++i)
                            {
                                args(i).arg[k] = dist(eng) + static_cast<TData>(1);
                            }
                            break;

                        case Range::PositiveAndZero:
                            matchedSwitch = true;
                            args(0).arg[k] = 0.0;
                            args(1).arg[k] = max;
                            for(size_t i = 2; i < TArgs::capacity; ++i)
                            {
                                args(i).arg[k] = dist(eng);
                            }
                            break;

                        case Range::NotZero:
                            matchedSwitch = true;
                            args(0).arg[k] = max;
                            args(1).arg[k] = low;
                            for(size_t i = 2; i < TArgs::capacity; ++i)
                            {
                                TData arg;
                                do
                                {
                                    arg = dist(eng);
                                } while(std::equal_to<TData>()(arg, 1));
                                if(i % 2 == 0)
                                    args(i).arg[k] = arg;
                                else
                                    args(i).arg[k] = -arg;
                            }
                            break;

                        case Range::Unrestricted:
                            matchedSwitch = true;
                            args(0).arg[k] = 0.0;
                            args(1).arg[k] = max;
                            args(2).arg[k] = low;
                            for(size_t i = 3; i < TArgs::capacity; ++i)
                            {
                                if(i % 2 == 0)
                                    args(i).arg[k] = dist(eng);
                                else
                                    args(i).arg[k] = -dist(eng);
                            }
                            break;
                        }
                        // disable gcc-warning "unused variable"
                        alpaka::ignore_unused(matchedSwitch);
                        assert(matchedSwitch);
                    }
                }

            } // namespace math
        } // namespace unit
    } // namespace test
} // namespace alpaka
