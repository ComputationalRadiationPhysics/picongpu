/* Copyright 2022 Jakob Krude, Benjamin Worpitz, Jeffrey Kelling, Jan Stephan, Sergei Bastrakov
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "Defines.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <random>

namespace mathtest
{
    //! Helper to generate random numbers of the given type for testing
    //!
    //! The general implementation supports float and double types
    //!
    //! @tparam TData generated type
    template<typename TData>
    struct RngWrapper
    {
        auto getMax()
        {
            return std::numeric_limits<TData>::max();
        }

        auto getLowest()
        {
            return std::numeric_limits<TData>::lowest();
        }

        auto getDistribution()
        {
            return std::uniform_real_distribution<TData>{0, 1000};
        }

        template<typename TDistribution, typename TEngine>
        auto getNumber(TDistribution& distribution, TEngine& engine)
        {
            return distribution(engine);
        }
    };

    //! Specialization for generating alpaka::Complex<TData>
    //!
    //! It has a much reduced range of numbers.
    //! The reason is, the results of operations much easier go to infinity area.
    //! Also, alpaka may emulate complex number math via calling other functions.
    //! As a result, it may produce some infinities and NaNs when the std:: implementation would not.
    //! So this range at least makes sure the "simple" cases work and therefore the implementation is
    //! logically correct.
    template<typename TData>
    struct RngWrapper<alpaka::Complex<TData>>
    {
        auto getMax()
        {
            return alpaka::Complex<TData>{TData{10}, TData{10}};
        }

        auto getLowest()
        {
            return -getMax();
        }

        auto getDistribution()
        {
            return std::uniform_real_distribution<TData>{0, 5};
        }

        template<typename TDistribution, typename TEngine>
        auto getNumber(TDistribution& distribution, TEngine& engine)
        {
            return alpaka::Complex<TData>{distribution(engine), distribution(engine)};
        }
    };

    /**
     * Fills buffer with random numbers (host-only).
     *
     * @tparam TData The used data-type (float, double, Complex<float> or Complex<double>).
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
        static_assert(TArgs::value_type::arity == TFunctor::arity, "Buffer properties must match TFunctor::arity");
        static_assert(TArgs::capacity > 6, "Set of args must provide > 6 entries.");
        auto rngWrapper = RngWrapper<TData>{};
        auto const max = rngWrapper.getMax();
        auto const low = rngWrapper.getLowest();
        std::default_random_engine eng{static_cast<std::default_random_engine::result_type>(seed)};

        // These pseudo-random numbers are implementation/platform specific!
        auto dist = rngWrapper.getDistribution();
        decltype(dist) distOne(-1, 1);
        for(size_t k = 0; k < TFunctor::arity_nr; ++k)
        {
            [[maybe_unused]] bool matchedSwitch = false;
            switch(functor.ranges[k])
            {
            case Range::OneNeighbourhood:
                matchedSwitch = true;
                for(size_t i = 0; i < TArgs::capacity; ++i)
                {
                    args(i).arg[k] = rngWrapper.getNumber(distOne, eng);
                }
                break;

            case Range::PositiveOnly:
                matchedSwitch = true;
                args(0).arg[k] = max;
                for(size_t i = 1; i < TArgs::capacity; ++i)
                {
                    args(i).arg[k] = rngWrapper.getNumber(dist, eng) + TData{1};
                }
                break;

            case Range::PositiveAndZero:
                matchedSwitch = true;
                args(0).arg[k] = TData{0};
                args(1).arg[k] = max;
                for(size_t i = 2; i < TArgs::capacity; ++i)
                {
                    args(i).arg[k] = rngWrapper.getNumber(dist, eng);
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
                        arg = rngWrapper.getNumber(dist, eng);
                    } while(std::equal_to<TData>()(arg, 1));
                    if(i % 2 == 0)
                        args(i).arg[k] = arg;
                    else
                        args(i).arg[k] = -arg;
                }
                break;

            case Range::Unrestricted:
                matchedSwitch = true;
                args(0).arg[k] = TData{0};
                args(1).arg[k] = max;
                args(2).arg[k] = low;
                for(size_t i = 3; i < TArgs::capacity; ++i)
                {
                    if(i % 2 == 0)
                        args(i).arg[k] = rngWrapper.getNumber(dist, eng);
                    else
                        args(i).arg[k] = -rngWrapper.getNumber(dist, eng);
                }
                break;

            case Range::Anything:
                matchedSwitch = true;
                args(0).arg[k] = TData{0};
                args(1).arg[k] = std::numeric_limits<TData>::quiet_NaN();
                args(2).arg[k] = std::numeric_limits<TData>::signaling_NaN();
                args(3).arg[k] = std::numeric_limits<TData>::infinity();
                args(4).arg[k] = -std::numeric_limits<TData>::infinity();
                constexpr size_t nFixed = 5;
                size_t i = nFixed;
                // no need to test for denormal for now: not supported by CUDA
                // for(; i < nFixed + (TArgs::capacity - nFixed) / 2; ++i)
                // {
                //     const TData v = rngWrapper.getNumber(dist, eng) *
                //     std::numeric_limits<TData>::denorm_min(); args(i).arg[k] = (i % 2 == 0) ? v : -v;
                // }
                for(; i < TArgs::capacity; ++i)
                {
                    TData const v = rngWrapper.getNumber(dist, eng);
                    args(i).arg[k] = (i % 2 == 0) ? v : -v;
                }
                break;
            }
            assert(matchedSwitch);
        }
    }
} // namespace mathtest
