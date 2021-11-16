/* Copyright 2019 Axel Huebl, Benjamin Worpitz, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/rand/TinyMT/Engine.hpp>
#include <alpaka/rand/Traits.hpp>

#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>

namespace alpaka
{
    namespace rand
    {
        //! "Tiny" state mersenne twister implementation
        class TinyMersenneTwister : public concepts::Implements<ConceptRand, TinyMersenneTwister>
        {
        };
        using RandStdLib = TinyMersenneTwister;

        //! The standard library mersenne twister implementation.
        class MersenneTwister : public concepts::Implements<ConceptRand, MersenneTwister>
        {
        };

        //! The standard library rand device implementation.
        class RandomDevice : public concepts::Implements<ConceptRand, RandomDevice>
        {
        };

        namespace engine
        {
            namespace cpu
            {
                //! The standard library mersenne twister random number generator.
                //!
                //! size of state: 19937 bytes
                class MersenneTwister
                {
                public:
                    std::mt19937 state;

                public:
                    MersenneTwister() = default;

                    ALPAKA_FN_HOST MersenneTwister(
                        std::uint32_t const& seed,
                        std::uint32_t const& subsequence = 0,
                        std::uint32_t const& offset = 0)
                        : // NOTE: XOR the seed and the subsequence to generate a unique seed.
                        state((seed ^ subsequence) + offset)
                    {
                    }

                    // STL UniformRandomBitGenerator concept interface
                    using result_type = std::mt19937::result_type;
                    ALPAKA_FN_HOST constexpr static result_type min()
                    {
                        return std::mt19937::min();
                    }
                    ALPAKA_FN_HOST constexpr static result_type max()
                    {
                        return std::mt19937::max();
                    }
                    ALPAKA_FN_HOST result_type operator()()
                    {
                        return state();
                    }
                };

                //! "Tiny" state mersenne twister implementation
                //!
                //! repository: github.com/MersenneTwister-Lab/TinyMT
                //!
                //! license: 3-clause BSD
                //!
                //! @author Mutsuo Saito (Hiroshima University)Tokio University.
                //! @author Makoto Matsumoto (The University of Tokyo)
                //!
                //! size of state: 28 bytes (127 bits?!)
                class TinyMersenneTwister
                {
                public:
                    TinyMTengine state;

                public:
                    TinyMersenneTwister() = default;

                    ALPAKA_FN_HOST TinyMersenneTwister(
                        std::uint32_t const& seed,
                        std::uint32_t const& subsequence = 0,
                        std::uint32_t const& offset = 0)
                        : // NOTE: XOR the seed and the subsequence to generate a unique seed.
                        state((seed ^ subsequence) + offset)
                    {
                    }

                    // STL UniformRandomBitGenerator concept interface
                    using result_type = TinyMTengine::result_type;
                    ALPAKA_FN_HOST constexpr static result_type min()
                    {
                        return TinyMTengine::min();
                    }
                    ALPAKA_FN_HOST constexpr static result_type max()
                    {
                        return TinyMTengine::max();
                    }
                    ALPAKA_FN_HOST result_type operator()()
                    {
                        return state();
                    }
                };

                //! The standard library's random device based on the local entropy pool.
                //!
                //! Warning: the entropy pool on many devices degrates quickly and performance
                //!          will drop significantly when this point occures.
                //!
                //! size of state: 1 byte
                class RandomDevice
                {
                public:
                    std::random_device state;

                public:
                    RandomDevice() = default;
                    RandomDevice(RandomDevice&&) : state{}
                    {
                    }

                    ALPAKA_FN_HOST RandomDevice(
                        std::uint32_t const&,
                        std::uint32_t const& = 0,
                        std::uint32_t const& = 0)
                        : state{}
                    {
                    }

                    // STL UniformRandomBitGenerator concept interface
                    using result_type = std::random_device::result_type;
                    ALPAKA_FN_HOST constexpr static result_type min()
                    {
                        return std::random_device::min();
                    }
                    ALPAKA_FN_HOST constexpr static result_type max()
                    {
                        return std::random_device::max();
                    }
                    ALPAKA_FN_HOST result_type operator()()
                    {
                        return state();
                    }
                };
            } // namespace cpu
        } // namespace engine

        namespace distribution
        {
            namespace cpu
            {
                //! The CPU random number normal distribution.
                template<typename T>
                class NormalReal
                {
                public:
                    template<typename TEngine>
                    ALPAKA_FN_HOST auto operator()(TEngine& engine) -> T
                    {
                        return m_dist(engine);
                    }
                    std::normal_distribution<T> m_dist;
                };

                //! The CPU random number uniform distribution.
                template<typename T>
                class UniformReal
                {
                public:
                    template<typename TEngine>
                    ALPAKA_FN_HOST auto operator()(TEngine& engine) -> T
                    {
                        return m_dist(engine);
                    }
                    std::uniform_real_distribution<T> m_dist;
                };

                //! The CPU random number normal distribution.
                template<typename T>
                class UniformUint
                {
                public:
                    template<typename TEngine>
                    ALPAKA_FN_HOST auto operator()(TEngine& engine) -> T
                    {
                        return m_dist(engine);
                    }
                    std::uniform_int_distribution<T> m_dist{
                        0, // For signed integer: std::numeric_limits<T>::lowest()
                        std::numeric_limits<T>::max()};
                };
            } // namespace cpu
        } // namespace distribution

        namespace distribution
        {
            namespace traits
            {
                //! The CPU device random number float normal distribution get trait specialization.
                template<typename T>
                struct CreateNormalReal<RandStdLib, T, std::enable_if_t<std::is_floating_point<T>::value>>
                {
                    ALPAKA_FN_HOST static auto createNormalReal(RandStdLib const& rand)
                        -> rand::distribution::cpu::NormalReal<T>
                    {
                        alpaka::ignore_unused(rand);
                        return rand::distribution::cpu::NormalReal<T>();
                    }
                };
                //! The CPU device random number float uniform distribution get trait specialization.
                template<typename T>
                struct CreateUniformReal<RandStdLib, T, std::enable_if_t<std::is_floating_point<T>::value>>
                {
                    ALPAKA_FN_HOST static auto createUniformReal(RandStdLib const& rand)
                        -> rand::distribution::cpu::UniformReal<T>
                    {
                        alpaka::ignore_unused(rand);
                        return rand::distribution::cpu::UniformReal<T>();
                    }
                };
                //! The CPU device random number integer uniform distribution get trait specialization.
                template<typename T>
                struct CreateUniformUint<RandStdLib, T, std::enable_if_t<std::is_integral<T>::value>>
                {
                    ALPAKA_FN_HOST static auto createUniformUint(RandStdLib const& rand)
                        -> rand::distribution::cpu::UniformUint<T>
                    {
                        alpaka::ignore_unused(rand);
                        return rand::distribution::cpu::UniformUint<T>();
                    }
                };
            } // namespace traits
        } // namespace distribution
        namespace engine
        {
            namespace traits
            {
                //! The CPU device random number default generator get trait specialization.
                template<>
                struct CreateDefault<TinyMersenneTwister>
                {
                    ALPAKA_FN_HOST static auto createDefault(
                        TinyMersenneTwister const& rand,
                        std::uint32_t const& seed = 0,
                        std::uint32_t const& subsequence = 0,
                        std::uint32_t const& offset = 0) -> rand::engine::cpu::TinyMersenneTwister
                    {
                        alpaka::ignore_unused(rand);
                        return rand::engine::cpu::TinyMersenneTwister(seed, subsequence, offset);
                    }
                };

                template<>
                struct CreateDefault<MersenneTwister>
                {
                    ALPAKA_FN_HOST static auto createDefault(
                        MersenneTwister const& rand,
                        std::uint32_t const& seed = 0,
                        std::uint32_t const& subsequence = 0,
                        std::uint32_t const& offset = 0) -> rand::engine::cpu::MersenneTwister
                    {
                        alpaka::ignore_unused(rand);
                        return rand::engine::cpu::MersenneTwister(seed, subsequence, offset);
                    }
                };

                template<>
                struct CreateDefault<RandomDevice>
                {
                    ALPAKA_FN_HOST static auto createDefault(
                        RandomDevice const& rand,
                        std::uint32_t const& seed = 0,
                        std::uint32_t const& subsequence = 0,
                        std::uint32_t const& offset = 0) -> rand::engine::cpu::RandomDevice
                    {
                        alpaka::ignore_unused(rand);
                        return rand::engine::cpu::RandomDevice(seed, subsequence, offset);
                    }
                };
            } // namespace traits
        } // namespace engine
    } // namespace rand
} // namespace alpaka
