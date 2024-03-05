/* Copyright 2022 Axel Huebl, Benjamin Worpitz, René Widera, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/rand/TinyMT/Engine.hpp"
#include "alpaka/rand/Traits.hpp"

#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>

namespace alpaka::rand
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

    namespace engine::cpu
    {
        //! The standard library mersenne twister random number generator.
        //!
        //! size of state: 19937 bytes
        class MersenneTwister
        {
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

            ALPAKA_FN_HOST static constexpr auto min() -> result_type
            {
                return std::mt19937::min();
            }

            ALPAKA_FN_HOST static constexpr auto max() -> result_type
            {
                return std::mt19937::max();
            }

            ALPAKA_FN_HOST auto operator()() -> result_type
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

            ALPAKA_FN_HOST static constexpr auto min() -> result_type
            {
                return TinyMTengine::min();
            }

            ALPAKA_FN_HOST static constexpr auto max() -> result_type
            {
                return TinyMTengine::max();
            }

            ALPAKA_FN_HOST auto operator()() -> result_type
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
            std::random_device state;

        public:
            RandomDevice() = default;

            ALPAKA_FN_HOST RandomDevice(std::uint32_t const&, std::uint32_t const& = 0, std::uint32_t const& = 0)
            {
            }

            // STL UniformRandomBitGenerator concept interface
            using result_type = std::random_device::result_type;

            ALPAKA_FN_HOST static constexpr auto min() -> result_type
            {
                return std::random_device::min();
            }

            ALPAKA_FN_HOST static constexpr auto max() -> result_type
            {
                return std::random_device::max();
            }

            ALPAKA_FN_HOST auto operator()() -> result_type
            {
                return state();
            }
        };
    } // namespace engine::cpu

    namespace distribution::cpu
    {
        //! The CPU random number normal distribution.
        template<typename T>
        struct NormalReal
        {
            template<typename TEngine>
            ALPAKA_FN_HOST auto operator()(TEngine& engine) -> T
            {
                return m_dist(engine);
            }

        private:
            std::normal_distribution<T> m_dist;
        };

        //! The CPU random number uniform distribution.
        template<typename T>
        struct UniformReal
        {
            template<typename TEngine>
            ALPAKA_FN_HOST auto operator()(TEngine& engine) -> T
            {
                return m_dist(engine);
            }

        private:
            std::uniform_real_distribution<T> m_dist;
        };

        //! The CPU random number normal distribution.
        template<typename T>
        struct UniformUint
        {
            template<typename TEngine>
            ALPAKA_FN_HOST auto operator()(TEngine& engine) -> T
            {
                return m_dist(engine);
            }

        private:
            std::uniform_int_distribution<T> m_dist{
                0, // For signed integer: std::numeric_limits<T>::lowest()
                std::numeric_limits<T>::max()};
        };
    } // namespace distribution::cpu

    namespace distribution::trait
    {
        //! The CPU device random number float normal distribution get trait specialization.
        template<typename T>
        struct CreateNormalReal<RandStdLib, T, std::enable_if_t<std::is_floating_point_v<T>>>
        {
            ALPAKA_FN_HOST static auto createNormalReal(RandStdLib const& /* rand */) -> cpu::NormalReal<T>
            {
                return {};
            }
        };

        //! The CPU device random number float uniform distribution get trait specialization.
        template<typename T>
        struct CreateUniformReal<RandStdLib, T, std::enable_if_t<std::is_floating_point_v<T>>>
        {
            ALPAKA_FN_HOST static auto createUniformReal(RandStdLib const& /* rand */) -> cpu::UniformReal<T>
            {
                return {};
            }
        };

        //! The CPU device random number integer uniform distribution get trait specialization.
        template<typename T>
        struct CreateUniformUint<RandStdLib, T, std::enable_if_t<std::is_integral_v<T>>>
        {
            ALPAKA_FN_HOST static auto createUniformUint(RandStdLib const& /* rand */) -> cpu::UniformUint<T>
            {
                return {};
            }
        };
    } // namespace distribution::trait

    namespace engine::trait
    {
        //! The CPU device random number default generator get trait specialization.
        template<>
        struct CreateDefault<TinyMersenneTwister>
        {
            ALPAKA_FN_HOST static auto createDefault(
                TinyMersenneTwister const& /* rand */,
                std::uint32_t const& seed = 0,
                std::uint32_t const& subsequence = 0,
                std::uint32_t const& offset = 0) -> cpu::TinyMersenneTwister
            {
                return {seed, subsequence, offset};
            }
        };

        template<>
        struct CreateDefault<MersenneTwister>
        {
            ALPAKA_FN_HOST static auto createDefault(
                MersenneTwister const& /* rand */,
                std::uint32_t const& seed = 0,
                std::uint32_t const& subsequence = 0,
                std::uint32_t const& offset = 0) -> cpu::MersenneTwister
            {
                return {seed, subsequence, offset};
            }
        };

        template<>
        struct CreateDefault<RandomDevice>
        {
            ALPAKA_FN_HOST static auto createDefault(
                RandomDevice const& /* rand */,
                std::uint32_t const& seed = 0,
                std::uint32_t const& subsequence = 0,
                std::uint32_t const& offset = 0) -> cpu::RandomDevice
            {
                return {seed, subsequence, offset};
            }
        };
    } // namespace engine::trait
} // namespace alpaka::rand
