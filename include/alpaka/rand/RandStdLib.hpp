/**
* \file
* Copyright 2015-2018 Benjamin Worpitz, Axel Huebl
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/rand/Traits.hpp>
#include <alpaka/rand/TinyMT/Engine.hpp>

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>

#include <cstdint>
#include <random>
#include <type_traits>

namespace alpaka
{
    namespace rand
    {
        //#############################################################################
        //! "Tiny" state mersenne twister implementation
        class TinyMersenneTwister
        {
        public:
            using RandBase = TinyMersenneTwister;
        };
        using RandStdLib = TinyMersenneTwister;

        //#############################################################################
        //! The standard library mersenne twister implementation.
        class MersenneTwister
        {
        public:
            using RandBase = MersenneTwister;
        };

        //#############################################################################
        //! The standard library rand device implementation.
        class RandomDevice
        {
        public:
            using RandBase = RandomDevice;
        };

        namespace generator
        {
            namespace cpu
            {
                //#############################################################################
                //! The standard library mersenne twister random number generator.
                //!
                //! size of state: 19937 bytes
                class MersenneTwister
                {
                public:

                    //-----------------------------------------------------------------------------
                    MersenneTwister() = default;

                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST MersenneTwister(
                        std::uint32_t const & seed,
                        std::uint32_t const & subsequence = 0,
                        std::uint32_t const & offset = 0) :
                        // NOTE: XOR the seed and the subsequence to generate a unique seed.
                        m_State((seed ^ subsequence) + offset)
                    {
                    }

                public:
                    std::mt19937 m_State;
                };

                //#############################################################################
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
                    //-----------------------------------------------------------------------------
                    TinyMersenneTwister() = default;

                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST TinyMersenneTwister(
                        std::uint32_t const & seed,
                        std::uint32_t const & subsequence = 0,
                        std::uint32_t const & offset = 0) :
                        // NOTE: XOR the seed and the subsequence to generate a unique seed.
                        m_State((seed ^ subsequence) + offset)
                    {
                    }

                public:
                    TinyMTengine m_State;
                };

                //#############################################################################
                //! The standard library's random device based on the local entropy pool.
                //!
                //! Warning: the entropy pool on many devices degrates quickly and performance
                //!          will drop significantly when this point occures.
                //!
                //! size of state: 1 byte
                class RandomDevice
                {
                public:
                    //-----------------------------------------------------------------------------
                    RandomDevice() = default;
                    RandomDevice(RandomDevice&&) :
                        m_State{}
                    {
                    }

                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST RandomDevice(
                        std::uint32_t const &,
                        std::uint32_t const & = 0,
                        std::uint32_t const & = 0) :
                        m_State{}
                    {
                    }

                public:
                    std::random_device m_State;
                };
            }
        }

        namespace distribution
        {
            namespace cpu
            {
                //#############################################################################
                //! The CPU random number normal distribution.
                template<
                    typename T>
                class NormalReal
                {
                public:
                    //-----------------------------------------------------------------------------
                    NormalReal() = default;

                    //-----------------------------------------------------------------------------
                    template<
                        typename TGenerator>
                    ALPAKA_FN_HOST auto operator()(
                        TGenerator & generator)
                    -> T
                    {
                        return m_dist(generator.m_State);
                    }
                    std::normal_distribution<T> m_dist;
                };

                //#############################################################################
                //! The CPU random number uniform distribution.
                template<
                    typename T>
                class UniformReal
                {
                public:
                    //-----------------------------------------------------------------------------
                    UniformReal() = default;

                    //-----------------------------------------------------------------------------
                    template<
                        typename TGenerator>
                    ALPAKA_FN_HOST auto operator()(
                        TGenerator & generator)
                    -> T
                    {
                        return m_dist(generator.m_State);
                    }
                    std::uniform_real_distribution<T> m_dist;
                };

                //#############################################################################
                //! The CPU random number normal distribution.
                template<
                    typename T>
                class UniformUint
                {
                public:
                    //-----------------------------------------------------------------------------
                    UniformUint() :
                        m_dist(
                            0,  // For signed integer: std::numeric_limits<T>::lowest()
                            std::numeric_limits<T>::max())
                    {}

                    //-----------------------------------------------------------------------------
                    template<
                        typename TGenerator>
                    ALPAKA_FN_HOST auto operator()(
                        TGenerator & generator)
                    -> T
                    {
                        return m_dist(generator.m_State);
                    }
                    std::uniform_int_distribution<T> m_dist;
                };
            }
        }

        namespace distribution
        {
            namespace traits
            {
                //#############################################################################
                //! The CPU device random number float normal distribution get trait specialization.
                template<
                    typename T>
                struct CreateNormalReal<
                    RandStdLib,
                    T,
                    typename std::enable_if<
                        std::is_floating_point<T>::value>::type>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto createNormalReal(
                        RandStdLib const & rand)
                    -> rand::distribution::cpu::NormalReal<T>
                    {
                        alpaka::ignore_unused(rand);
                        return rand::distribution::cpu::NormalReal<T>();
                    }
                };
                //#############################################################################
                //! The CPU device random number float uniform distribution get trait specialization.
                template<
                    typename T>
                struct CreateUniformReal<
                    RandStdLib,
                    T,
                    typename std::enable_if<
                        std::is_floating_point<T>::value>::type>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto createUniformReal(
                        RandStdLib const & rand)
                    -> rand::distribution::cpu::UniformReal<T>
                    {
                        alpaka::ignore_unused(rand);
                        return rand::distribution::cpu::UniformReal<T>();
                    }
                };
                //#############################################################################
                //! The CPU device random number integer uniform distribution get trait specialization.
                template<
                    typename T>
                struct CreateUniformUint<
                    RandStdLib,
                    T,
                    typename std::enable_if<
                        std::is_integral<T>::value>::type>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto createUniformUint(
                        RandStdLib const & rand)
                    -> rand::distribution::cpu::UniformUint<T>
                    {
                        alpaka::ignore_unused(rand);
                        return rand::distribution::cpu::UniformUint<T>();
                    }
                };
            }
        }
        namespace generator
        {
            namespace traits
            {
                //#############################################################################
                //! The CPU device random number default generator get trait specialization.
                template<>
                struct CreateDefault<
                    TinyMersenneTwister>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto createDefault(
                        TinyMersenneTwister const & rand,
                        std::uint32_t const & seed,
                        std::uint32_t const & subsequence)
                    -> rand::generator::cpu::TinyMersenneTwister
                    {
                        alpaka::ignore_unused(rand);
                        return rand::generator::cpu::TinyMersenneTwister(
                            seed,
                            subsequence);
                    }
                };

                template<>
                struct CreateDefault<
                    MersenneTwister>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto createDefault(
                        MersenneTwister const & rand,
                        std::uint32_t const & seed,
                        std::uint32_t const & subsequence)
                    -> rand::generator::cpu::MersenneTwister
                    {
                        alpaka::ignore_unused(rand);
                        return rand::generator::cpu::MersenneTwister(
                            seed,
                            subsequence);
                    }
                };

                template<>
                struct CreateDefault<
                    RandomDevice>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto createDefault(
                        RandomDevice const & rand,
                        std::uint32_t const & seed,
                        std::uint32_t const & subsequence)
                    -> rand::generator::cpu::RandomDevice
                    {
                        alpaka::ignore_unused(rand);
                        return rand::generator::cpu::RandomDevice(
                            seed,
                            subsequence);
                    }
                };
            }
        }
    }
}
