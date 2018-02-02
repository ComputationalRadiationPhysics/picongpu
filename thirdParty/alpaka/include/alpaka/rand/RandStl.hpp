/**
* \file
* Copyright 2015 Benjamin Worpitz
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

#include <alpaka/core/Common.hpp>

#include <boost/core/ignore_unused.hpp>

#include <random>
#include <type_traits>

namespace alpaka
{
    namespace rand
    {
        //#############################################################################
        //! The standard library rand implementation.
        class RandStl
        {
        public:
            using RandBase = RandStl;
        };

        namespace generator
        {
            namespace cpu
            {
                //#############################################################################
                //! The STL mersenne twister random number generator.
                class MersenneTwister
                {
                public:

                    //-----------------------------------------------------------------------------
                    MersenneTwister() = default;

                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA MersenneTwister(
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
                    ALPAKA_FN_ACC_NO_CUDA auto operator()(
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
                    ALPAKA_FN_ACC_NO_CUDA auto operator()(
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
                    ALPAKA_FN_ACC_NO_CUDA auto operator()(
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
                    RandStl,
                    T,
                    typename std::enable_if<
                        std::is_floating_point<T>::value>::type>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA static auto createNormalReal(
                        RandStl const & rand)
                    -> rand::distribution::cpu::NormalReal<T>
                    {
                        boost::ignore_unused(rand);
                        return rand::distribution::cpu::NormalReal<T>();
                    }
                };
                //#############################################################################
                //! The CPU device random number float uniform distribution get trait specialization.
                template<
                    typename T>
                struct CreateUniformReal<
                    RandStl,
                    T,
                    typename std::enable_if<
                        std::is_floating_point<T>::value>::type>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA static auto createUniformReal(
                        RandStl const & rand)
                    -> rand::distribution::cpu::UniformReal<T>
                    {
                        boost::ignore_unused(rand);
                        return rand::distribution::cpu::UniformReal<T>();
                    }
                };
                //#############################################################################
                //! The CPU device random number integer uniform distribution get trait specialization.
                template<
                    typename T>
                struct CreateUniformUint<
                    RandStl,
                    T,
                    typename std::enable_if<
                        std::is_integral<T>::value>::type>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA static auto createUniformUint(
                        RandStl const & rand)
                    -> rand::distribution::cpu::UniformUint<T>
                    {
                        boost::ignore_unused(rand);
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
                    RandStl>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA static auto createDefault(
                        RandStl const & rand,
                        std::uint32_t const & seed,
                        std::uint32_t const & subsequence)
                    -> rand::generator::cpu::MersenneTwister
                    {
                        boost::ignore_unused(rand);
                        return rand::generator::cpu::MersenneTwister(
                            seed,
                            subsequence);
                    }
                };
            }
        }
    }
}
