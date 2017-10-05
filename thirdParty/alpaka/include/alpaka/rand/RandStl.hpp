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

#include <alpaka/rand/Traits.hpp>       // CreateNormalReal, ...

#include <alpaka/core/Common.hpp>       // ALPAKA_FN_*

#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

#include <random>                       // std::mt19937, std::uniform_real_distribution, ...
#include <type_traits>                  // std::enable_if

namespace alpaka
{
    namespace rand
    {
        //#############################################################################
        //! The standard library rand implementation.
        //#############################################################################
        class RandStl
        {
        public:
            using RandBase = RandStl;
        };

        namespace distribution
        {
            namespace traits
            {
                //#############################################################################
                //! The CPU device random number float normal distribution get trait specialization.
                //#############################################################################
                template<
                    typename T>
                struct CreateNormalReal<
                    RandStl,
                    T,
                    typename std::enable_if<
                        std::is_floating_point<T>::value>::type>
                {
                    //-----------------------------------------------------------------------------
                    //
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA static auto createNormalReal(
                        RandStl const & rand)
                    -> std::normal_distribution<T>
                    {
                        boost::ignore_unused(rand);
                        return std::normal_distribution<T>();
                    }
                };
                //#############################################################################
                //! The CPU device random number float uniform distribution get trait specialization.
                //#############################################################################
                template<
                    typename T>
                struct CreateUniformReal<
                    RandStl,
                    T,
                    typename std::enable_if<
                        std::is_floating_point<T>::value>::type>
                {
                    //-----------------------------------------------------------------------------
                    //
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA static auto createUniformReal(
                        RandStl const & rand)
                    -> std::uniform_real_distribution<T>
                    {
                        boost::ignore_unused(rand);
                        return std::uniform_real_distribution<T>();
                    }
                };
                //#############################################################################
                //! The CPU device random number integer uniform distribution get trait specialization.
                //#############################################################################
                template<
                    typename T>
                struct CreateUniformUint<
                    RandStl,
                    T,
                    typename std::enable_if<
                        std::is_integral<T>::value>::type>
                {
                    //-----------------------------------------------------------------------------
                    //
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA static auto createUniformUint(
                        RandStl const & rand)
                    -> std::uniform_int_distribution<T>
                    {
                        boost::ignore_unused(rand);
                        return std::uniform_int_distribution<T>(
                            0,  // For signed integer: std::numeric_limits<T>::lowest()
                            std::numeric_limits<T>::max());
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
                //#############################################################################
                template<>
                struct CreateDefault<
                    RandStl>
                {
                    //-----------------------------------------------------------------------------
                    //
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA static auto createDefault(
                        RandStl const & rand,
                        std::uint32_t const & seed,
                        std::uint32_t const & subsequence)
                    -> std::mt19937
                    {
                        boost::ignore_unused(rand);
                        // NOTE: XOR the seed and the subsequence to generate a unique seed.
                        return std::mt19937(seed ^ subsequence);
                    }
                };
            }
        }
    }
}
