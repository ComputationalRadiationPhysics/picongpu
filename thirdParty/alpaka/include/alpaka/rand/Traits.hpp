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

#include <alpaka/meta/IsStrictBase.hpp>

#include <alpaka/core/Common.hpp>

#include <boost/config.hpp>

#include <type_traits>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The random number generation specifics.
    namespace rand
    {
        //-----------------------------------------------------------------------------
        //! The random number generator distribution specifics.
        namespace distribution
        {
            //-----------------------------------------------------------------------------
            //! The random number generator distribution traits.
            namespace traits
            {
                //#############################################################################
                //! The random number float normal distribution get trait.
                template<
                    typename TRand,
                    typename T,
                    typename TSfinae = void>
                struct CreateNormalReal;

                //#############################################################################
                //! The random number float uniform distribution get trait.
                template<
                    typename TRand,
                    typename T,
                    typename TSfinae = void>
                struct CreateUniformReal;

                //#############################################################################
                //! The random number integer uniform distribution get trait.
                template<
                    typename TRand,
                    typename T,
                    typename TSfinae = void>
                struct CreateUniformUint;
            }

            //-----------------------------------------------------------------------------
            //! \return A normal float distribution with mean 0.0f and standard deviation 1.0f.
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename T,
                typename TRand>
            ALPAKA_FN_HOST_ACC auto createNormalReal(
                TRand const & rand)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
            -> decltype(
                traits::CreateNormalReal<
                    TRand,
                    T>
                ::createNormalReal(
                    rand))
#endif
            {
                static_assert(
                    std::is_floating_point<T>::value,
                    "The value type T has to be a floating point type!");

                return
                    traits::CreateNormalReal<
                        TRand,
                        T>
                    ::createNormalReal(
                        rand);
            }
            //-----------------------------------------------------------------------------
            //! \return A uniform floating point distribution [0.0, 1.0).
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename T,
                typename TRand>
            ALPAKA_FN_HOST_ACC auto createUniformReal(
                TRand const & rand)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
            -> decltype(
                traits::CreateUniformReal<
                    TRand,
                    T>
                ::createUniformReal(
                    rand))
#endif
            {
                static_assert(
                    std::is_floating_point<T>::value,
                    "The value type T has to be a floating point type!");

                return
                    traits::CreateUniformReal<
                        TRand,
                        T>
                    ::createUniformReal(
                        rand);
            }
            //-----------------------------------------------------------------------------
            //! \return A uniform integer distribution [0, UINT_MAX].
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename T,
                typename TRand>
            ALPAKA_FN_HOST_ACC auto createUniformUint(
                TRand const & rand)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
            -> decltype(
                traits::CreateUniformUint<
                    TRand,
                    T>
                ::createUniformUint(
                    rand))
#endif
            {
                static_assert(
                    std::is_integral<T>::value && std::is_unsigned<T>::value,
                    "The value type T has to be a unsigned integral type!");

                return
                    traits::CreateUniformUint<
                        TRand,
                        T>
                    ::createUniformUint(
                        rand);
            }
            namespace traits
            {
                //#############################################################################
                //! The CreateNormalReal specialization for classes with RandBase member type.
                template<
                    typename TRand,
                    typename T>
                struct CreateNormalReal<
                    TRand,
                    T,
                    typename std::enable_if<
                        std::is_base_of<typename TRand::RandBase, typename std::decay<TRand>::type>::value
                        && (!std::is_same<typename TRand::RandBase, typename std::decay<TRand>::type>::value)>::type>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC static auto createNormalReal(
                        TRand const & rand)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                    -> decltype(
                        rand::distribution::createNormalReal<T>(
                            static_cast<typename TRand::RandBase const &>(rand)))
#endif
                    {
                        // Delegate the call to the base class.
                        return
                            rand::distribution::createNormalReal<T>(
                                static_cast<typename TRand::RandBase const &>(rand));
                    }
                };
                //#############################################################################
                //! The CreateUniformReal specialization for classes with RandBase member type.
                template<
                    typename TRand,
                    typename T>
                struct CreateUniformReal<
                    TRand,
                    T,
                    typename std::enable_if<
                        std::is_base_of<typename TRand::RandBase, typename std::decay<TRand>::type>::value
                        && (!std::is_same<typename TRand::RandBase, typename std::decay<TRand>::type>::value)>::type>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC static auto createUniformReal(
                        TRand const & rand)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                    -> decltype(
                        rand::distribution::createUniformReal<T>(
                            static_cast<typename TRand::RandBase const &>(rand)))
#endif
                    {
                        // Delegate the call to the base class.
                        return
                            rand::distribution::createUniformReal<T>(
                                static_cast<typename TRand::RandBase const &>(rand));
                    }
                };
                //#############################################################################
                //! The CreateUniformUint specialization for classes with RandBase member type.
                template<
                    typename TRand,
                    typename T>
                struct CreateUniformUint<
                    TRand,
                    T,
                    typename std::enable_if<
                        std::is_base_of<typename TRand::RandBase, typename std::decay<TRand>::type>::value
                        && (!std::is_same<typename TRand::RandBase, typename std::decay<TRand>::type>::value)>::type>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC static auto createUniformUint(
                        TRand const & rand)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                    -> decltype(
                        rand::distribution::createUniformUint<T>(
                            static_cast<typename TRand::RandBase const &>(rand)))
#endif
                    {
                        // Delegate the call to the base class.
                        return
                            rand::distribution::createUniformUint<T>(
                                static_cast<typename TRand::RandBase const &>(rand));
                    }
                };
            }
        }
        //-----------------------------------------------------------------------------
        //! The random number generator specifics.
        namespace generator
        {
            //-----------------------------------------------------------------------------
            //! The random number generator traits.
            namespace traits
            {
                //#############################################################################
                //! The random number default generator get trait.
                template<
                    typename TRand,
                    typename TSfinae = void>
                struct CreateDefault;
            }
            //-----------------------------------------------------------------------------
            //! \return A default random number generator.
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TRand>
            ALPAKA_FN_HOST_ACC auto createDefault(
                TRand const & rand,
                std::uint32_t const & seed,
                std::uint32_t const & subsequence)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
            -> decltype(
                traits::CreateDefault<
                    TRand>
                ::createDefault(
                    rand,
                    seed,
                    subsequence))
#endif
            {
                return
                    traits::CreateDefault<
                        TRand>
                    ::createDefault(
                        rand,
                        seed,
                        subsequence);
            }
            namespace traits
            {
                //#############################################################################
                //! The CreateDefault specialization for classes with RandBase member type.
                template<
                    typename TRand>
                struct CreateDefault<
                    TRand,
                    typename std::enable_if<
                        meta::IsStrictBase<
                            typename TRand::RandBase,
                            TRand
                        >::value
                    >::type>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC static auto createDefault(
                        TRand const & rand,
                        std::uint32_t const & seed,
                        std::uint32_t const & subsequence)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                    -> decltype(
                        rand::generator::createDefault(
                            static_cast<typename TRand::RandBase const &>(rand),
                            seed,
                            subsequence))
#endif
                    {
                        // Delegate the call to the base class.
                        return
                            rand::generator::createDefault(
                                static_cast<typename TRand::RandBase const &>(rand),
                                seed,
                                subsequence);
                    }
                };
            }
        }
    }
}
