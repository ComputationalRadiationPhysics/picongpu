/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
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
    namespace math
    {
        namespace traits
        {
            //#############################################################################
            //! The round trait.
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Round;

            //#############################################################################
            //! The round trait.
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Lround;

            //#############################################################################
            //! The round trait.
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Llround;
        }

        //-----------------------------------------------------------------------------
        //! Computes the nearest integer value to arg (in floating-point format), rounding halfway cases away from zero, regardless of the current rounding mode.
        //!
        //! \tparam T The type of the object specializing Round.
        //! \tparam TArg The arg type.
        //! \param round The object specializing Round.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename TArg>
        ALPAKA_FN_HOST_ACC auto round(
            T const & round,
            TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Round<
                T,
                TArg>
            ::round(
                round,
                arg))
#endif
        {
            return
                traits::Round<
                    T,
                    TArg>
                ::round(
                    round,
                    arg);
        }
        //-----------------------------------------------------------------------------
        //! Computes the nearest integer value to arg (in integer format), rounding halfway cases away from zero, regardless of the current rounding mode.
        //!
        //! \tparam T The type of the object specializing Round.
        //! \tparam TArg The arg type.
        //! \param lround The object specializing Round.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename TArg>
        ALPAKA_FN_HOST_ACC auto lround(
            T const & lround,
            TArg const & arg)
        -> long int
        {
            return
                traits::Lround<
                    T,
                    TArg>
                ::lround(
                    lround,
                    arg);
        }
        //-----------------------------------------------------------------------------
        //! Computes the nearest integer value to arg (in integer format), rounding halfway cases away from zero, regardless of the current rounding mode.
        //!
        //! \tparam T The type of the object specializing Round.
        //! \tparam TArg The arg type.
        //! \param llround The object specializing Round.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename TArg>
        ALPAKA_FN_HOST_ACC auto llround(
            T const & llround,
            TArg const & arg)
        -> long long int
        {
            return
                traits::Llround<
                    T,
                    TArg>
                ::llround(
                    llround,
                    arg);
        }

        namespace traits
        {
            //#############################################################################
            //! The Round specialization for classes with RoundBase member type.
            template<
                typename T,
                typename TArg>
            struct Round<
                T,
                TArg,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::RoundBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto round(
                    T const & round,
                    TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::round(
                        static_cast<typename T::RoundBase const &>(round),
                        arg))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::round(
                            static_cast<typename T::RoundBase const &>(round),
                            arg);
                }
            };
            //#############################################################################
            //! The Lround specialization for classes with RoundBase member type.
            template<
                typename T,
                typename TArg>
            struct Lround<
                T,
                TArg,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::RoundBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto lround(
                    T const & lround,
                    TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::lround(
                        static_cast<typename T::RoundBase const &>(lround),
                        arg))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::lround(
                            static_cast<typename T::RoundBase const &>(lround),
                            arg);
                }
            };
            //#############################################################################
            //! The Llround specialization for classes with RoundBase member type.
            template<
                typename T,
                typename TArg>
            struct Llround<
                T,
                TArg,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::RoundBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto llround(
                    T const & llround,
                    TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::llround(
                        static_cast<typename T::RoundBase const &>(llround),
                        arg))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::llround(
                            static_cast<typename T::RoundBase const &>(llround),
                            arg);
                }
            };
        }
    }
}
