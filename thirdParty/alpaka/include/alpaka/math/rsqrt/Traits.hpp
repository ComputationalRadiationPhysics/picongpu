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
            //! The rsqrt trait.
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Rsqrt;
        }

        //-----------------------------------------------------------------------------
        //! Computes the rsqrt.
        //!
        //! \tparam T The type of the object specializing Rsqrt.
        //! \tparam TArg The arg type.
        //! \param rsqrt The object specializing Rsqrt.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename TArg>
        ALPAKA_FN_HOST_ACC auto rsqrt(
            T const & rsqrt,
            TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Rsqrt<
                T,
                TArg>
            ::rsqrt(
                rsqrt,
                arg))
#endif
        {
            return
                traits::Rsqrt<
                    T,
                    TArg>
                ::rsqrt(
                    rsqrt,
                    arg);
        }

        namespace traits
        {
            //#############################################################################
            //! The Rsqrt specialization for classes with RsqrtBase member type.
            template<
                typename T,
                typename TArg>
            struct Rsqrt<
                T,
                TArg,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::RsqrtBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto rsqrt(
                    T const & rsqrt,
                    TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::rsqrt(
                        static_cast<typename T::RsqrtBase const &>(rsqrt),
                        arg))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::rsqrt(
                            static_cast<typename T::RsqrtBase const &>(rsqrt),
                            arg);
                }
            };
        }
    }
}
