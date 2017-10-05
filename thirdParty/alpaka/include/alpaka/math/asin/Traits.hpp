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

#include <alpaka/meta/IsStrictBase.hpp> // meta::IsStrictBase

#include <alpaka/core/Common.hpp>       // ALPAKA_FN_*

#include <boost/config.hpp>             // BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION

#include <type_traits>                  // std::enable_if

namespace alpaka
{
    namespace math
    {
        namespace traits
        {
            //#############################################################################
            //! The asin trait.
            //#############################################################################
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Asin;
        }

        //-----------------------------------------------------------------------------
        //! Computes the principal value of the arc sine.
        //!
        //! \tparam TArg The arg type.
        //! \param asin The object specializing Asin.
        //! \param arg The arg.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename TArg>
        ALPAKA_FN_HOST_ACC auto asin(
            T const & asin,
            TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Asin<
                T,
                TArg>
            ::asin(
                asin,
                arg))
#endif
        {
            return
                traits::Asin<
                    T,
                    TArg>
                ::asin(
                    asin,
                    arg);
        }

        namespace traits
        {
            //#############################################################################
            //! The Asin specialization for classes with AsinBase member type.
            //#############################################################################
            template<
                typename T,
                typename TArg>
            struct Asin<
                T,
                TArg,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::AsinBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto asin(
                    T const & asin,
                    TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::asin(
                        static_cast<typename T::AsinBase const &>(asin),
                        arg))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::asin(
                            static_cast<typename T::AsinBase const &>(asin),
                            arg);
                }
            };
        }
    }
}
