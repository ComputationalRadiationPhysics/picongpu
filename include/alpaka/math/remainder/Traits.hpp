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
            //! The remainder trait.
            template<
                typename T,
                typename Tx,
                typename Ty,
                typename TSfinae = void>
            struct Remainder;
        }

        //-----------------------------------------------------------------------------
        //! Computes the IEEE remainder of the floating point division operation x/y.
        //!
        //! \tparam T The type of the object specializing Remainder.
        //! \tparam Tx The type of the first argument.
        //! \tparam Ty The type of the second argument.
        //! \param remainder The object specializing Max.
        //! \param x The first argument.
        //! \param y The second argument.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename Tx,
            typename Ty>
        ALPAKA_FN_HOST_ACC auto remainder(
            T const & remainder,
            Tx const & x,
            Ty const & y)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Remainder<
                T,
                Tx,
                Ty>
            ::remainder(
                remainder,
                x,
                y))
#endif
        {
            return
                traits::Remainder<
                    T,
                    Tx,
                    Ty>
                ::remainder(
                    remainder,
                    x,
                    y);
        }

        namespace traits
        {
            //#############################################################################
            //! The Remainder specialization for classes with RemainderBase member type.
            template<
                typename T,
                typename Tx,
                typename Ty>
            struct Remainder<
                T,
                Tx,
                Ty,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::RemainderBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto remainder(
                    T const & remainder,
                    Tx const & x,
                    Ty const & y)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::remainder(
                        static_cast<typename T::RemainderBase const &>(remainder),
                        x,
                        y))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::remainder(
                            static_cast<typename T::RemainderBase const &>(remainder),
                            x,
                            y);
                }
            };
        }
    }
}
