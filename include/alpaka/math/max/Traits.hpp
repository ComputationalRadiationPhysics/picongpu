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
            //! The max trait.
            template<
                typename T,
                typename Tx,
                typename Ty,
                typename TSfinae = void>
            struct Max;
        }

        //-----------------------------------------------------------------------------
        //! Returns the larger of two arguments.
        //! NaNs are treated as missing data (between a NaN and a numeric value, the numeric value is chosen).
        //!
        //! \tparam T The type of the object specializing Max.
        //! \tparam Tx The type of the first argument.
        //! \tparam Ty The type of the second argument.
        //! \param max The object specializing Max.
        //! \param x The first argument.
        //! \param y The second argument.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename Tx,
            typename Ty>
        ALPAKA_FN_HOST_ACC auto max(
            T const & max,
            Tx const & x,
            Ty const & y)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Max<
                T,
                Tx,
                Ty>
            ::max(
                max,
                x,
                y))
#endif
        {
            return
                traits::Max<
                    T,
                    Tx,
                    Ty>
                ::max(
                    max,
                    x,
                    y);
        }

        namespace traits
        {
            //#############################################################################
            //! The Max specialization for classes with MaxBase member type.
            template<
                typename T,
                typename Tx,
                typename Ty>
            struct Max<
                T,
                Tx,
                Ty,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::MaxBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto max(
                    T const & max,
                    Tx const & x,
                    Ty const & y)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::max(
                        static_cast<typename T::MaxBase const &>(max),
                        x,
                        y))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::max(
                            static_cast<typename T::MaxBase const &>(max),
                            x,
                            y);
                }
            };
        }
    }
}
