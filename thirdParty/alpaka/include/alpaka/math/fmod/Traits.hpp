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
            //! The fmod trait.
            template<
                typename T,
                typename Tx,
                typename Ty,
                typename TSfinae = void>
            struct Fmod;
        }

        //-----------------------------------------------------------------------------
        //! Computes the floating-point remainder of the division operation x/y.
        //!
        //! \tparam T The type of the object specializing Fmod.
        //! \tparam Tx The type of the first argument.
        //! \tparam Ty The type of the second argument.
        //! \param fmod The object specializing Fmod.
        //! \param x The first argument.
        //! \param y The second argument.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename Tx,
            typename Ty>
        ALPAKA_FN_HOST_ACC auto fmod(
            T const & fmod,
            Tx const & x,
            Ty const & y)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Fmod<
                T,
                Tx,
                Ty>
            ::fmod(
                fmod,
                x,
                y))
#endif
        {
            return
                traits::Fmod<
                    T,
                    Tx,
                    Ty>
                ::fmod(
                    fmod,
                    x,
                    y);
        }

        namespace traits
        {
            //#############################################################################
            //! The Fmod specialization for classes with FmodBase member type.
            template<
                typename T,
                typename TArg>
            struct Fmod<
                T,
                TArg,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::FmodBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto fmod(
                    T const & fmod,
                    TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::fmod(
                        static_cast<typename T::FmodBase const &>(fmod),
                        arg))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::fmod(
                            static_cast<typename T::FmodBase const &>(fmod),
                            arg);
                }
            };
        }
    }
}
