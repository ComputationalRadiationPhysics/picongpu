/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
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
        //! \param fmod_ctx The object specializing Fmod.
        //! \param x The first argument.
        //! \param y The second argument.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename Tx,
            typename Ty>
        ALPAKA_FN_HOST_ACC auto fmod(
            T const & fmod_ctx,
            Tx const & x,
            Ty const & y)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Fmod<
                T,
                Tx,
                Ty>
            ::fmod(
                fmod_ctx,
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
                    fmod_ctx,
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
                    T const & fmod_ctx,
                    TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::fmod(
                        static_cast<typename T::FmodBase const &>(fmod_ctx),
                        arg))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::fmod(
                            static_cast<typename T::FmodBase const &>(fmod_ctx),
                            arg);
                }
            };
        }
    }
}
