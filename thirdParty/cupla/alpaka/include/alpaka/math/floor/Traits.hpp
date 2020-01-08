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
            //! The floor trait.
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Floor;
        }

        //-----------------------------------------------------------------------------
        //! Computes the largest integer value not greater than arg.
        //!
        //! \tparam T The type of the object specializing Floor.
        //! \tparam TArg The arg type.
        //! \param floor_ctx The object specializing Floor.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename TArg>
        ALPAKA_FN_HOST_ACC auto floor(
            T const & floor_ctx,
            TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Floor<
                T,
                TArg>
            ::floor(
                floor_ctx,
                arg))
#endif
        {
            return
                traits::Floor<
                    T,
                    TArg>
                ::floor(
                    floor_ctx,
                    arg);
        }

        namespace traits
        {
            //#############################################################################
            //! The Floor specialization for classes with FloorBase member type.
            template<
                typename T,
                typename TArg>
            struct Floor<
                T,
                TArg,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::FloorBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto floor(
                    T const & floor_ctx,
                    TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::floor(
                        static_cast<typename T::FloorBase const &>(floor_ctx),
                        arg))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::floor(
                            static_cast<typename T::FloorBase const &>(floor_ctx),
                            arg);
                }
            };
        }
    }
}
