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

#include <type_traits>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The time traits specifics.
    namespace time
    {
        //-----------------------------------------------------------------------------
        //! The time traits.
        namespace traits
        {
            //#############################################################################
            //! The clock trait.
            template<
                typename TTime,
                typename TSfinae = void>
            struct Clock;
        }

        //-----------------------------------------------------------------------------
        //! \return A counter that is increasing every clock cycle.
        //!
        //! \tparam TTime The time implementation type.
        //! \param time The time implementation.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TTime>
        ALPAKA_FN_HOST_ACC auto clock(
            TTime const & time)
        -> std::uint64_t
        {
            return
                traits::Clock<
                    TTime>
                ::clock(
                    time);
        }

        namespace traits
        {
            //#############################################################################
            //! The Clock trait specialization for classes with TimeBase member type.
            template<
                typename TTime>
            struct Clock<
                TTime,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename TTime::TimeBase,
                        TTime
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto clock(
                    TTime const & time)
                -> std::uint64_t
                {
                    // Delegate the call to the base class.
                    return
                        time::clock(
                            static_cast<typename TTime::TimeBase const &>(time));
                }
            };
        }
    }
}
