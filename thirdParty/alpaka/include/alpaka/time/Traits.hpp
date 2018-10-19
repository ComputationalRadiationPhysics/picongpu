/**
* \file
* Copyright 2016 Benjamin Worpitz
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
