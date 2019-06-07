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

#include <alpaka/idx/Traits.hpp>

#include <alpaka/vec/Vec.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Common.hpp>

#include <boost/config.hpp>

#include <type_traits>
#include <utility>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The work division traits specifics.
    namespace workdiv
    {
        //-----------------------------------------------------------------------------
        //! The work division traits.
        namespace traits
        {
            //#############################################################################
            //! The work div trait.
            template<
                typename TWorkDiv,
                typename TOrigin,
                typename TUnit,
                typename TSfinae = void>
            struct GetWorkDiv;
        }

        //-----------------------------------------------------------------------------
        //! Get the extent requested.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TOrigin,
            typename TUnit,
            typename TWorkDiv>
        ALPAKA_FN_HOST_ACC auto getWorkDiv(
            TWorkDiv const & workDiv)
        -> vec::Vec<dim::Dim<TWorkDiv>, idx::Idx<TWorkDiv>>
        {
            return
                traits::GetWorkDiv<
                    TWorkDiv,
                    TOrigin,
                    TUnit>
                ::getWorkDiv(
                    workDiv);
        }

        namespace traits
        {
            //#############################################################################
            //! The work div grid block extent trait specialization for classes with WorkDivBase member type.
            template<
                typename TWorkDiv>
            struct GetWorkDiv<
                TWorkDiv,
                origin::Grid,
                unit::Blocks,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename TWorkDiv::WorkDivBase,
                        TWorkDiv
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getWorkDiv(
                    TWorkDiv const & workDiv)
                -> vec::Vec<dim::Dim<typename TWorkDiv::WorkDivBase>, idx::Idx<typename TWorkDiv::WorkDivBase>>
                {
                    // Delegate the call to the base class.
                    return
                        workdiv::getWorkDiv<
                            origin::Grid,
                            unit::Blocks>(
                                static_cast<typename TWorkDiv::WorkDivBase const &>(workDiv));
                }
            };
            //#############################################################################
            //! The work div block thread extent trait specialization for classes with WorkDivBase member type.
            template<
                typename TWorkDiv>
            struct GetWorkDiv<
                TWorkDiv,
                origin::Block,
                unit::Threads,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename TWorkDiv::WorkDivBase,
                        TWorkDiv
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getWorkDiv(
                    TWorkDiv const & workDiv)
                -> vec::Vec<dim::Dim<typename TWorkDiv::WorkDivBase>, idx::Idx<typename TWorkDiv::WorkDivBase>>
                {
                    // Delegate the call to the base class.
                    return
                        workdiv::getWorkDiv<
                            origin::Block,
                            unit::Threads>(
                                static_cast<typename TWorkDiv::WorkDivBase const &>(workDiv));
                }
            };
            //#############################################################################
            //! The work div block thread extent trait specialization for classes with WorkDivBase member type.
            template<
                typename TWorkDiv>
            struct GetWorkDiv<
                TWorkDiv,
                origin::Thread,
                unit::Elems,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename TWorkDiv::WorkDivBase,
                        TWorkDiv
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getWorkDiv(
                    TWorkDiv const & workDiv)
                -> vec::Vec<dim::Dim<typename TWorkDiv::WorkDivBase>, idx::Idx<typename TWorkDiv::WorkDivBase>>
                {
                    // Delegate the call to the base class.
                    return
                        workdiv::getWorkDiv<
                            origin::Thread,
                            unit::Elems>(
                                static_cast<typename TWorkDiv::WorkDivBase const &>(workDiv));
                }
            };

            //#############################################################################
            //! The work div grid thread extent trait specialization.
            template<
                typename TWorkDiv>
            struct GetWorkDiv<
                TWorkDiv,
                origin::Grid,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getWorkDiv(
                    TWorkDiv const & workDiv)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    workdiv::getWorkDiv<origin::Grid, unit::Blocks>(workDiv)
                    * workdiv::getWorkDiv<origin::Block, unit::Threads>(workDiv))
#endif
                {
                    return
                        workdiv::getWorkDiv<origin::Grid, unit::Blocks>(workDiv)
                        * workdiv::getWorkDiv<origin::Block, unit::Threads>(workDiv);
                }
            };
            //#############################################################################
            //! The work div grid element extent trait specialization.
            template<
                typename TWorkDiv>
            struct GetWorkDiv<
                TWorkDiv,
                origin::Grid,
                unit::Elems>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getWorkDiv(
                    TWorkDiv const & workDiv)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    workdiv::getWorkDiv<origin::Grid, unit::Threads>(workDiv)
                    * workdiv::getWorkDiv<origin::Thread, unit::Elems>(workDiv))
#endif
                {
                    return
                        workdiv::getWorkDiv<origin::Grid, unit::Threads>(workDiv)
                        * workdiv::getWorkDiv<origin::Thread, unit::Elems>(workDiv);
                }
            };
            //#############################################################################
            //! The work div block element extent trait specialization.
            template<
                typename TWorkDiv>
            struct GetWorkDiv<
                TWorkDiv,
                origin::Block,
                unit::Elems>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getWorkDiv(
                    TWorkDiv const & workDiv)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    workdiv::getWorkDiv<origin::Block, unit::Threads>(workDiv)
                    * workdiv::getWorkDiv<origin::Thread, unit::Elems>(workDiv))
#endif
                {
                    return
                        workdiv::getWorkDiv<origin::Block, unit::Threads>(workDiv)
                        * workdiv::getWorkDiv<origin::Thread, unit::Elems>(workDiv);
                }
            };
        }
    }
}
