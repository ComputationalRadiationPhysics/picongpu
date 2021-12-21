/* Copyright 2020 Sergei Bastrakov
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Concepts.hpp>

#include <cstdint>
#include <type_traits>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The thread warp specifics
    namespace warp
    {
        struct ConceptWarp
        {
        };

        //-----------------------------------------------------------------------------
        //! The warp traits.
        namespace traits
        {
            //#############################################################################
            //! The warp size trait.
            template<typename TWarp, typename TSfinae = void>
            struct GetSize;

            //#############################################################################
            //! The all warp vote trait.
            template<typename TWarp, typename TSfinae = void>
            struct All;

            //#############################################################################
            //! The any warp vote trait.
            template<typename TWarp, typename TSfinae = void>
            struct Any;

            //#############################################################################
            //! The ballot warp vote trait.
            template<typename TWarp, typename TSfinae = void>
            struct Ballot;

            //#############################################################################
            //! The active mask trait.
            template<typename TWarp, typename TSfinae = void>
            struct Activemask;
        } // namespace traits

        //-----------------------------------------------------------------------------
        //! Returns warp size.
        //!
        //! \tparam TWarp The warp implementation type.
        //! \param warp The warp implementation.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TWarp>
        ALPAKA_FN_ACC auto getSize(TWarp const& warp) -> std::int32_t
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptWarp, TWarp>;
            return traits::GetSize<ImplementationBase>::getSize(warp);
        }

        //-----------------------------------------------------------------------------
        //! Returns a 32- or 64-bit unsigned integer (depending on the
        //! accelerator) whose Nth bit is set if and only if the Nth thread
        //! of the warp is active.
        //!
        //! Note: decltype for return type is required there, otherwise
        //! compilcation with a CPU and a GPU accelerator enabled fails as it
        //! tries to call device function from a host-device one. The reason
        //! is unclear, but likely related to deducing the return type.
        //!
        //! \tparam TWarp The warp implementation type.
        //! \param warp The warp implementation.
        //! \return 32-bit or 64-bit unsigned type depending on the accelerator.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TWarp>
        ALPAKA_FN_ACC auto activemask(TWarp const& warp)
            -> decltype(traits::Activemask<concepts::ImplementationBase<ConceptWarp, TWarp>>::activemask(warp))
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptWarp, TWarp>;
            return traits::Activemask<ImplementationBase>::activemask(warp);
        }

        //-----------------------------------------------------------------------------
        //! Evaluates predicate for all active threads of the warp and returns
        //! non-zero if and only if predicate evaluates to non-zero for all of them.
        //!
        //! It follows the logic of __all(predicate) in CUDA before version 9.0 and HIP,
        //! the operation is applied for all active threads.
        //! The modern CUDA counterpart would be __all_sync(__activemask(), predicate).
        //!
        //! \tparam TWarp The warp implementation type.
        //! \param warp The warp implementation.
        //! \param predicate The predicate value for current thread.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TWarp>
        ALPAKA_FN_ACC auto all(TWarp const& warp, std::int32_t predicate) -> std::int32_t
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptWarp, TWarp>;
            return traits::All<ImplementationBase>::all(warp, predicate);
        }

        //-----------------------------------------------------------------------------
        //! Evaluates predicate for all active threads of the warp and returns
        //! non-zero if and only if predicate evaluates to non-zero for any of them.
        //!
        //! It follows the logic of __any(predicate) in CUDA before version 9.0 and HIP,
        //! the operation is applied for all active threads.
        //! The modern CUDA counterpart would be __any_sync(__activemask(), predicate).
        //!
        //! \tparam TWarp The warp implementation type.
        //! \param warp The warp implementation.
        //! \param predicate The predicate value for current thread.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TWarp>
        ALPAKA_FN_ACC auto any(TWarp const& warp, std::int32_t predicate) -> std::int32_t
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptWarp, TWarp>;
            return traits::Any<ImplementationBase>::any(warp, predicate);
        }

        //-----------------------------------------------------------------------------
        //! Evaluates predicate for all non-exited threads in a warp and returns
        //! a 32- or 64-bit unsigned integer (depending on the accelerator)
        //! whose Nth bit is set if and only if predicate evaluates to non-zero
        //! for the Nth thread of the warp and the Nth thread is active.
        //!
        //! It follows the logic of __ballot(predicate) in CUDA before version 9.0 and HIP,
        //! the operation is applied for all active threads.
        //! The modern CUDA counterpart would be __ballot_sync(__activemask(), predicate).
        //! Return type is 64-bit to fit all platforms.
        //!
        //! \tparam TWarp The warp implementation type.
        //! \param warp The warp implementation.
        //! \param predicate The predicate value for current thread.
        //! \return 32-bit or 64-bit unsigned type depending on the accelerator.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TWarp>
        ALPAKA_FN_ACC auto ballot(TWarp const& warp, std::int32_t predicate)
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptWarp, TWarp>;
            return traits::Ballot<ImplementationBase>::ballot(warp, predicate);
        }
    } // namespace warp
} // namespace alpaka
