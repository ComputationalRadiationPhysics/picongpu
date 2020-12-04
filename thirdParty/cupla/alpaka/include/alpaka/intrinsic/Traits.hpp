/* Copyright 2020 Sergei Bastrakov
 *
 * This file is part of alpaka.
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
    struct ConceptIntrinsic
    {
    };

    //-----------------------------------------------------------------------------
    //! The intrinsics traits.
    namespace traits
    {
        //#############################################################################
        //! The popcount trait.
        template<typename TWarp, typename TSfinae = void>
        struct Popcount;

        //#############################################################################
        //! The ffs trait.
        template<typename TWarp, typename TSfinae = void>
        struct Ffs;
    } // namespace traits

    //-----------------------------------------------------------------------------
    //! Returns the number of 1 bits in the given 32-bit value.
    //!
    //! \tparam TIntrinsic The intrinsic implementation type.
    //! \param intrinsic The intrinsic implementation.
    //! \param value The input value.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TIntrinsic>
    ALPAKA_FN_ACC auto popcount(TIntrinsic const& intrinsic, std::uint32_t value) -> std::int32_t
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptIntrinsic, TIntrinsic>;
        return traits::Popcount<ImplementationBase>::popcount(intrinsic, value);
    }

    //-----------------------------------------------------------------------------
    //! Returns the number of 1 bits in the given 64-bit value.
    //!
    //! \tparam TIntrinsic The intrinsic implementation type.
    //! \param intrinsic The intrinsic implementation.
    //! \param value The input value.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TIntrinsic>
    ALPAKA_FN_ACC auto popcount(TIntrinsic const& intrinsic, std::uint64_t value) -> std::int32_t
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptIntrinsic, TIntrinsic>;
        return traits::Popcount<ImplementationBase>::popcount(intrinsic, value);
    }

    //-----------------------------------------------------------------------------
    //! Returns the 1-based position of the least significant bit set to 1
    //! in the given 32-bit value. Returns 0 for input value 0.
    //!
    //! \tparam TIntrinsic The intrinsic implementation type.
    //! \param intrinsic The intrinsic implementation.
    //! \param value The input value.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TIntrinsic>
    ALPAKA_FN_ACC auto ffs(TIntrinsic const& intrinsic, std::int32_t value) -> std::int32_t
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptIntrinsic, TIntrinsic>;
        return traits::Ffs<ImplementationBase>::ffs(intrinsic, value);
    }

    //-----------------------------------------------------------------------------
    //! Returns the 1-based position of the least significant bit set to 1
    //! in the given 64-bit value. Returns 0 for input value 0.
    //!
    //! \tparam TIntrinsic The intrinsic implementation type.
    //! \param intrinsic The intrinsic implementation.
    //! \param value The input value.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TIntrinsic>
    ALPAKA_FN_ACC auto ffs(TIntrinsic const& intrinsic, std::int64_t value) -> std::int32_t
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptIntrinsic, TIntrinsic>;
        return traits::Ffs<ImplementationBase>::ffs(intrinsic, value);
    }
} // namespace alpaka
