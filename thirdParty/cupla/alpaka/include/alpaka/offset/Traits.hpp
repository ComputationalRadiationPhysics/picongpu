/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/idx/Traits.hpp>

#include <type_traits>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The offset traits.
    namespace traits
    {
        //#############################################################################
        //! The x offset get trait.
        //!
        //! If not specialized explicitly it returns 0.
        template<typename TIdx, typename TOffsets, typename TSfinae = void>
        struct GetOffset
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getOffset(TOffsets const&) -> Idx<TOffsets>
            {
                return static_cast<Idx<TOffsets>>(0);
            }
        };

        //#############################################################################
        //! The x offset set trait.
        template<typename TIdx, typename TOffsets, typename TOffset, typename TSfinae = void>
        struct SetOffset;
    } // namespace traits

    //-----------------------------------------------------------------------------
    //! \return The offset in the given dimension.
    ALPAKA_NO_HOST_ACC_WARNING
    template<std::size_t Tidx, typename TOffsets>
    ALPAKA_FN_HOST_ACC auto getOffset(TOffsets const& offsets) -> Idx<TOffsets>
    {
        return traits::GetOffset<DimInt<Tidx>, TOffsets>::getOffset(offsets);
    }
    //-----------------------------------------------------------------------------
    //! \return The offset in x dimension.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TOffsets>
    ALPAKA_FN_HOST_ACC auto getOffsetX(TOffsets const& offsets = TOffsets()) -> Idx<TOffsets>
    {
        return getOffset<Dim<TOffsets>::value - 1u>(offsets);
    }
    //-----------------------------------------------------------------------------
    //! \return The offset in y dimension.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TOffsets>
    ALPAKA_FN_HOST_ACC auto getOffsetY(TOffsets const& offsets = TOffsets()) -> Idx<TOffsets>
    {
        return getOffset<Dim<TOffsets>::value - 2u>(offsets);
    }
    //-----------------------------------------------------------------------------
    //! \return The offset in z dimension.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TOffsets>
    ALPAKA_FN_HOST_ACC auto getOffsetZ(TOffsets const& offsets = TOffsets()) -> Idx<TOffsets>
    {
        return getOffset<Dim<TOffsets>::value - 3u>(offsets);
    }

    //-----------------------------------------------------------------------------
    //! Sets the offset in the given dimension.
    ALPAKA_NO_HOST_ACC_WARNING
    template<std::size_t Tidx, typename TOffsets, typename TOffset>
    ALPAKA_FN_HOST_ACC auto setOffset(TOffsets const& offsets, TOffset const& offset) -> void
    {
        traits::SetOffset<DimInt<Tidx>, TOffsets, TOffset>::setOffset(offsets, offset);
    }
    //-----------------------------------------------------------------------------
    //! Sets the offset in x dimension.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TOffsets, typename TOffset>
    ALPAKA_FN_HOST_ACC auto setOffsetX(TOffsets const& offsets, TOffset const& offset) -> void
    {
        setOffset<Dim<TOffsets>::value - 1u>(offsets, offset);
    }
    //-----------------------------------------------------------------------------
    //! Sets the offset in y dimension.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TOffsets, typename TOffset>
    ALPAKA_FN_HOST_ACC auto setOffsetY(TOffsets const& offsets, TOffset const& offset) -> void
    {
        setOffset<Dim<TOffsets>::value - 2u>(offsets, offset);
    }
    //-----------------------------------------------------------------------------
    //! Sets the offset in z dimension.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TOffsets, typename TOffset>
    ALPAKA_FN_HOST_ACC auto setOffsetZ(TOffsets const& offsets, TOffset const& offset) -> void
    {
        setOffset<Dim<TOffsets>::value - 3u>(offsets, offset);
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for unsigned integral types.
    namespace traits
    {
        //#############################################################################
        //! The unsigned integral x offset get trait specialization.
        template<typename TOffsets>
        struct GetOffset<DimInt<0u>, TOffsets, std::enable_if_t<std::is_integral<TOffsets>::value>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getOffset(TOffsets const& offset) -> Idx<TOffsets>
            {
                return offset;
            }
        };
        //#############################################################################
        //! The unsigned integral x offset set trait specialization.
        template<typename TOffsets, typename TOffset>
        struct SetOffset<DimInt<0u>, TOffsets, TOffset, std::enable_if_t<std::is_integral<TOffsets>::value>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto setOffset(TOffsets const& offsets, TOffset const& offset) -> void
            {
                offsets = offset;
            }
        };
    } // namespace traits
} // namespace alpaka
