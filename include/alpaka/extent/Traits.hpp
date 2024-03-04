/* Copyright 2023 Benjamin Worpitz, Andrea Bocci, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/core/Unreachable.hpp"
#include "alpaka/dim/DimIntegralConst.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/meta/Fold.hpp"
#include "alpaka/vec/Vec.hpp"

#include <functional>
#include <type_traits>
#include <utility>

namespace alpaka
{
    //! The extent traits.
    namespace trait
    {
        //! The extent get trait.
        //!
        //! If not specialized explicitly it returns 1.
        template<typename TIdxIntegralConst, typename TExtent, typename TSfinae = void>
        struct [[deprecated("Specialize GetExtents instead")]] GetExtent
        {
            ALPAKA_NO_HOST_ACC_WARNING ALPAKA_FN_HOST_ACC static auto getExtent(TExtent const&) -> Idx<TExtent>
            {
                return static_cast<Idx<TExtent>>(1);
            } // namespace trait
        }; // namespace alpaka

        //! The GetExtents trait for getting the extents of an object as an alpaka::Vec.
        template<typename TExtent, typename TSfinae = void>
        struct GetExtents;
    } // namespace trait

    //! \return The extent in the given dimension.
    ALPAKA_NO_HOST_ACC_WARNING
    template<std::size_t Tidx, typename TExtent>
    [[deprecated("use getExtents(extent)[Tidx] instead")]] ALPAKA_FN_HOST_ACC auto getExtent(
        TExtent const& extent = TExtent()) -> Idx<TExtent>
    {
#if BOOST_COMP_CLANG || BOOST_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return trait::GetExtent<DimInt<Tidx>, TExtent>::getExtent(extent);
#if BOOST_COMP_CLANG || BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif
    }

    //! \return The extents of the given object.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T>
    ALPAKA_FN_HOST_ACC auto getExtents(T const& object) -> Vec<Dim<T>, Idx<T>>
    {
        return trait::GetExtents<T>{}(object);
    }

    //! \tparam T has to specialize GetExtent.
    //! \return The extents of the given object.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T>
    [[deprecated("use getExtents() instead")]] ALPAKA_FN_HOST_ACC constexpr auto getExtentVec(T const& object = {})
        -> Vec<Dim<T>, Idx<T>>
    {
        return getExtents(object);
    }

    //! \tparam T has to specialize GetExtent.
    //! \return The extent but only the last TDim elements.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TDim, typename T>
    ALPAKA_FN_HOST_ACC constexpr auto getExtentVecEnd(T const& object = {}) -> Vec<TDim, Idx<T>>
    {
        static_assert(TDim::value <= Dim<T>::value, "Cannot get more items than the extent holds");

        [[maybe_unused]] auto const e = getExtents(object);
        Vec<TDim, Idx<T>> v{};
        if constexpr(TDim::value > 0)
        {
            for(unsigned i = 0; i < TDim::value; i++)
                v[i] = e[(Dim<T>::value - TDim::value) + i];
        }
        return v;
    }

    //! \return The width.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TExtent>
    ALPAKA_FN_HOST_ACC auto getWidth(TExtent const& extent = TExtent()) -> Idx<TExtent>
    {
        if constexpr(Dim<TExtent>::value >= 1)
            return getExtents(extent)[Dim<TExtent>::value - 1u];
        else
            return 1;

        ALPAKA_UNREACHABLE({});
    }

    //! \return The height.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TExtent>
    ALPAKA_FN_HOST_ACC auto getHeight(TExtent const& extent = TExtent()) -> Idx<TExtent>
    {
        if constexpr(Dim<TExtent>::value >= 2)
            return getExtents(extent)[Dim<TExtent>::value - 2u];
        else
            return 1;

        ALPAKA_UNREACHABLE({});
    }

    //! \return The depth.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TExtent>
    ALPAKA_FN_HOST_ACC auto getDepth(TExtent const& extent = TExtent()) -> Idx<TExtent>
    {
        if constexpr(Dim<TExtent>::value >= 3)
            return getExtents(extent)[Dim<TExtent>::value - 3u];
        else
            return 1;

        ALPAKA_UNREACHABLE({});
    }

    //! \return The product of the extents of the given object.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T>
    ALPAKA_FN_HOST_ACC auto getExtentProduct(T const& object) -> Idx<T>
    {
        return getExtents(object).prod();
    }

    namespace trait
    {
        //! The Vec extent get trait specialization.
        template<typename TDim, typename TVal>
        struct GetExtents<Vec<TDim, TVal>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC constexpr auto operator()(Vec<TDim, TVal> const& extent) const -> Vec<TDim, TVal>
            {
                return extent;
            }
        };

        template<typename Integral>
        struct GetExtents<Integral, std::enable_if_t<std::is_integral_v<Integral>>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC auto operator()(Integral i) const
            {
                return Vec{i};
            }
        };
    } // namespace trait
} // namespace alpaka
