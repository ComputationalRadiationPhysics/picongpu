/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Erik Zenker, Matthias Werner, René Widera, Andrea Bocci, Jan Stephan,
 * Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Align.hpp>
#include <alpaka/core/Assert.hpp>
#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unreachable.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/meta/Fold.hpp>
#include <alpaka/meta/Functional.hpp>
#include <alpaka/meta/IntegerSequence.hpp>
#include <alpaka/offset/Traits.hpp>
#include <alpaka/vec/Traits.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <ostream>
#include <tuple>
#include <type_traits>
#include <utility>

namespace alpaka
{
    template<typename TDim, typename TVal>
    class Vec;

    //! Single value constructor helper.
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TDim,
        template<std::size_t>
        class TTFnObj,
        typename... TArgs,
        typename TIdxSize,
        TIdxSize... TIndices>
    ALPAKA_FN_HOST_ACC auto createVecFromIndexedFnArbitrary(
        std::integer_sequence<TIdxSize, TIndices...> const& /* indices */,
        TArgs&&... args)
    {
        return Vec<TDim, decltype(TTFnObj<0>::create(std::forward<TArgs>(args)...))>(
            (TTFnObj<TIndices>::create(std::forward<TArgs>(args)...))...);
    }
    //! Creator using func<idx>(args...) to initialize all values of the vector.
    //! The idx is in the range [0, TDim].
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TDim, template<std::size_t> class TTFnObj, typename... TArgs>
    ALPAKA_FN_HOST_ACC auto createVecFromIndexedFn(TArgs&&... args)
    {
        using IdxSequence = std::make_integer_sequence<typename TDim::value_type, TDim::value>;
        return createVecFromIndexedFnArbitrary<TDim, TTFnObj>(IdxSequence(), std::forward<TArgs>(args)...);
    }

    //! Creator using func<idx>(args...) to initialize all values of the vector.
    //! The idx is in the range [TIdxOffset, TIdxOffset + TDim].
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TDim, template<std::size_t> class TTFnObj, typename TIdxOffset, typename... TArgs>
    ALPAKA_FN_HOST_ACC auto createVecFromIndexedFnOffset(TArgs&&... args)
    {
        using IdxSubSequenceSigned = meta::MakeIntegerSequenceOffset<std::intmax_t, TIdxOffset::value, TDim::value>;
        using IdxSubSequence = meta::ConvertIntegerSequence<typename TIdxOffset::value_type, IdxSubSequenceSigned>;
        return createVecFromIndexedFnArbitrary<TDim, TTFnObj>(IdxSubSequence(), std::forward<TArgs>(args)...);
    }

    //! A n-dimensional vector.
    template<typename TDim, typename TVal>
    class Vec final
    {
    public:
        static_assert(TDim::value >= 0u, "Invalid dimensionality");

        using Dim = TDim;
        using Val = TVal;

    private:
        //! A sequence of integers from 0 to dim-1.
        //! This can be used to write compile time indexing algorithms.
        using IdxSequence = std::make_integer_sequence<std::size_t, TDim::value>;

    public:
        ALPAKA_FN_HOST_ACC constexpr Vec() : m_data{}
        {
        }

        //! Value constructor.
        //! This constructor is only available if the number of parameters matches the vector idx.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename... TArgs,
            typename = std::enable_if_t<
                sizeof...(TArgs) == TDim::value && (std::is_convertible_v<std::decay_t<TArgs>, TVal> && ...)>>
        ALPAKA_FN_HOST_ACC constexpr Vec(TArgs&&... args) : m_data{static_cast<TVal>(std::forward<TArgs>(args))...}
        {
        }

        //! \brief Single value constructor.
        //!
        //! Creates a vector with all values set to val.
        //! \param val The initial value.
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC static constexpr auto all(TVal const& val) -> Vec<TDim, TVal>
        {
            Vec<TDim, TVal> v;
            for(auto& e : v)
                e = val;
            return v;
        }

        //! Zero value constructor.
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC static constexpr auto zeros() -> Vec<TDim, TVal>
        {
            return all(static_cast<TVal>(0));
        }

        //! One value constructor.
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC static constexpr auto ones() -> Vec<TDim, TVal>
        {
            return all(static_cast<TVal>(1));
        }

        ALPAKA_FN_HOST_ACC constexpr auto begin() -> TVal*
        {
            return m_data;
        }

        ALPAKA_FN_HOST_ACC constexpr auto begin() const -> const TVal*
        {
            return m_data;
        }

        ALPAKA_FN_HOST_ACC constexpr auto end() -> TVal*
        {
            return m_data + TDim::value;
        }

        ALPAKA_FN_HOST_ACC constexpr auto end() const -> const TVal*
        {
            return m_data + TDim::value;
        }

        //! Value reference accessor at the given non-unsigned integer index.
        //! \return A reference to the value at the given index.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TIdx, typename = std::enable_if_t<std::is_integral_v<TIdx>>>
        ALPAKA_FN_HOST_ACC constexpr auto operator[](TIdx const iIdx) -> TVal&
        {
            core::assertValueUnsigned(iIdx);
            auto const idx = static_cast<typename TDim::value_type>(iIdx);
            core::assertGreaterThan<TDim>(idx);
            return m_data[idx];
        }

        //! Value accessor at the given non-unsigned integer index.
        //! \return The value at the given index.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TIdx, typename = std::enable_if_t<std::is_integral_v<TIdx>>>
        ALPAKA_FN_HOST_ACC constexpr auto operator[](TIdx const iIdx) const -> TVal
        {
            core::assertValueUnsigned(iIdx);
            auto const idx = static_cast<typename TDim::value_type>(iIdx);
            core::assertGreaterThan<TDim>(idx);
            return m_data[idx];
        }

        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TFnObj, std::size_t... TIndices>
        [[nodiscard]] ALPAKA_FN_HOST_ACC constexpr auto foldrByIndices(
            TFnObj const& f,
            std::integer_sequence<std::size_t, TIndices...>) const
        {
            return meta::foldr(f, (*this)[TIndices]...);
        }

        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TFnObj, std::size_t... TIndices>
        [[nodiscard]] ALPAKA_FN_HOST_ACC constexpr auto foldrByIndices(
            TFnObj const& f,
            std::integer_sequence<std::size_t, TIndices...>,
            TVal initial) const
        {
            return meta::foldr(f, (*this)[TIndices]..., initial);
        }

        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TFnObj>
        [[nodiscard]] ALPAKA_FN_HOST_ACC constexpr auto foldrAll(TFnObj const& f) const
        {
            return foldrByIndices(f, IdxSequence());
        }

        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TFnObj>
        [[nodiscard]] ALPAKA_FN_HOST_ACC constexpr auto foldrAll(TFnObj const& f, TVal initial) const
        {
            return foldrByIndices(f, IdxSequence(), initial);
        }

// suppress strange warning produced by nvcc+MSVC in release mode
#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#    pragma warning(push)
#    pragma warning(disable : 4702) // unreachable code
#endif
        //! \return The product of all values.
        ALPAKA_NO_HOST_ACC_WARNING
        [[nodiscard]] ALPAKA_FN_HOST_ACC constexpr auto prod() const -> TVal
        {
            return foldrAll(std::multiplies<TVal>(), TVal(1));
        }
#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#    pragma warning(pop)
#endif
        //! \return The sum of all values.
        ALPAKA_NO_HOST_ACC_WARNING
        [[nodiscard]] ALPAKA_FN_HOST_ACC constexpr auto sum() const -> TVal
        {
            return foldrAll(std::plus<TVal>(), TVal(0));
        }

        //! \return The min of all values.
        ALPAKA_NO_HOST_ACC_WARNING
        [[nodiscard]] ALPAKA_FN_HOST_ACC constexpr auto min() const -> TVal
        {
            return foldrAll(meta::min<TVal>(), std::numeric_limits<TVal>::max());
        }

        //! \return The max of all values.
        ALPAKA_NO_HOST_ACC_WARNING
        [[nodiscard]] ALPAKA_FN_HOST_ACC constexpr auto max() const -> TVal
        {
            return foldrAll(meta::max<TVal>(), std::numeric_limits<TVal>::min());
        }

        //! \return The index of the minimal element.
        [[nodiscard]] ALPAKA_FN_HOST constexpr auto minElem() const -> typename TDim::value_type
        {
            return static_cast<typename TDim::value_type>(
                std::distance(std::begin(m_data), std::min_element(std::begin(m_data), std::end(m_data))));
        }

        //! \return The index of the maximal element.
        [[nodiscard]] ALPAKA_FN_HOST constexpr auto maxElem() const -> typename TDim::value_type
        {
            return static_cast<typename TDim::value_type>(
                std::distance(std::begin(m_data), std::max_element(std::begin(m_data), std::end(m_data))));
        }

        template<size_t I>
        ALPAKA_FN_HOST_ACC constexpr auto get() -> TVal&
        {
            return (*this)[I];
        }

        template<size_t I>
        [[nodiscard]] ALPAKA_FN_HOST_ACC constexpr auto get() const -> TVal
        {
            return (*this)[I];
        }

        //! \return The element-wise sum of two vectors.
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC friend constexpr auto operator+(Vec const& p, Vec const& q) -> Vec
        {
            Vec r;
            if constexpr(TDim::value > 0)
            {
                for(typename TDim::value_type i = 0; i < TDim::value; ++i)
                    r[i] = p[i] + q[i];
            }
            return r;
        }

        //! \return The element-wise difference of two vectors.
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC friend constexpr auto operator-(Vec const& p, Vec const& q) -> Vec
        {
            Vec r;
#if BOOST_COMP_NVCC && BOOST_COMP_NVCC < BOOST_VERSION_NUMBER(11, 3, 0)
            if(TDim::value > 0)
#else
            if constexpr(TDim::value > 0)
#endif
            {
#if BOOST_COMP_NVCC && BOOST_COMP_NVCC < BOOST_VERSION_NUMBER(11, 3, 0)
#    pragma diag_suppress = unsigned_compare_with_zero
#endif
                for(typename TDim::value_type i = 0; i < TDim::value; ++i)
#if BOOST_COMP_NVCC && BOOST_COMP_NVCC < BOOST_VERSION_NUMBER(11, 3, 0)
#    pragma diag_default = unsigned_compare_with_zero
#endif
                    r[i] = p[i] - q[i];
            }
            return r;
        }

        //! \return The element-wise product of two vectors.
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC friend constexpr auto operator*(Vec const& p, Vec const& q) -> Vec
        {
            Vec r;
            if constexpr(TDim::value > 0)
            {
                for(typename TDim::value_type i = 0; i < TDim::value; ++i)
                    r[i] = p[i] * q[i];
            }
            return r;
        }

        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC friend constexpr auto operator==(Vec const& a, Vec const& b) -> bool
        {
#if BOOST_COMP_NVCC && BOOST_COMP_NVCC < BOOST_VERSION_NUMBER(11, 3, 0)
            if(TDim::value > 0)
#else
            if constexpr(TDim::value > 0)
#endif
            {
#if BOOST_COMP_NVCC && BOOST_COMP_NVCC < BOOST_VERSION_NUMBER(11, 3, 0)
#    pragma diag_suppress = unsigned_compare_with_zero
#endif
                for(typename TDim::value_type i(0); i < TDim::value; ++i)
#if BOOST_COMP_NVCC && BOOST_COMP_NVCC < BOOST_VERSION_NUMBER(11, 3, 0)
#    pragma diag_default = unsigned_compare_with_zero
#endif
                {
                    if(a[i] != b[i])
                        return false;
                }
            }
            return true;
        }

        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC friend constexpr auto operator!=(Vec const& a, Vec const& b) -> bool
        {
            return !(a == b);
        }

        //! \return The element-wise less than relation of two vectors.
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC friend constexpr auto operator<(Vec const& p, Vec const& q) -> Vec<TDim, bool>
        {
            Vec<TDim, bool> r;
            if constexpr(TDim::value > 0)
            {
                for(typename TDim::value_type i = 0; i < TDim::value; ++i)
                    r[i] = p[i] < q[i];
            }
            return r;
        }

        //! \return The element-wise less than relation of two vectors.
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC friend constexpr auto operator<=(Vec const& p, Vec const& q) -> Vec<TDim, bool>
        {
            Vec<TDim, bool> r;
            if constexpr(TDim::value > 0)
            {
                for(typename TDim::value_type i = 0; i < TDim::value; ++i)
                    r[i] = p[i] <= q[i];
            }
            return r;
        }

        //! \return The element-wise greater than relation of two vectors.
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC friend constexpr auto operator>(Vec const& p, Vec const& q) -> Vec<TDim, bool>
        {
            Vec<TDim, bool> r;
            if constexpr(TDim::value > 0)
            {
                for(typename TDim::value_type i = 0; i < TDim::value; ++i)
                    r[i] = p[i] > q[i];
            }
            return r;
        }

        //! \return The element-wise greater equal than relation of two vectors.
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC friend constexpr auto operator>=(Vec const& p, Vec const& q) -> Vec<TDim, bool>
        {
            Vec<TDim, bool> r;
            if constexpr(TDim::value > 0)
            {
                for(typename TDim::value_type i = 0; i < TDim::value; ++i)
                    r[i] = p[i] >= q[i];
            }
            return r;
        }

        ALPAKA_FN_HOST friend constexpr auto operator<<(std::ostream& os, Vec const& v) -> std::ostream&
        {
            os << "(";
#if BOOST_COMP_NVCC && BOOST_COMP_NVCC < BOOST_VERSION_NUMBER(11, 3, 0)
            if(TDim::value > 0)
#else
            if constexpr(TDim::value > 0)
#endif
            {
#if BOOST_COMP_NVCC && BOOST_COMP_NVCC < BOOST_VERSION_NUMBER(11, 3, 0)
#    pragma diag_suppress = unsigned_compare_with_zero
#endif
                for(typename TDim::value_type i = 0; i < TDim::value; ++i)
#if BOOST_COMP_NVCC && BOOST_COMP_NVCC < BOOST_VERSION_NUMBER(11, 3, 0)
#    pragma diag_default = unsigned_compare_with_zero
#endif
                {
                    os << v[i];
                    if(i != TDim::value - 1)
                        os << ", ";
                }
            }
            else
                os << ".";
            os << ")";

            return os;
        }

    private:
        // Zero sized arrays are not allowed, therefore zero-dimensional vectors have one member.
        TVal m_data[TDim::value == 0u ? 1u : TDim::value];
    };

    namespace trait
    {
        //! The Vec dimension get trait specialization.
        template<typename TDim, typename TVal>
        struct DimType<Vec<TDim, TVal>>
        {
            using type = TDim;
        };

        //! The Vec idx type trait specialization.
        template<typename TDim, typename TVal>
        struct IdxType<Vec<TDim, TVal>>
        {
            using type = TVal;
        };

        //! Specialization for selecting a sub-vector.
        template<typename TDim, typename TVal, std::size_t... TIndices>
        struct SubVecFromIndices<Vec<TDim, TVal>, std::index_sequence<TIndices...>>
        {
            ALPAKA_NO_HOST_ACC_WARNING ALPAKA_FN_HOST_ACC static constexpr auto subVecFromIndices(
                Vec<TDim, TVal> const& vec) -> Vec<DimInt<sizeof...(TIndices)>, TVal>
            {
                if constexpr(std::is_same_v<std::index_sequence<TIndices...>, std::make_index_sequence<TDim::value>>)
                {
                    return vec; // Return whole vector.
                }
                else
                {
                    static_assert(
                        sizeof...(TIndices) <= TDim::value,
                        "The sub-vector's dimensionality must be smaller than or equal to the original "
                        "dimensionality.");
                    return {vec[TIndices]...}; // Return sub-vector.
                }
                ALPAKA_UNREACHABLE({});
            }
        };

        template<typename TValNew, typename TDim, typename TVal>
        struct CastVec<TValNew, Vec<TDim, TVal>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static constexpr auto castVec(Vec<TDim, TVal> const& vec) -> Vec<TDim, TValNew>
            {
                if constexpr(std::is_same_v<TValNew, TVal>)
                {
                    return vec;
                }
                else
                {
                    Vec<TDim, TValNew> r;
                    if constexpr(TDim::value > 0)
                    {
                        for(typename TDim::value_type i = 0; i < TDim::value; ++i)
                            r[i] = static_cast<TValNew>(vec[i]);
                    }
                    return r;
                }
                ALPAKA_UNREACHABLE({});
            }
        };
    } // namespace trait

    namespace trait
    {
        //! ReverseVec specialization for Vec.
        template<typename TDim, typename TVal>
        struct ReverseVec<Vec<TDim, TVal>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static constexpr auto reverseVec(Vec<TDim, TVal> const& vec) -> Vec<TDim, TVal>
            {
                if constexpr(TDim::value <= 1)
                {
                    return vec;
                }
                else
                {
                    Vec<TDim, TVal> r;
                    for(typename TDim::value_type i = 0; i < TDim::value; ++i)
                        r[i] = vec[TDim::value - 1u - i];
                    return r;
                }
                ALPAKA_UNREACHABLE({});
            }
        };

        //! Concatenation specialization for Vec.
        template<typename TDimL, typename TDimR, typename TVal>
        struct ConcatVec<Vec<TDimL, TVal>, Vec<TDimR, TVal>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static constexpr auto concatVec(
                Vec<TDimL, TVal> const& vecL,
                Vec<TDimR, TVal> const& vecR) -> Vec<DimInt<TDimL::value + TDimR::value>, TVal>
            {
                Vec<DimInt<TDimL::value + TDimR::value>, TVal> r;
                if constexpr(TDimL::value > 0)
                {
                    for(typename TDimL::value_type i = 0; i < TDimL::value; ++i)
                        r[i] = vecL[i];
                }
                if constexpr(TDimR::value > 0)
                {
                    for(typename TDimR::value_type i = 0; i < TDimR::value; ++i)
                        r[TDimL::value + i] = vecR[i];
                }
                return r;
            }
        };
    } // namespace trait

    namespace detail
    {
        //! A function object that returns the extent for each index.
        template<std::size_t Tidx>
        struct CreateExtent
        {
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TExtent>
            ALPAKA_FN_HOST_ACC static constexpr auto create(TExtent const& extent) -> Idx<TExtent>
            {
                return getExtent<Tidx>(extent);
            }
        };
    } // namespace detail

    //! \tparam TExtent has to specialize GetExtent.
    //! \return The extent vector.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TExtent>
    ALPAKA_FN_HOST_ACC auto constexpr getExtentVec(TExtent const& extent = {}) -> Vec<Dim<TExtent>, Idx<TExtent>>
    {
        return createVecFromIndexedFn<Dim<TExtent>, detail::CreateExtent>(extent);
    }

    //! \tparam TExtent has to specialize GetExtent.
    //! \return The extent but only the last TDim elements.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TDim, typename TExtent>
    ALPAKA_FN_HOST_ACC auto constexpr getExtentVecEnd(TExtent const& extent = {}) -> Vec<TDim, Idx<TExtent>>
    {
        static_assert(TDim::value <= Dim<TExtent>::value, "Cannot get more items than the extent holds");

        using IdxOffset = std::integral_constant<
            std::intmax_t,
            static_cast<std::intmax_t>(Dim<TExtent>::value) - static_cast<std::intmax_t>(TDim::value)>;
        return createVecFromIndexedFnOffset<TDim, detail::CreateExtent, IdxOffset>(extent);
    }

    namespace detail
    {
        //! A function object that returns the offsets for each index.
        template<std::size_t Tidx>
        struct CreateOffset
        {
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TOffsets>
            ALPAKA_FN_HOST_ACC static constexpr auto create(TOffsets const& offsets) -> Idx<TOffsets>
            {
                return getOffset<Tidx>(offsets);
            }
        };
    } // namespace detail

    //! \tparam TOffsets has to specialize GetOffset.
    //! \return The offset vector.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TOffsets>
    ALPAKA_FN_HOST_ACC constexpr auto getOffsetVec(TOffsets const& offsets = {}) -> Vec<Dim<TOffsets>, Idx<TOffsets>>
    {
        return createVecFromIndexedFn<Dim<TOffsets>, detail::CreateOffset>(offsets);
    }

    //! \tparam TOffsets has to specialize GetOffset.
    //! \return The offset vector but only the last TDim elements.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TDim, typename TOffsets>
    ALPAKA_FN_HOST_ACC constexpr auto getOffsetVecEnd(TOffsets const& offsets = {}) -> Vec<TDim, Idx<TOffsets>>
    {
        static_assert(TDim::value <= Dim<TOffsets>::value, "Cannot get more items than the offsets hold");

        using IdxOffset = std::integral_constant<
            std::size_t,
            static_cast<std::size_t>(
                static_cast<std::intmax_t>(Dim<TOffsets>::value) - static_cast<std::intmax_t>(TDim::value))>;
        return createVecFromIndexedFnOffset<TDim, detail::CreateOffset, IdxOffset>(offsets);
    }

    namespace trait
    {
        //! The Vec extent get trait specialization.
        template<typename TIdxIntegralConst, typename TDim, typename TVal>
        struct GetExtent<
            TIdxIntegralConst,
            Vec<TDim, TVal>,
            std::enable_if_t<(TDim::value > TIdxIntegralConst::value)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static constexpr auto getExtent(Vec<TDim, TVal> const& extent) -> TVal
            {
                return extent[TIdxIntegralConst::value];
            }
        };
        //! The Vec extent set trait specialization.
        template<typename TIdxIntegralConst, typename TDim, typename TVal, typename TExtentVal>
        struct SetExtent<
            TIdxIntegralConst,
            Vec<TDim, TVal>,
            TExtentVal,
            std::enable_if_t<(TDim::value > TIdxIntegralConst::value)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static constexpr auto setExtent(Vec<TDim, TVal>& extent, TExtentVal const& extentVal)
                -> void
            {
                extent[TIdxIntegralConst::value] = extentVal;
            }
        };

        //! The Vec offset get trait specialization.
        template<typename TIdxIntegralConst, typename TDim, typename TVal>
        struct GetOffset<
            TIdxIntegralConst,
            Vec<TDim, TVal>,
            std::enable_if_t<(TDim::value > TIdxIntegralConst::value)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static constexpr auto getOffset(Vec<TDim, TVal> const& offsets) -> TVal
            {
                return offsets[TIdxIntegralConst::value];
            }
        };
        //! The Vec offset set trait specialization.
        template<typename TIdxIntegralConst, typename TDim, typename TVal, typename TOffset>
        struct SetOffset<
            TIdxIntegralConst,
            Vec<TDim, TVal>,
            TOffset,
            std::enable_if_t<(TDim::value > TIdxIntegralConst::value)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static constexpr auto setOffset(Vec<TDim, TVal>& offsets, TOffset const& offset) -> void
            {
                offsets[TIdxIntegralConst::value] = offset;
            }
        };
    } // namespace trait
} // namespace alpaka

#if defined(__clang__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wmismatched-tags"
#endif
namespace std
{
    template<typename TDim, typename TVal>
    struct tuple_size<alpaka::Vec<TDim, TVal>> : integral_constant<size_t, TDim::value>
    {
    };

    template<size_t I, typename TDim, typename TVal>
    struct tuple_element<I, alpaka::Vec<TDim, TVal>>
    {
        using type = TVal;
    };
} // namespace std
#if defined(__clang__)
#    pragma GCC diagnostic pop
#endif
