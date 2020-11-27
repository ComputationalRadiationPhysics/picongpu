/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Erik Zenker, Matthias Werner, René Widera
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
#include <alpaka/core/Unused.hpp>
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
#include <ostream>
#include <type_traits>
#include <utility>

namespace alpaka
{
    template<typename TDim, typename TVal>
    class Vec;

    //-----------------------------------------------------------------------------
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
        std::integer_sequence<TIdxSize, TIndices...> const& indices,
        TArgs&&... args)
    {
        alpaka::ignore_unused(indices);

        return Vec<TDim, decltype(TTFnObj<0>::create(std::forward<TArgs>(args)...))>(
            (TTFnObj<TIndices>::create(std::forward<TArgs>(args)...))...);
    }
    //-----------------------------------------------------------------------------
    //! Creator using func<idx>(args...) to initialize all values of the vector.
    //! The idx is in the range [0, TDim].
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TDim, template<std::size_t> class TTFnObj, typename... TArgs>
    ALPAKA_FN_HOST_ACC auto createVecFromIndexedFn(TArgs&&... args)
    {
        using IdxSequence = std::make_integer_sequence<typename TDim::value_type, TDim::value>;
        return createVecFromIndexedFnArbitrary<TDim, TTFnObj>(IdxSequence(), std::forward<TArgs>(args)...);
    }
    //-----------------------------------------------------------------------------
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

    //#############################################################################
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
        //-----------------------------------------------------------------------------
        // The default constructor is only available when the vector is zero-dimensional.
        ALPAKA_NO_HOST_ACC_WARNING
        template<bool B = (TDim::value == 0u), typename = std::enable_if_t<B>>
        ALPAKA_FN_HOST_ACC Vec() : m_data{static_cast<TVal>(0u)}
        {
        }


        //-----------------------------------------------------------------------------
        //! Value constructor.
        //! This constructor is only available if the number of parameters matches the vector idx.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TArg0,
            typename... TArgs,
            typename = std::enable_if_t<
                // There have to be dim arguments.
                (sizeof...(TArgs) + 1 == TDim::value) && (std::is_same<TVal, std::decay_t<TArg0>>::value)>>
        ALPAKA_FN_HOST_ACC Vec(TArg0&& arg0, TArgs&&... args)
            : m_data{std::forward<TArg0>(arg0), std::forward<TArgs>(args)...}
        {
        }

        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC
        Vec(Vec const&) = default;
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC
        Vec(Vec&&) noexcept = default;
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC
        auto operator=(Vec const&) -> Vec& = default;
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC
        auto operator=(Vec&&) noexcept -> Vec& = default;
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC ~Vec() = default;

    private:
        //#############################################################################
        //! A function object that returns the given value for each index.
        template<std::size_t Tidx>
        struct CreateSingleVal
        {
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto create(TVal const& val) -> TVal
            {
                return val;
            }
        };

    public:
        //-----------------------------------------------------------------------------
        //! \brief Single value constructor.
        //!
        //! Creates a vector with all values set to val.
        //! \param val The initial value.
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC static auto all(TVal const& val) -> Vec<TDim, TVal>
        {
            return createVecFromIndexedFn<TDim, CreateSingleVal>(val);
        }
        //-----------------------------------------------------------------------------
        //! Zero value constructor.
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC static auto zeros() -> Vec<TDim, TVal>
        {
            return all(static_cast<TVal>(0));
        }
        //-----------------------------------------------------------------------------
        //! One value constructor.
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC static auto ones() -> Vec<TDim, TVal>
        {
            return all(static_cast<TVal>(1));
        }

        //-----------------------------------------------------------------------------
        //! Value reference accessor at the given non-unsigned integer index.
        //! \return A reference to the value at the given index.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TIdx, typename = std::enable_if_t<std::is_integral<TIdx>::value>>
        ALPAKA_FN_HOST_ACC auto operator[](TIdx const iIdx) -> TVal&
        {
            core::assertValueUnsigned(iIdx);
            auto const idx(static_cast<typename TDim::value_type>(iIdx));
            core::assertGreaterThan<TDim>(idx);
            return m_data[idx];
        }

        //-----------------------------------------------------------------------------
        //! Value accessor at the given non-unsigned integer index.
        //! \return The value at the given index.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TIdx, typename = std::enable_if_t<std::is_integral<TIdx>::value>>
        ALPAKA_FN_HOST_ACC auto operator[](TIdx const iIdx) const -> TVal
        {
            core::assertValueUnsigned(iIdx);
            auto const idx(static_cast<typename TDim::value_type>(iIdx));
            core::assertGreaterThan<TDim>(idx);
            return m_data[idx];
        }

        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC auto operator==(Vec const& rhs) const -> bool
        {
            for(typename TDim::value_type i(0); i < TDim::value; ++i)
            {
                if((*this)[i] != rhs[i])
                {
                    return false;
                }
            }
            return true;
        }
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC auto operator!=(Vec const& rhs) const -> bool
        {
            return !((*this) == rhs);
        }
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TFnObj, std::size_t... TIndices>
        ALPAKA_FN_HOST_ACC auto foldrByIndices(
            TFnObj const& f,
            std::integer_sequence<std::size_t, TIndices...> const& indices) const
        {
            alpaka::ignore_unused(indices);

            return meta::foldr(f, ((*this)[TIndices])...);
        }
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TFnObj>
        ALPAKA_FN_HOST_ACC auto foldrAll(TFnObj const& f) const
        {
            return foldrByIndices(f, IdxSequence());
        }
// suppress strange warning produced by nvcc+MSVC in release mode
#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#    pragma warning(push)
#    pragma warning(disable : 4702) // unreachable code
#endif
        //-----------------------------------------------------------------------------
        //! \return The product of all values.
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC auto prod() const -> TVal
        {
            return foldrAll(std::multiplies<TVal>());
        }
#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#    pragma warning(pop)
#endif
        //-----------------------------------------------------------------------------
        //! \return The sum of all values.
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC auto sum() const -> TVal
        {
            return foldrAll(std::plus<TVal>());
        }
        //-----------------------------------------------------------------------------
        //! \return The min of all values.
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC auto min() const -> TVal
        {
            return foldrAll(meta::min<TVal>());
        }
        //-----------------------------------------------------------------------------
        //! \return The max of all values.
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC auto max() const -> TVal
        {
            return foldrAll(meta::max<TVal>());
        }
        //-----------------------------------------------------------------------------
        //! \return The index of the minimal element.
        ALPAKA_FN_HOST auto minElem() const -> typename TDim::value_type
        {
            return static_cast<typename TDim::value_type>(
                std::distance(std::begin(m_data), std::min_element(std::begin(m_data), std::end(m_data))));
        }
        //-----------------------------------------------------------------------------
        //! \return The index of the maximal element.
        ALPAKA_FN_HOST auto maxElem() const -> typename TDim::value_type
        {
            return static_cast<typename TDim::value_type>(
                std::distance(std::begin(m_data), std::max_element(std::begin(m_data), std::end(m_data))));
        }

    private:
        // Zero sized arrays are not allowed, therefore zero-dimensional vectors have one member.
        TVal m_data[TDim::value == 0u ? 1u : TDim::value];
    };

    namespace detail
    {
        //#############################################################################
        //! This is used to create a Vec by applying a binary operation onto the corresponding elements of two input
        //! vectors.
        template<template<typename> class TFnObj, std::size_t Tidx>
        struct CreateVecByApplyingBinaryFnToTwoIndexedVecs
        {
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TDim, typename TVal>
            ALPAKA_FN_HOST_ACC static auto create(Vec<TDim, TVal> const& p, Vec<TDim, TVal> const& q)
            {
                return TFnObj<TVal>()(p[Tidx], q[Tidx]);
            }
        };
    } // namespace detail

    namespace detail
    {
        template<std::size_t Tidx>
        using CreateVecFromTwoIndexedVecsPlus = CreateVecByApplyingBinaryFnToTwoIndexedVecs<std::plus, Tidx>;
    }

    //-----------------------------------------------------------------------------
    //! \return The element-wise sum of two vectors.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TDim, typename TVal>
    ALPAKA_FN_HOST_ACC auto operator+(Vec<TDim, TVal> const& p, Vec<TDim, TVal> const& q) -> Vec<TDim, TVal>
    {
        return createVecFromIndexedFn<TDim, detail::CreateVecFromTwoIndexedVecsPlus>(p, q);
    }

    namespace detail
    {
        template<std::size_t Tidx>
        using CreateVecFromTwoIndexedVecsMinus = CreateVecByApplyingBinaryFnToTwoIndexedVecs<std::minus, Tidx>;
    }

    //-----------------------------------------------------------------------------
    //! \return The element-wise difference of two vectors.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TDim, typename TVal>
    ALPAKA_FN_HOST_ACC auto operator-(Vec<TDim, TVal> const& p, Vec<TDim, TVal> const& q) -> Vec<TDim, TVal>
    {
        return createVecFromIndexedFn<TDim, detail::CreateVecFromTwoIndexedVecsMinus>(p, q);
    }

    namespace detail
    {
        template<std::size_t Tidx>
        using CreateVecFromTwoIndexedVecsMul = CreateVecByApplyingBinaryFnToTwoIndexedVecs<std::multiplies, Tidx>;
    }

    //-----------------------------------------------------------------------------
    //! \return The element-wise product of two vectors.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TDim, typename TVal>
    ALPAKA_FN_HOST_ACC auto operator*(Vec<TDim, TVal> const& p, Vec<TDim, TVal> const& q) -> Vec<TDim, TVal>
    {
        return createVecFromIndexedFn<TDim, detail::CreateVecFromTwoIndexedVecsMul>(p, q);
    }

    namespace detail
    {
        template<std::size_t Tidx>
        using CreateVecFromTwoIndexedVecsLess = CreateVecByApplyingBinaryFnToTwoIndexedVecs<std::less, Tidx>;
    }

    //-----------------------------------------------------------------------------
    //! \return The element-wise less than relation of two vectors.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TDim, typename TVal>
    ALPAKA_FN_HOST_ACC auto operator<(Vec<TDim, TVal> const& p, Vec<TDim, TVal> const& q) -> Vec<TDim, bool>
    {
        return createVecFromIndexedFn<TDim, detail::CreateVecFromTwoIndexedVecsLess>(p, q);
    }

    namespace detail
    {
        template<std::size_t Tidx>
        using CreateVecFromTwoIndexedVecsLessEqual
            = CreateVecByApplyingBinaryFnToTwoIndexedVecs<std::less_equal, Tidx>;
    }

    //-----------------------------------------------------------------------------
    //! \return The element-wise less than or equal relation of two vectors.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TDim, typename TVal>
    ALPAKA_FN_HOST_ACC auto operator<=(Vec<TDim, TVal> const& p, Vec<TDim, TVal> const& q) -> Vec<TDim, bool>
    {
        return createVecFromIndexedFn<TDim, detail::CreateVecFromTwoIndexedVecsLessEqual>(p, q);
    }

    namespace detail
    {
        template<std::size_t Tidx>
        using CreateVecFromTwoIndexedVecsGreaterEqual
            = CreateVecByApplyingBinaryFnToTwoIndexedVecs<std::greater_equal, Tidx>;
    }

    //-----------------------------------------------------------------------------
    //! \return The element-wise greater than or equal relation of two vectors.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TDim, typename TVal>
    ALPAKA_FN_HOST_ACC auto operator>=(Vec<TDim, TVal> const& p, Vec<TDim, TVal> const& q) -> Vec<TDim, bool>
    {
        return createVecFromIndexedFn<TDim, detail::CreateVecFromTwoIndexedVecsGreaterEqual>(p, q);
    }

    namespace detail
    {
        template<std::size_t Tidx>
        using CreateVecFromTwoIndexedVecsGreater = CreateVecByApplyingBinaryFnToTwoIndexedVecs<std::greater, Tidx>;
    }

    //-----------------------------------------------------------------------------
    //! \return The element-wise greater than relation of two vectors.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TDim, typename TVal>
    ALPAKA_FN_HOST_ACC auto operator>(Vec<TDim, TVal> const& p, Vec<TDim, TVal> const& q) -> Vec<TDim, bool>
    {
        return createVecFromIndexedFn<TDim, detail::CreateVecFromTwoIndexedVecsGreater>(p, q);
    }

    //-----------------------------------------------------------------------------
    //! Stream out operator.
    template<typename TDim, typename TVal>
    ALPAKA_FN_HOST auto operator<<(std::ostream& os, Vec<TDim, TVal> const& v) -> std::ostream&
    {
        os << "(";
        for(typename TDim::value_type i(0); i < TDim::value; ++i)
        {
            os << v[i];
            if(i != TDim::value - 1)
            {
                os << ", ";
            }
        }
        os << ")";

        return os;
    }

    namespace traits
    {
        //#############################################################################
        //! The Vec dimension get trait specialization.
        template<typename TDim, typename TVal>
        struct DimType<Vec<TDim, TVal>>
        {
            using type = TDim;
        };

        //#############################################################################
        //! The Vec idx type trait specialization.
        template<typename TDim, typename TVal>
        struct IdxType<Vec<TDim, TVal>>
        {
            using type = TVal;
        };

        //#############################################################################
        //! Specialization for selecting a sub-vector.
        template<typename TDim, typename TVal, std::size_t... TIndices>
        struct SubVecFromIndices<
            Vec<TDim, TVal>,
            std::integer_sequence<std::size_t, TIndices...>,
            std::enable_if_t<!std::is_same<
                std::integer_sequence<std::size_t, TIndices...>,
                std::make_integer_sequence<std::size_t, TDim::value>>::value>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto subVecFromIndices(Vec<TDim, TVal> const& vec)
                -> Vec<DimInt<sizeof...(TIndices)>, TVal>
            {
                // In the case of a zero dimensional vector, vec is unused.
                alpaka::ignore_unused(vec);

                static_assert(
                    sizeof...(TIndices) <= TDim::value,
                    "The sub-vector has to be smaller (or same size) than the origin vector.");

                return Vec<DimInt<sizeof...(TIndices)>, TVal>(vec[TIndices]...);
            }
        };
        //#############################################################################
        //! Specialization for selecting the whole vector.
        template<typename TDim, typename TVal, std::size_t... TIndices>
        struct SubVecFromIndices<
            Vec<TDim, TVal>,
            std::integer_sequence<std::size_t, TIndices...>,
            std::enable_if_t<std::is_same<
                std::integer_sequence<std::size_t, TIndices...>,
                std::make_integer_sequence<std::size_t, TDim::value>>::value>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto subVecFromIndices(Vec<TDim, TVal> const& vec) -> Vec<TDim, TVal>
            {
                return vec;
            }
        };
    } // namespace traits

    namespace detail
    {
        //#############################################################################
        //! A function object that returns the given value for each index.
        template<std::size_t Tidx>
        struct CreateCast
        {
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TSizeNew, typename TDim, typename TVal>
            ALPAKA_FN_HOST_ACC static auto create(TSizeNew const& /* valNew*/, Vec<TDim, TVal> const& vec) -> TSizeNew
            {
                return static_cast<TSizeNew>(vec[Tidx]);
            }
        };
    } // namespace detail
    namespace traits
    {
        //#############################################################################
        //! CastVec specialization for Vec.
        template<typename TSizeNew, typename TDim, typename TVal>
        struct CastVec<TSizeNew, Vec<TDim, TVal>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto castVec(Vec<TDim, TVal> const& vec) -> Vec<TDim, TSizeNew>
            {
                return createVecFromIndexedFn<TDim, alpaka::detail::CreateCast>(TSizeNew(), vec);
            }
        };

        //#############################################################################
        //! (Non-)CastVec specialization for Vec when src and dst types are identical.
        //#############################################################################
        template<typename TDim, typename TVal>
        struct CastVec<TVal, Vec<TDim, TVal>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto castVec(Vec<TDim, TVal> const& vec) -> Vec<TDim, TVal>
            {
                return vec;
            }
        };
    } // namespace traits

    namespace detail
    {
        //#############################################################################
        //! A function object that returns the value at the index from the back of the vector.
        template<std::size_t Tidx>
        struct CreateReverse
        {
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TDim, typename TVal>
            ALPAKA_FN_HOST_ACC static auto create(Vec<TDim, TVal> const& vec) -> TVal
            {
                return vec[TDim::value - 1u - Tidx];
            }
        };
    } // namespace detail
    namespace traits
    {
        //#############################################################################
        //! ReverseVec specialization for Vec.
        template<typename TDim, typename TVal>
        struct ReverseVec<Vec<TDim, TVal>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto reverseVec(Vec<TDim, TVal> const& vec) -> Vec<TDim, TVal>
            {
                return createVecFromIndexedFn<TDim, alpaka::detail::CreateReverse>(vec);
            }
        };

        //#############################################################################
        //! (Non-)ReverseVec specialization for 1D Vec.
        template<typename TVal>
        struct ReverseVec<Vec<DimInt<1u>, TVal>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto reverseVec(Vec<DimInt<1u>, TVal> const& vec) -> Vec<DimInt<1u>, TVal>
            {
                return vec;
            }
        };
    } // namespace traits

    namespace detail
    {
        //#############################################################################
        //! A function object that returns the value at the index from the back of the vector.
        template<std::size_t Tidx>
        struct CreateConcat
        {
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TDimL, typename TDimR, typename TVal>
            ALPAKA_FN_HOST_ACC static auto create(Vec<TDimL, TVal> const& vecL, Vec<TDimR, TVal> const& vecR) -> TVal
            {
                return Tidx < TDimL::value ? vecL[Tidx] : vecR[Tidx - TDimL::value];
            }
        };
    } // namespace detail
    namespace traits
    {
        //#############################################################################
        //! Concatenation specialization for Vec.
        template<typename TDimL, typename TDimR, typename TVal>
        struct ConcatVec<Vec<TDimL, TVal>, Vec<TDimR, TVal>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto concatVec(Vec<TDimL, TVal> const& vecL, Vec<TDimR, TVal> const& vecR)
                -> Vec<DimInt<TDimL::value + TDimR::value>, TVal>
            {
                return createVecFromIndexedFn<DimInt<TDimL::value + TDimR::value>, alpaka::detail::CreateConcat>(
                    vecL,
                    vecR);
            }
        };
    } // namespace traits

    namespace extent
    {
        namespace detail
        {
            //#############################################################################
            //! A function object that returns the extent for each index.
            template<std::size_t Tidx>
            struct CreateExtent
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<typename TExtent>
                ALPAKA_FN_HOST_ACC static auto create(TExtent const& extent) -> Idx<TExtent>
                {
                    return extent::getExtent<Tidx>(extent);
                }
            };
        } // namespace detail
        //-----------------------------------------------------------------------------
        //! \tparam TExtent has to specialize extent::GetExtent.
        //! \return The extent vector.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TExtent>
        ALPAKA_FN_HOST_ACC auto getExtentVec(TExtent const& extent = TExtent()) -> Vec<Dim<TExtent>, Idx<TExtent>>
        {
            return createVecFromIndexedFn<Dim<TExtent>, detail::CreateExtent>(extent);
        }
        //-----------------------------------------------------------------------------
        //! \tparam TExtent has to specialize extent::GetExtent.
        //! \return The extent but only the last N elements.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TDim, typename TExtent>
        ALPAKA_FN_HOST_ACC auto getExtentVecEnd(TExtent const& extent = TExtent()) -> Vec<TDim, Idx<TExtent>>
        {
            using IdxOffset = std::integral_constant<
                std::intmax_t,
                static_cast<std::intmax_t>(Dim<TExtent>::value) - static_cast<std::intmax_t>(TDim::value)>;
            return createVecFromIndexedFnOffset<TDim, detail::CreateExtent, IdxOffset>(extent);
        }
    } // namespace extent

    namespace detail
    {
        //#############################################################################
        //! A function object that returns the offsets for each index.
        template<std::size_t Tidx>
        struct CreateOffset
        {
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TOffsets>
            ALPAKA_FN_HOST_ACC static auto create(TOffsets const& offsets) -> Idx<TOffsets>
            {
                return getOffset<Tidx>(offsets);
            }
        };
    } // namespace detail
    //-----------------------------------------------------------------------------
    //! \tparam TOffsets has to specialize GetOffset.
    //! \return The offset vector.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TOffsets>
    ALPAKA_FN_HOST_ACC auto getOffsetVec(TOffsets const& offsets = TOffsets()) -> Vec<Dim<TOffsets>, Idx<TOffsets>>
    {
        return createVecFromIndexedFn<Dim<TOffsets>, detail::CreateOffset>(offsets);
    }
    //-----------------------------------------------------------------------------
    //! \tparam TOffsets has to specialize GetOffset.
    //! \return The offset vector but only the last N elements.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TDim, typename TOffsets>
    ALPAKA_FN_HOST_ACC auto getOffsetVecEnd(TOffsets const& offsets = TOffsets()) -> Vec<TDim, Idx<TOffsets>>
    {
        using IdxOffset = std::integral_constant<
            std::size_t,
            static_cast<std::size_t>(
                static_cast<std::intmax_t>(Dim<TOffsets>::value) - static_cast<std::intmax_t>(TDim::value))>;
        return createVecFromIndexedFnOffset<TDim, detail::CreateOffset, IdxOffset>(offsets);
    }
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The Vec extent get trait specialization.
            template<typename TIdxIntegralConst, typename TDim, typename TVal>
            struct GetExtent<
                TIdxIntegralConst,
                Vec<TDim, TVal>,
                std::enable_if_t<(TDim::value > TIdxIntegralConst::value)>>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getExtent(Vec<TDim, TVal> const& extent) -> TVal
                {
                    return extent[TIdxIntegralConst::value];
                }
            };
            //#############################################################################
            //! The Vec extent set trait specialization.
            template<typename TIdxIntegralConst, typename TDim, typename TVal, typename TExtentVal>
            struct SetExtent<
                TIdxIntegralConst,
                Vec<TDim, TVal>,
                TExtentVal,
                std::enable_if_t<(TDim::value > TIdxIntegralConst::value)>>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto setExtent(Vec<TDim, TVal>& extent, TExtentVal const& extentVal) -> void
                {
                    extent[TIdxIntegralConst::value] = extentVal;
                }
            };
        } // namespace traits
    } // namespace extent
    namespace traits
    {
        //#############################################################################
        //! The Vec offset get trait specialization.
        template<typename TIdxIntegralConst, typename TDim, typename TVal>
        struct GetOffset<
            TIdxIntegralConst,
            Vec<TDim, TVal>,
            std::enable_if_t<(TDim::value > TIdxIntegralConst::value)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getOffset(Vec<TDim, TVal> const& offsets) -> TVal
            {
                return offsets[TIdxIntegralConst::value];
            }
        };
        //#############################################################################
        //! The Vec offset set trait specialization.
        template<typename TIdxIntegralConst, typename TDim, typename TVal, typename TOffset>
        struct SetOffset<
            TIdxIntegralConst,
            Vec<TDim, TVal>,
            TOffset,
            std::enable_if_t<(TDim::value > TIdxIntegralConst::value)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto setOffset(Vec<TDim, TVal>& offsets, TOffset const& offset) -> void
            {
                offsets[TIdxIntegralConst::value] = offset;
            }
        };
    } // namespace traits
} // namespace alpaka
