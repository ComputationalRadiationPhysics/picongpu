/**
 * \file
 * Copyright 2014-2017 Benjamin Worpitz
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

#include <alpaka/vec/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/offset/Traits.hpp>
#include <alpaka/size/Traits.hpp>

#include <alpaka/meta/IntegerSequence.hpp>
#include <alpaka/meta/Fold.hpp>
#include <alpaka/core/Align.hpp>
#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Common.hpp>

#include <boost/predef.h>
#include <boost/config.hpp>
#if !BOOST_ARCH_CUDA_DEVICE
    #include <boost/core/ignore_unused.hpp>
#endif

#include <cstdint>
#include <ostream>
#include <cassert>
#include <type_traits>
#include <algorithm>

// Some compilers do not support the out of class versions:
// - the nvcc CUDA compiler (at least 7.0, 7.5 and 8.0)
// - the intel compiler
#if BOOST_COMP_NVCC || BOOST_COMP_INTEL || (BOOST_COMP_CLANG_CUDA >= BOOST_VERSION_NUMBER(4, 0, 0))
    #define ALPAKA_CREATE_VEC_IN_CLASS
#endif

namespace alpaka
{
    namespace vec
    {
        template<
            typename TDim,
            typename TSize>
        class Vec;

#ifndef ALPAKA_CREATE_VEC_IN_CLASS
        //-----------------------------------------------------------------------------
        //! Single value constructor helper.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TDim,
            template<std::size_t> class TTFnObj,
            typename... TArgs,
            typename TIdxSize,
            TIdxSize... TIndices>
        ALPAKA_FN_HOST_ACC auto createVecFromIndexedFnArbitrary(
#if BOOST_ARCH_CUDA_DEVICE
            meta::IntegerSequence<TIdxSize, TIndices...> const &,
#else
            meta::IntegerSequence<TIdxSize, TIndices...> const & indices,
#endif
            TArgs && ... args)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> Vec<TDim, decltype(TTFnObj<0>::create(std::forward<TArgs>(args)...))>
#endif
        {
#if !BOOST_ARCH_CUDA_DEVICE
            boost::ignore_unused(indices);
#endif
            return Vec<TDim, decltype(TTFnObj<0>::create(std::forward<TArgs>(args)...))>(
                (TTFnObj<TIndices>::create(std::forward<TArgs>(args)...))...);
        }
        //-----------------------------------------------------------------------------
        //! Creator using func<idx>(args...) to initialize all values of the vector.
        //! The idx is in the range [0, TDim].
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TDim,
            template<std::size_t> class TTFnObj,
            typename... TArgs>
        ALPAKA_FN_HOST_ACC auto createVecFromIndexedFn(
            TArgs && ... args)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            createVecFromIndexedFnArbitrary<
                TDim,
                TTFnObj>(
                    meta::MakeIntegerSequence<typename TDim::value_type, TDim::value>(),
                    std::forward<TArgs>(args)...))
#endif
        {
            using IdxSequence = meta::MakeIntegerSequence<typename TDim::value_type, TDim::value>;
            return
                createVecFromIndexedFnArbitrary<
                    TDim,
                    TTFnObj>(
                        IdxSequence(),
                        std::forward<TArgs>(args)...);
        }
        //-----------------------------------------------------------------------------
        //! Creator using func<idx>(args...) to initialize all values of the vector.
        //! The idx is in the range [TIdxOffset, TIdxOffset + TDim].
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TDim,
            template<std::size_t> class TTFnObj,
            typename TIdxOffset,
            typename... TArgs>
        ALPAKA_FN_HOST_ACC auto createVecFromIndexedFnOffset(
            TArgs && ... args)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            createVecFromIndexedFnArbitrary<
                TDim,
                TTFnObj>(
                    meta::ConvertIntegerSequence<typename TIdxOffset::value_type, meta::MakeIntegerSequenceOffset<std::intmax_t, TIdxOffset::value, TDim::value>>(),
                    std::forward<TArgs>(args)...))
#endif
        {
            using IdxSubSequenceSigned = meta::MakeIntegerSequenceOffset<std::intmax_t, TIdxOffset::value, TDim::value>;
            using IdxSubSequence = meta::ConvertIntegerSequence<typename TIdxOffset::value_type, IdxSubSequenceSigned>;
            return
                createVecFromIndexedFnArbitrary<
                    TDim,
                    TTFnObj>(
                        IdxSubSequence(),
                        std::forward<TArgs>(args)...);
        }
#endif

        //#############################################################################
        //! A n-dimensional vector.
        template<
            typename TDim,
            typename TSize>
        class Vec final
        {
        public:
            using Dim = TDim;
            static constexpr auto s_uiDim = TDim::value;
            using Val = TSize;

        private:
            //! A sequence of integers from 0 to dim-1.
            //! This can be used to write compile time indexing algorithms.
            using IdxSequence = meta::MakeIntegerSequence<std::size_t, TDim::value>;

        public:
            //-----------------------------------------------------------------------------
            // The default constructor is only available when the vector is zero-dimensional.
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                bool B = (TDim::value == 0u),
                typename = typename std::enable_if<B>::type>
            ALPAKA_FN_HOST_ACC Vec() :
                m_data{static_cast<TSize>(0u)}
            {}


            //-----------------------------------------------------------------------------
            //! Value constructor.
            //! This constructor is only available if the number of parameters matches the vector size.
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TArg0,
                typename... TArgs,
                typename = typename std::enable_if<
                    // There have to be dim arguments.
                    (sizeof...(TArgs)+1 == TDim::value)
                    &&
                    (std::is_same<TSize, typename std::decay<TArg0>::type>::value)
                    >::type>
            ALPAKA_FN_HOST_ACC Vec(
                TArg0 && arg0,
                TArgs && ... args) :
                    m_data{std::forward<TArg0>(arg0), std::forward<TArgs>(args)...}
            {}

#ifdef ALPAKA_CREATE_VEC_IN_CLASS
            //-----------------------------------------------------------------------------
            //! Creator using func<idx>(args...) to initialize all values of the vector.
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                template<std::size_t> class TTFnObj,
                typename... TArgs,
                typename TIdxSize,
                TIdxSize... TIndices>
            ALPAKA_FN_HOST_ACC static auto createVecFromIndexedFnArbitrary(
#if BOOST_ARCH_CUDA_DEVICE
                meta::IntegerSequence<TIdxSize, TIndices...> const &,
#else
                meta::IntegerSequence<TIdxSize, TIndices...> const & indices,
#endif
                TArgs && ... args)
            -> Vec<TDim, TSize>
            {
#if !BOOST_ARCH_CUDA_DEVICE
                boost::ignore_unused(indices);
#endif
                return Vec<TDim, TSize>(
                    (TTFnObj<TIndices>::create(std::forward<TArgs>(args)...))...);
            }
            //-----------------------------------------------------------------------------
            //! Creator using func<idx>(args...) to initialize all values of the vector.
            //! The idx is in the range [0, TDim].
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                template<std::size_t> class TTFnObj,
                typename... TArgs>
            ALPAKA_FN_HOST_ACC static auto createVecFromIndexedFn(
                TArgs && ... args)
            -> Vec<TDim, TSize>
            {
                return
                    createVecFromIndexedFnArbitrary<
                        TTFnObj>(
                            IdxSequence(),
                            std::forward<TArgs>(args)...);
            }
            //-----------------------------------------------------------------------------
            //! Creator using func<idx>(args...) to initialize all values of the vector.
            //! The idx is in the range [TIdxOffset, TIdxOffset + TDim].
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                template<std::size_t> class TTFnObj,
                typename TIdxOffset,
                typename... TArgs>
            ALPAKA_FN_HOST_ACC static auto createVecFromIndexedFnOffset(
                TArgs && ... args)
            -> Vec<TDim, TSize>
            {
                using IdxSubSequenceSigned = meta::MakeIntegerSequenceOffset<std::intmax_t, TIdxOffset::value, TDim::value>;
                using IdxSubSequence = meta::ConvertIntegerSequence<typename TDim::value_type, IdxSubSequenceSigned>;
                return
                    createVecFromIndexedFnArbitrary<
                        TTFnObj>(
                            IdxSubSequence(),
                            std::forward<TArgs>(args)...);
            }
#endif

            //-----------------------------------------------------------------------------
            Vec(Vec const &) = default;
            //-----------------------------------------------------------------------------
            Vec(Vec &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(Vec const &) -> Vec & = default;
            //-----------------------------------------------------------------------------
            auto operator=(Vec &&) -> Vec & = default;
            //-----------------------------------------------------------------------------
            ~Vec() = default;

        private:
            //#############################################################################
            //! A function object that returns the given value for each index.
            template<
                std::size_t Tidx>
            struct CreateSingleVal
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto create(
                    TSize const & val)
                -> TSize
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
            ALPAKA_FN_HOST_ACC static auto all(
                TSize const & val)
            -> Vec<TDim, TSize>
            {
                return
                    createVecFromIndexedFn<
#ifndef ALPAKA_CREATE_VEC_IN_CLASS
                        TDim,
#endif
                        CreateSingleVal>(
                            val);
            }
            //-----------------------------------------------------------------------------
            //! Zero value constructor.
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto zeros()
            -> Vec<TDim, TSize>
            {
                return all(static_cast<TSize>(0));
            }
            //-----------------------------------------------------------------------------
            //! One value constructor.
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto ones()
            -> Vec<TDim, TSize>
            {
                return all(static_cast<TSize>(1));
            }

            //-----------------------------------------------------------------------------
            //! Value reference accessor at the given non-unsigned integer index.
            //! \return A reference to the value at the given index.
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TIdx,
                typename = typename std::enable_if<
                    std::is_integral<TIdx>::value>::type>
            ALPAKA_FN_HOST_ACC auto operator[](
                TIdx const iIdx)
            -> TSize &
            {
                core::assertValueUnsigned(iIdx);
                auto const idx(static_cast<typename TDim::value_type>(iIdx));
                assert(TDim::value > 0u && idx<TDim::value);
                return m_data[idx];
            }

            //-----------------------------------------------------------------------------
            //! Value accessor at the given non-unsigned integer index.
            //! \return The value at the given index.
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TIdx,
                typename = typename std::enable_if<
                    std::is_integral<TIdx>::value>::type>
            ALPAKA_FN_HOST_ACC auto operator[](
                TIdx const iIdx) const
            -> TSize
            {
                core::assertValueUnsigned(iIdx);
                auto const idx(static_cast<typename TDim::value_type>(iIdx));
                assert(TDim::value > 0u && idx<TDim::value);
                return m_data[idx];
            }

            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC auto operator==(
                Vec const & rhs) const
            -> bool
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
            ALPAKA_FN_HOST_ACC auto operator!=(
                Vec const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TFnObj,
                std::size_t... TIndices>
            ALPAKA_FN_HOST_ACC auto foldrByIndices(
                TFnObj const & f,
#if BOOST_ARCH_CUDA_DEVICE
                meta::IntegerSequence<std::size_t, TIndices...> const &) const
#else
                meta::IntegerSequence<std::size_t, TIndices...> const & indices) const
#endif
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
            -> decltype(
                meta::foldr(
                    f,
                    ((*this)[TIndices])...))
#endif
            {
#if !BOOST_ARCH_CUDA_DEVICE
                boost::ignore_unused(indices);
#endif
                return
                    meta::foldr(
                        f,
                        ((*this)[TIndices])...);
            }
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TFnObj>
            ALPAKA_FN_HOST_ACC auto foldrAll(
                TFnObj const & f) const
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
            -> decltype(
#if (BOOST_COMP_GNUC && (BOOST_COMP_GNUC < BOOST_VERSION_NUMBER(5, 0, 0))) || BOOST_COMP_INTEL || BOOST_COMP_NVCC
                this->foldrByIndices(
#else
                foldrByIndices(
#endif
                    f,
                    IdxSequence()))
#endif
            {
                return
                    foldrByIndices(
                        f,
                        IdxSequence());
            }
            //-----------------------------------------------------------------------------
            //! \return The product of all values.
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC auto prod() const
            -> TSize
            {
                return foldrAll(
                    [](TSize a, TSize b)
                    {
                        return static_cast<TSize>(a * b);
                    });
            }
            //-----------------------------------------------------------------------------
            //! \return The sum of all values.
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC auto sum() const
            -> TSize
            {
                return foldrAll(
                    [](TSize a, TSize b)
                    {
                        return static_cast<TSize>(a + b);
                    });
            }
            //-----------------------------------------------------------------------------
            //! \return The min of all values.
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC auto min() const
            -> TSize
            {
                return foldrAll(
                    [](TSize a, TSize b)
                    {
                        return (b < a) ? b : a;
                    });
            }
            //-----------------------------------------------------------------------------
            //! \return The max of all values.
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC auto max() const
            -> TSize
            {
                return foldrAll(
                    [](TSize a, TSize b)
                    {
                        return (b > a) ? b : a;
                    });
            }
            //-----------------------------------------------------------------------------
            //! \return The index of the minimal element.
            ALPAKA_FN_HOST auto minElem() const
            -> typename TDim::value_type
            {
                return
                    static_cast<typename TDim::value_type>(
                        std::distance(
                            std::begin(m_data),
                            std::min_element(
                                std::begin(m_data),
                                std::end(m_data))));
            }
            //-----------------------------------------------------------------------------
            //! \return The index of the maximal element.
            ALPAKA_FN_HOST auto maxElem() const
            -> typename TDim::value_type
            {
                return
                    static_cast<typename TDim::value_type>(
                        std::distance(
                            std::begin(m_data),
                            std::max_element(
                                std::begin(m_data),
                                std::end(m_data))));
            }

        private:
            // Zero sized arrays are not allowed, therefore zero-dimensional vectors have one member.
            TSize m_data[TDim::value == 0u ? 1u : TDim::value];
        };

        //-----------------------------------------------------------------------------
        //! This is a conveniance method to have a out-of-class factory method even though the out-of-class version is not supported by all compilers.
        //! Depending of the compiler conformance, the internal or external factory function is called.
        //! This has the draw-back, that it requires the TSize parameter even though it should not be necessary.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TDim,
            typename TSize,
            template<std::size_t> class TTFnObj,
            typename... TArgs>
        ALPAKA_FN_HOST_ACC auto createVecFromIndexedFnWorkaround(
            TArgs && ... args)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> alpaka::vec::Vec<TDim, TSize>
#endif
        {
            return
                alpaka::vec::
#ifdef ALPAKA_CREATE_VEC_IN_CLASS
                Vec<TDim, TSize>::template
#endif
                createVecFromIndexedFn<
#ifndef ALPAKA_CREATE_VEC_IN_CLASS
                    TDim,
#endif
                    TTFnObj>(
                        std::forward<TArgs>(args)...);
        }

        //-----------------------------------------------------------------------------
        //! This is a conveniance method to have a out-of-class factory method even though the out-of-class version is not supported by all compilers.
        //! Depending of the compiler conformance, the internal or external factory function is called.
        //! This has the draw-back, that it requires the TSize parameter even though it should not be necessary.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TDim,
            typename TSize,
            template<std::size_t> class TTFnObj,
            typename TIdxOffset,
            typename... TArgs>
        ALPAKA_FN_HOST_ACC auto createVecFromIndexedFnOffsetWorkaround(
            TArgs && ... args)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> alpaka::vec::Vec<TDim, TSize>
#endif
        {
            return
                alpaka::vec::
#ifdef ALPAKA_CREATE_VEC_IN_CLASS
                Vec<TDim, TSize>::template
#endif
                createVecFromIndexedFnOffset<
#ifndef ALPAKA_CREATE_VEC_IN_CLASS
                    TDim,
#endif
                    TTFnObj,
                    TIdxOffset>(
                        std::forward<TArgs>(args)...);
        }

        namespace detail
        {
            //#############################################################################
            //! A function object that returns the sum of the two input vectors elements.
            template<
                std::size_t Tidx>
            struct CreateAdd
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TDim,
                    typename TSize>
                ALPAKA_FN_HOST_ACC static auto create(
                    Vec<TDim, TSize> const & p,
                    Vec<TDim, TSize> const & q)
                -> TSize
                {
                    return p[Tidx] + q[Tidx];
                }
            };
        }
        //-----------------------------------------------------------------------------
        //! \return The element wise sum of two vectors.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TDim,
            typename TSize>
        ALPAKA_FN_HOST_ACC auto operator+(
            Vec<TDim, TSize> const & p,
            Vec<TDim, TSize> const & q)
        -> Vec<TDim, TSize>
        {
            return
                createVecFromIndexedFnWorkaround<
                    TDim,
                    TSize,
                    detail::CreateAdd>(
                        p,
                        q);
        }

        namespace detail
        {
            //##################################################################################
            //! A function object that returns the difference of the two input vectors elements.
            template<
                std::size_t Tidx>
            struct CreateSub
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TDim,
                    typename TSize>
                ALPAKA_FN_HOST_ACC static auto create(
                    Vec<TDim, TSize> const & p,
                    Vec<TDim, TSize> const & q)
                -> TSize
                {
                    return p[Tidx] - q[Tidx];
                }
            };
        }

        //-----------------------------------------------------------------------------
        //! \return The element wise difference of two vectors.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TDim,
            typename TSize>
        ALPAKA_FN_HOST_ACC auto operator-(
            Vec<TDim, TSize> const & p,
            Vec<TDim, TSize> const & q)
        -> Vec<TDim, TSize>
        {
            return
                createVecFromIndexedFnWorkaround<
                    TDim,
                    TSize,
                    detail::CreateSub>(
                        p,
                        q);
        }

        namespace detail
        {
            //#############################################################################
            //! A function object that returns the product of the two input vectors elements.
            template<
                std::size_t Tidx>
            struct CreateMul
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TDim,
                    typename TSize>
                ALPAKA_FN_HOST_ACC static auto create(
                    Vec<TDim, TSize> const & p,
                    Vec<TDim, TSize> const & q)
                -> TSize
                {
                    return p[Tidx] * q[Tidx];
                }
            };
        }

        //-----------------------------------------------------------------------------
        //! \return The element wise product of two vectors.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TDim,
            typename TSize>
        ALPAKA_FN_HOST_ACC auto operator*(
            Vec<TDim, TSize> const & p,
            Vec<TDim, TSize> const & q)
        -> Vec<TDim, TSize>
        {
            return
                createVecFromIndexedFnWorkaround<
                    TDim,
                    TSize,
                    detail::CreateMul>(
                        p,
                        q);
        }

        namespace detail
        {
            //#############################################################################
            //! A function object that returns the element wise less than relation of two vectors.
            template<
                std::size_t Tidx>
            struct CreateLessEqual
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TDim,
                    typename TSize>
                ALPAKA_FN_HOST_ACC static auto create(
                    Vec<TDim, TSize> const & p,
                    Vec<TDim, TSize> const & q)
                -> bool
                {
                    return p[Tidx] <= q[Tidx];
                }
            };
        }

        //-----------------------------------------------------------------------------
        //! \return The element wise less than relation of two vectors.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TDim,
            typename TSize>
        ALPAKA_FN_HOST_ACC auto operator<=(
            Vec<TDim, TSize> const & p,
            Vec<TDim, TSize> const & q)
        -> Vec<TDim, bool>
        {
            return
                createVecFromIndexedFnWorkaround<
                    TDim,
                    bool,
                    detail::CreateLessEqual>(
                        p,
                        q);
        }

        //-----------------------------------------------------------------------------
        //! Stream out operator.
        template<
            typename TDim,
            typename TSize>
        ALPAKA_FN_HOST auto operator<<(
            std::ostream & os,
            Vec<TDim, TSize> const & v)
        -> std::ostream &
        {
            os << "(";
            for(typename TDim::value_type i(0); i<TDim::value; ++i)
            {
                os << v[i];
                if(i != TDim::value-1)
                {
                    os << ", ";
                }
            }
            os << ")";

            return os;
        }
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The Vec dimension get trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct DimType<
                vec::Vec<TDim, TSize>>
            {
                using type = TDim;
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator work division size type trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                vec::Vec<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
    namespace vec
    {
        namespace traits
        {
            //#############################################################################
            //! Specialization for selecting a sub-vector.
            template<
                typename TDim,
                typename TSize,
                std::size_t... TIndices>
            struct SubVecFromIndices<
                Vec<TDim, TSize>,
                meta::IntegerSequence<std::size_t, TIndices...>,
                typename std::enable_if<
                    !std::is_same<
                        meta::IntegerSequence<std::size_t, TIndices...>,
                        meta::MakeIntegerSequence<std::size_t, TDim::value>
                    >::value
                >::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto subVecFromIndices(
                    Vec<TDim, TSize> const & vec)
                -> Vec<dim::DimInt<sizeof...(TIndices)>, TSize>
                {
#if !BOOST_ARCH_CUDA_DEVICE
                    // In the case of a zero dimensional vector, vec is unused.
                    boost::ignore_unused(vec);
#endif

                    static_assert(sizeof...(TIndices) <= TDim::value, "The sub-vector has to be smaller (or same size) then the origin vector.");

                    return Vec<dim::DimInt<sizeof...(TIndices)>, TSize>(vec[TIndices]...);
                }
            };
            //#############################################################################
            //! Specialization for selecting the whole vector.
            template<
                typename TDim,
                typename TSize>
            struct SubVecFromIndices<
                Vec<TDim, TSize>,
                meta::MakeIntegerSequence<std::size_t, TDim::value>>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto subVecFromIndices(
                    Vec<TDim, TSize> const & vec)
                -> Vec<TDim, TSize>
                {
                    return vec;
                }
            };
        }

        namespace detail
        {
            //#############################################################################
            //! A function object that returns the given value for each index.
            template<
                std::size_t Tidx>
            struct CreateCast
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TSizeNew,
                    typename TDim,
                    typename TSize>
                ALPAKA_FN_HOST_ACC static auto create(
                    TSizeNew const &/* valNew*/,
                    Vec<TDim, TSize> const & vec)
                -> TSizeNew
                {
                    return
                        static_cast<TSizeNew>(
                            vec[Tidx]);
                }
            };
        }
        namespace traits
        {
            //#############################################################################
            //! Cast specialization for Vec.
            template<
                typename TSizeNew,
                typename TDim,
                typename TSize>
            struct Cast<
                TSizeNew,
                Vec<TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto cast(
                    Vec<TDim, TSize> const & vec)
                -> Vec<TDim, TSizeNew>
                {
                    return
                        createVecFromIndexedFnWorkaround<
                            TDim,
                            TSizeNew,
                            vec::detail::CreateCast>(
                                TSizeNew(),
                                vec);
                }
            };

            //#############################################################################
            //! (Non-)Cast specialization for Vec when src and dst types are identical.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct Cast<
                TSize,
                Vec<TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto cast(
                    Vec<TDim, TSize> const & vec)
                -> Vec<TDim, TSize>
                {
                    return vec;
                }
            };
        }

        namespace detail
        {
            //#############################################################################
            //! A function object that returns the value at the index from the back of the vector.
            template<
                std::size_t Tidx>
            struct CreateReverse
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TDim,
                    typename TSize>
                ALPAKA_FN_HOST_ACC static auto create(
                    Vec<TDim, TSize> const & vec)
                -> TSize
                {
                    return vec[TDim::value - 1u - Tidx];
                }
            };
        }
        namespace traits
        {
            //#############################################################################
            //! Reverse specialization for Vec.
            template<
                typename TDim,
                typename TSize>
            struct Reverse<
                Vec<TDim, TSize>>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto reverse(
                    Vec<TDim, TSize> const & vec)
                -> Vec<TDim, TSize>
                {
                    return
                        createVecFromIndexedFnWorkaround<
                            TDim,
                            TSize,
                            vec::detail::CreateReverse>(
                                vec);
                }
            };

            //#############################################################################
            //! (Non-)Reverse specialization for 1D Vec.
            template<
                typename TSize>
            struct Reverse<
                Vec<dim::DimInt<1u>, TSize>>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto reverse(
                    Vec<dim::DimInt<1u>, TSize> const & vec)
                -> Vec<dim::DimInt<1u>, TSize>
                {
                    return vec;
                }
            };
        }

        namespace detail
        {
            //#############################################################################
            //! A function object that returns the value at the index from the back of the vector.
            template<
                std::size_t Tidx>
            struct CreateConcat
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TDimL,
                    typename TDimR,
                    typename TSize>
                ALPAKA_FN_HOST_ACC static auto create(
                    Vec<TDimL, TSize> const & vecL,
                    Vec<TDimR, TSize> const & vecR)
                -> TSize
                {
                    return Tidx < TDimL::value ? vecL[Tidx] : vecR[Tidx - TDimL::value];
                }
            };
        }
        namespace traits
        {
            //#############################################################################
            //! Concatenation specialization for Vec.
            template<
                typename TDimL,
                typename TDimR,
                typename TSize>
            struct Concat<
                Vec<TDimL, TSize>,
                Vec<TDimR, TSize>>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto concat(
                    Vec<TDimL, TSize> const & vecL,
                    Vec<TDimR, TSize> const & vecR)
                -> Vec<dim::DimInt<TDimL::value + TDimR::value>, TSize>
                {
                    return
                        createVecFromIndexedFnWorkaround<
                            dim::DimInt<TDimL::value + TDimR::value>,
                            TSize,
                            vec::detail::CreateConcat>(
                                vecL,
                                vecR);
                }
            };
        }
    }

    namespace extent
    {
        namespace detail
        {
            //#############################################################################
            //! A function object that returns the extent for each index.
            template<
                std::size_t Tidx>
            struct CreateExtent
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TExtent>
                ALPAKA_FN_HOST_ACC static auto create(
                    TExtent const & extent)
                -> size::Size<TExtent>
                {
                    return extent::getExtent<Tidx>(extent);
                }
            };
        }
        //-----------------------------------------------------------------------------
        //! \tparam TExtent has to specialize extent::GetExtent.
        //! \return The extent vector.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TExtent>
        ALPAKA_FN_HOST_ACC auto getExtentVec(
            TExtent const & extent = TExtent())
        -> vec::Vec<dim::Dim<TExtent>, size::Size<TExtent>>
        {
            return
                vec::createVecFromIndexedFnWorkaround<
                    dim::Dim<TExtent>,
                    size::Size<TExtent>,
                    detail::CreateExtent>(
                        extent);
        }
        //-----------------------------------------------------------------------------
        //! \tparam TExtent has to specialize extent::GetExtent.
        //! \return The extent but only the last N elements.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TDim,
            typename TExtent>
        ALPAKA_FN_HOST_ACC auto getExtentVecEnd(
            TExtent const & extent = TExtent())
        -> vec::Vec<TDim, size::Size<TExtent>>
        {
            using IdxOffset = std::integral_constant<std::intmax_t, static_cast<std::intmax_t>(dim::Dim<TExtent>::value) - static_cast<std::intmax_t>(TDim::value)>;
            return
                vec::createVecFromIndexedFnOffsetWorkaround<
                    TDim,
                    size::Size<TExtent>,
                    detail::CreateExtent,
                    IdxOffset>(
                        extent);
        }
    }

    namespace offset
    {
        namespace detail
        {
            //#############################################################################
            //! A function object that returns the offsets for each index.
            template<
                std::size_t Tidx>
            struct CreateOffset
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TOffsets>
                ALPAKA_FN_HOST_ACC static auto create(
                    TOffsets const & offsets)
                -> size::Size<TOffsets>
                {
                    return offset::getOffset<Tidx>(offsets);
                }
            };
        }
        //-----------------------------------------------------------------------------
        //! \tparam TOffsets has to specialize offset::GetOffset.
        //! \return The offset vector.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TOffsets>
        ALPAKA_FN_HOST_ACC auto getOffsetVec(
            TOffsets const & offsets = TOffsets())
        -> vec::Vec<dim::Dim<TOffsets>, size::Size<TOffsets>>
        {
            return
                vec::createVecFromIndexedFnWorkaround<
                    dim::Dim<TOffsets>,
                    size::Size<TOffsets>,
                    detail::CreateOffset>(
                        offsets);
        }
        //-----------------------------------------------------------------------------
        //! \tparam TOffsets has to specialize offset::GetOffset.
        //! \return The offset vector but only the last N elements.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TDim,
            typename TOffsets>
        ALPAKA_FN_HOST_ACC auto getOffsetVecEnd(
            TOffsets const & offsets = TOffsets())
        -> vec::Vec<TDim, size::Size<TOffsets>>
        {
            using IdxOffset = std::integral_constant<std::size_t, static_cast<std::size_t>(static_cast<std::intmax_t>(dim::Dim<TOffsets>::value) - static_cast<std::intmax_t>(TDim::value))>;
            return
                vec::createVecFromIndexedFnOffsetWorkaround<
                    TDim,
                    size::Size<TOffsets>,
                    detail::CreateOffset,
                    IdxOffset>(
                        offsets);
        }
    }
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The Vec extent get trait specialization.
            template<
                typename TIdx,
                typename TDim,
                typename TSize>
            struct GetExtent<
                TIdx,
                vec::Vec<TDim, TSize>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getExtent(
                    vec::Vec<TDim, TSize> const & extent)
                -> TSize
                {
                    return extent[TIdx::value];
                }
            };
            //#############################################################################
            //! The Vec extent set trait specialization.
            template<
                typename TIdx,
                typename TDim,
                typename TSize,
                typename TExtentVal>
            struct SetExtent<
                TIdx,
                vec::Vec<TDim, TSize>,
                TExtentVal,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto setExtent(
                    vec::Vec<TDim, TSize> & extent,
                    TExtentVal const & extentVal)
                -> void
                {
                    extent[TIdx::value] = extentVal;
                }
            };
        }
    }
    namespace offset
    {
        namespace traits
        {
            //#############################################################################
            //! The Vec offset get trait specialization.
            template<
                typename TIdx,
                typename TDim,
                typename TSize>
            struct GetOffset<
                TIdx,
                vec::Vec<TDim, TSize>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getOffset(
                    vec::Vec<TDim, TSize> const & offsets)
                -> TSize
                {
                    return offsets[TIdx::value];
                }
            };
            //#############################################################################
            //! The Vec offset set trait specialization.
            template<
                typename TIdx,
                typename TDim,
                typename TSize,
                typename TOffset>
            struct SetOffset<
                TIdx,
                vec::Vec<TDim, TSize>,
                TOffset,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto setOffset(
                    vec::Vec<TDim, TSize> & offsets,
                    TOffset const & offset)
                -> void
                {
                    offsets[TIdx::value] = offset;
                }
            };
        }
    }
}
