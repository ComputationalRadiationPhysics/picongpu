/* Copyright 2023 Axel Huebl, Benjamin Worpitz, Erik Zenker, Matthias Werner, René Widera, Andrea Bocci, Jan Stephan,
 *                Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Align.hpp"
#include "alpaka/core/Assert.hpp"
#include "alpaka/core/BoostPredef.hpp"
#include "alpaka/core/Common.hpp"
#include "alpaka/core/Unreachable.hpp"
#include "alpaka/dim/DimIntegralConst.hpp"
#include "alpaka/dim/Traits.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/meta/Fold.hpp"
#include "alpaka/meta/Functional.hpp"
#include "alpaka/meta/IntegerSequence.hpp"
#include "alpaka/vec/Traits.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <ostream>
#include <tuple>
#include <type_traits>
#include <utility>

namespace alpaka
{
    template<typename TDim, typename TVal>
    class Vec;

    //! A n-dimensional vector.
    template<typename TDim, typename TVal>
    class Vec final
    {
    public:
        static_assert(TDim::value >= 0u, "Invalid dimensionality");

        using Dim = TDim;
        using Val = TVal;
        using value_type = Val; //!< STL-like value_type.

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
#if BOOST_COMP_NVCC && BOOST_COMP_NVCC >= BOOST_VERSION_NUMBER(11, 3, 0)                                              \
    && BOOST_COMP_NVCC < BOOST_VERSION_NUMBER(11, 4, 0)
        // This constructor tries to avoid SFINAE, which crashes nvcc 11.3. We also need to have a first
        // argument, so an unconstrained ctor with forwarding references does not hijack the compiler provided
        // copy-ctor.
        template<typename... TArgs>
        ALPAKA_FN_HOST_ACC constexpr Vec(TVal arg0, TArgs&&... args)
            : m_data{std::move(arg0), static_cast<TVal>(std::forward<TArgs>(args))...}
        {
            static_assert(
                1 + sizeof...(TArgs) == TDim::value && (std::is_convertible_v<std::decay_t<TArgs>, TVal> && ...),
                "Wrong number of arguments to Vec constructor or types are not convertible to TVal.");
        }
#else
        template<
            typename... TArgs,
            typename = std::enable_if_t<
                sizeof...(TArgs) == TDim::value && (std::is_convertible_v<std::decay_t<TArgs>, TVal> && ...)>>
        ALPAKA_FN_HOST_ACC constexpr Vec(TArgs&&... args) : m_data{static_cast<TVal>(std::forward<TArgs>(args))...}
        {
        }
#endif

        //! Generator constructor.
        //! Initializes the vector with the values returned from generator(IC) in order, where IC::value runs from 0 to
        //! TDim - 1 (inclusive).
#if BOOST_COMP_NVCC && BOOST_COMP_NVCC >= BOOST_VERSION_NUMBER(11, 3, 0)                                              \
    && BOOST_COMP_NVCC < BOOST_VERSION_NUMBER(11, 4, 0)
        template<typename F>
        ALPAKA_FN_HOST_ACC constexpr explicit Vec(
            F&& generator,
            std::void_t<decltype(generator(std::integral_constant<std::size_t, 0>{}))>* ignore = nullptr)
            : Vec(std::forward<F>(generator), std::make_index_sequence<TDim::value>{})
        {
            static_cast<void>(ignore);
        }
#else
        template<typename F, std::enable_if_t<std::is_invocable_v<F, std::integral_constant<std::size_t, 0>>, int> = 0>
        ALPAKA_FN_HOST_ACC constexpr explicit Vec(F&& generator)
            : Vec(std::forward<F>(generator), std::make_index_sequence<TDim::value>{})
        {
        }
#endif

    private:
        template<typename F, std::size_t... Is>
        ALPAKA_FN_HOST_ACC constexpr explicit Vec(F&& generator, std::index_sequence<Is...>)
            : m_data{generator(std::integral_constant<std::size_t, Is>{})...}
        {
        }

    public:
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

        ALPAKA_FN_HOST_ACC constexpr auto begin() const -> TVal const*
        {
            return m_data;
        }

        ALPAKA_FN_HOST_ACC constexpr auto end() -> TVal*
        {
            return m_data + TDim::value;
        }

        ALPAKA_FN_HOST_ACC constexpr auto end() const -> TVal const*
        {
            return m_data + TDim::value;
        }

        ALPAKA_FN_HOST_ACC constexpr auto front() -> TVal&
        {
            return m_data[0];
        }

        ALPAKA_FN_HOST_ACC constexpr auto front() const -> TVal const&
        {
            return m_data[0];
        }

        ALPAKA_FN_HOST_ACC constexpr auto back() -> TVal&
        {
            return m_data[Dim::value - 1];
        }

        ALPAKA_FN_HOST_ACC constexpr auto back() const -> TVal const&
        {
            return m_data[Dim::value - 1];
        }

        //! access elements by name
        //!
        //! names: x,y,z,w
        //! @{
        template<typename TDefer = Dim, std::enable_if_t<std::is_same_v<TDefer, Dim> && Dim::value >= 1, int> = 0>
        ALPAKA_FN_HOST_ACC constexpr decltype(auto) x() const
        {
            return m_data[Dim::value - 1];
        }

        template<typename TDefer = Dim, std::enable_if_t<std::is_same_v<TDefer, Dim> && Dim::value >= 1, int> = 0>
        ALPAKA_FN_HOST_ACC constexpr decltype(auto) x()
        {
            return m_data[Dim::value - 1];
        }

        template<typename TDefer = Dim, std::enable_if_t<std::is_same_v<TDefer, Dim> && Dim::value >= 2, int> = 0>
        ALPAKA_FN_HOST_ACC constexpr decltype(auto) y() const
        {
            return m_data[Dim::value - 2];
        }

        template<typename TDefer = Dim, std::enable_if_t<std::is_same_v<TDefer, Dim> && Dim::value >= 2, int> = 0>
        ALPAKA_FN_HOST_ACC constexpr decltype(auto) y()
        {
            return m_data[Dim::value - 2];
        }

        template<typename TDefer = Dim, std::enable_if_t<std::is_same_v<TDefer, Dim> && Dim::value >= 3, int> = 0>
        ALPAKA_FN_HOST_ACC constexpr decltype(auto) z() const
        {
            return m_data[Dim::value - 3];
        }

        template<typename TDefer = Dim, std::enable_if_t<std::is_same_v<TDefer, Dim> && Dim::value >= 3, int> = 0>
        ALPAKA_FN_HOST_ACC constexpr decltype(auto) z()
        {
            return m_data[Dim::value - 3];
        }

        template<typename TDefer = Dim, std::enable_if_t<std::is_same_v<TDefer, Dim> && Dim::value >= 4, int> = 0>
        ALPAKA_FN_HOST_ACC constexpr decltype(auto) w() const
        {
            return m_data[Dim::value - 4];
        }

        template<typename TDefer = Dim, std::enable_if_t<std::is_same_v<TDefer, Dim> && Dim::value >= 4, int> = 0>
        ALPAKA_FN_HOST_ACC constexpr decltype(auto) w()
        {
            return m_data[Dim::value - 4];
        }

        //! @}

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
            return foldrAll(std::multiplies<TVal>{}, TVal{1});
        }
#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#    pragma warning(pop)
#endif
        //! \return The sum of all values.
        ALPAKA_NO_HOST_ACC_WARNING
        [[nodiscard]] ALPAKA_FN_HOST_ACC constexpr auto sum() const -> TVal
        {
            return foldrAll(std::plus<TVal>{}, TVal{0});
        }

        //! \return The min of all values.
        ALPAKA_NO_HOST_ACC_WARNING
        [[nodiscard]] ALPAKA_FN_HOST_ACC constexpr auto min() const -> TVal
        {
            return foldrAll(meta::min<TVal>{}, std::numeric_limits<TVal>::max());
        }

        //! \return The max of all values.
        ALPAKA_NO_HOST_ACC_WARNING
        [[nodiscard]] ALPAKA_FN_HOST_ACC constexpr auto max() const -> TVal
        {
            return foldrAll(meta::max<TVal>{}, std::numeric_limits<TVal>::min());
        }

        //! \return True if all values are true, i.e., the "logical and" of all values.
        ALPAKA_NO_HOST_ACC_WARNING
        [[nodiscard]] ALPAKA_FN_HOST_ACC constexpr auto all() const -> bool
        {
            return foldrAll(std::logical_and<TVal>{}, true);
        }

        //! \return True if any value is true, i.e., the "logical or" of all values.
        ALPAKA_NO_HOST_ACC_WARNING
        [[nodiscard]] ALPAKA_FN_HOST_ACC constexpr auto any() const -> bool
        {
            return foldrAll(std::logical_or<TVal>{}, false);
        }

        //! \return True if none of the values are true
        ALPAKA_NO_HOST_ACC_WARNING
        [[nodiscard]] ALPAKA_FN_HOST_ACC constexpr auto none() const -> bool
        {
            return !foldrAll(std::logical_or<TVal>{}, false);
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
#if BOOST_COMP_NVCC && BOOST_COMP_NVCC < BOOST_VERSION_NUMBER(11, 3, 0)
            if(TDim::value > 0)
#else
            if constexpr(TDim::value > 0)
#endif
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
#if BOOST_COMP_NVCC && BOOST_COMP_NVCC < BOOST_VERSION_NUMBER(11, 3, 0)
            if(TDim::value > 0)
#else
            if constexpr(TDim::value > 0)
#endif
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
#if BOOST_COMP_NVCC && BOOST_COMP_NVCC < BOOST_VERSION_NUMBER(11, 3, 0)
            if(TDim::value > 0)
#else
            if constexpr(TDim::value > 0)
#endif
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
#if BOOST_COMP_NVCC && BOOST_COMP_NVCC < BOOST_VERSION_NUMBER(11, 3, 0)
            if(TDim::value > 0)
#else
            if constexpr(TDim::value > 0)
#endif
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
#if BOOST_COMP_NVCC && BOOST_COMP_NVCC < BOOST_VERSION_NUMBER(11, 3, 0)
            if(TDim::value > 0)
#else
            if constexpr(TDim::value > 0)
#endif
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
#if BOOST_COMP_NVCC && BOOST_COMP_NVCC < BOOST_VERSION_NUMBER(11, 3, 0)
            if(TDim::value > 0)
#else
            if constexpr(TDim::value > 0)
#endif
            {
                for(typename TDim::value_type i = 0; i < TDim::value; ++i)
                    r[i] = p[i] >= q[i];
            }
            return r;
        }

        //! \return The element-wise logical and relation of two vectors.
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC friend constexpr auto operator&&(Vec const& p, Vec const& q) -> Vec<TDim, bool>
        {
            Vec<TDim, bool> r;
#if BOOST_COMP_NVCC && BOOST_COMP_NVCC < BOOST_VERSION_NUMBER(11, 3, 0)
            if(TDim::value > 0)
#else
            if constexpr(TDim::value > 0)
#endif
            {
                for(typename TDim::value_type i = 0; i < TDim::value; ++i)
                    r[i] = p[i] && q[i];
            }
            return r;
        }

        //! \return The element-wise logical or relation of two vectors.
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC friend constexpr auto operator||(Vec const& p, Vec const& q) -> Vec<TDim, bool>
        {
            Vec<TDim, bool> r;
#if BOOST_COMP_NVCC && BOOST_COMP_NVCC < BOOST_VERSION_NUMBER(11, 3, 0)
            if(TDim::value > 0)
#else
            if constexpr(TDim::value > 0)
#endif
            {
                for(typename TDim::value_type i = 0; i < TDim::value; ++i)
                    r[i] = p[i] || q[i];
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

    template<typename TFirstIndex, typename... TRestIndices>
    Vec(TFirstIndex&&, TRestIndices&&...) -> Vec<DimInt<1 + sizeof...(TRestIndices)>, std::decay_t<TFirstIndex>>;

    template<typename T>
    inline constexpr bool isVec = false;

    template<typename TDim, typename TVal>
    inline constexpr bool isVec<Vec<TDim, TVal>> = true;

    //! Converts a Vec to a std::array
    template<typename TDim, typename TVal>
    ALPAKA_FN_HOST_ACC constexpr auto toArray(Vec<TDim, TVal> const& v) -> std::array<TVal, TDim::value>
    {
        std::array<TVal, TDim::value> a{};
#if BOOST_COMP_NVCC && BOOST_COMP_NVCC < BOOST_VERSION_NUMBER(11, 3, 0)
        if(TDim::value > 0)
#else
        if constexpr(TDim::value > 0)
#endif
        {
            for(unsigned i = 0; i < TDim::value; i++)
                a[i] = v[i];
        }
        return a;
    }

    //! \return The element-wise minimum of one or more vectors.
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TDim,
        typename TVal,
        typename... Vecs,
        typename = std::enable_if_t<(std::is_same_v<Vec<TDim, TVal>, Vecs> && ...)>>
    ALPAKA_FN_HOST_ACC constexpr auto elementwise_min(Vec<TDim, TVal> const& p, Vecs const&... qs) -> Vec<TDim, TVal>
    {
        Vec<TDim, TVal> r;
#if BOOST_COMP_NVCC && BOOST_COMP_NVCC < BOOST_VERSION_NUMBER(11, 3, 0)
        if(TDim::value > 0)
#else
        if constexpr(TDim::value > 0)
#endif
        {
            for(typename TDim::value_type i = 0; i < TDim::value; ++i)
                r[i] = std::min({p[i], qs[i]...});
        }
        return r;
    }

    //! \return The element-wise maximum of one or more vectors.
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TDim,
        typename TVal,
        typename... Vecs,
        typename = std::enable_if_t<(std::is_same_v<Vec<TDim, TVal>, Vecs> && ...)>>
    ALPAKA_FN_HOST_ACC constexpr auto elementwise_max(Vec<TDim, TVal> const& p, Vecs const&... qs) -> Vec<TDim, TVal>
    {
        Vec<TDim, TVal> r;
#if BOOST_COMP_NVCC && BOOST_COMP_NVCC < BOOST_VERSION_NUMBER(11, 3, 0)
        if(TDim::value > 0)
#else
        if constexpr(TDim::value > 0)
#endif
        {
            for(typename TDim::value_type i = 0; i < TDim::value; ++i)
                r[i] = std::max({p[i], qs[i]...});
        }
        return r;
    }

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
#if BOOST_COMP_NVCC && BOOST_COMP_NVCC < BOOST_VERSION_NUMBER(11, 3, 0)
                    if(TDim::value > 0)
#else
                    if constexpr(TDim::value > 0)
#endif
                    {
                        for(typename TDim::value_type i = 0; i < TDim::value; ++i)
                            r[i] = static_cast<TValNew>(vec[i]);
                    }
                    return r;
                }
                ALPAKA_UNREACHABLE({});
            }
        };

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
