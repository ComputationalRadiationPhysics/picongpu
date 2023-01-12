/* Copyright 2022 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <alpaka/atomic/Op.hpp>
#    include <alpaka/atomic/Traits.hpp>
#    include <alpaka/core/Positioning.hpp>
#    include <alpaka/meta/DependentFalseType.hpp>

#    include <CL/sycl.hpp>

#    include <cstdint>
#    include <type_traits>

namespace alpaka::experimental
{
    //! The SYCL accelerator atomic ops.
    //
    //  Atomics can used in the hierarchy level grids, blocks and threads.
    //  Atomics are not guaranteed to be safe between devices
    class AtomicGenericSycl
    {
    };

    namespace detail
    {
        template<typename THierarchy>
        struct SyclMemoryScope
        {
        };

        template<>
        struct SyclMemoryScope<hierarchy::Grids>
        {
            static constexpr auto value = sycl::memory_scope::device;
        };

        template<>
        struct SyclMemoryScope<hierarchy::Blocks>
        {
            static constexpr auto value = sycl::memory_scope::device;
        };

        template<>
        struct SyclMemoryScope<hierarchy::Threads>
        {
            static constexpr auto value = sycl::memory_scope::work_group;
        };

        template<typename T>
        inline auto get_global_ptr(T* const addr)
        {
            return sycl::make_ptr<T, sycl::access::address_space::global_space>(addr);
        }

        template<typename T>
        inline auto get_local_ptr(T* const addr)
        {
            return sycl::make_ptr<T, sycl::access::address_space::local_space>(addr);
        }

        // atomic_ref is already part of the SYCL spec but oneAPI has not caught up yet.
        template<typename T, typename THierarchy>
        using global_ref = sycl::ext::oneapi::atomic_ref<
            T,
            sycl::ext::oneapi::memory_order::relaxed,
            SyclMemoryScope<THierarchy>::value,
            sycl::access::address_space::global_space>;

        template<typename T, typename THierarchy>
        using local_ref = sycl::ext::oneapi::atomic_ref<
            T,
            sycl::ext::oneapi::memory_order::relaxed,
            SyclMemoryScope<THierarchy>::value,
            sycl::access::address_space::local_space>;

        template<typename THierarchy, typename T, typename TOp>
        inline auto callAtomicOp(T* const addr, TOp&& op)
        {
            if(auto ptr = get_global_ptr(addr); ptr != nullptr)
            {
                auto ref = global_ref<T, THierarchy>{*addr};
                return op(ref);
            }
            else
            {
                auto ref = local_ref<T, THierarchy>{*addr};
                return op(ref);
            }
        }
    } // namespace detail
} // namespace alpaka::experimental

namespace alpaka::trait
{
    // Add.
    //! The SYCL accelerator atomic operation.
    template<typename T, typename THierarchy>
    struct AtomicOp<AtomicAdd, experimental::AtomicGenericSycl, T, THierarchy>
    {
        static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>, "SYCL atomics do not support this type");

        static auto atomicOp(experimental::AtomicGenericSycl const&, T* const addr, T const& value) -> T
        {
            return experimental::detail::callAtomicOp<THierarchy>(
                addr,
                [&value](auto& ref) { return ref.fetch_add(value); });
        }
    };

    // Sub.
    //! The SYCL accelerator atomic operation.
    template<typename T, typename THierarchy>
    struct AtomicOp<AtomicSub, experimental::AtomicGenericSycl, T, THierarchy>
    {
        static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>, "SYCL atomics do not support this type");

        static auto atomicOp(experimental::AtomicGenericSycl const&, T* const addr, T const& value) -> T
        {
            return experimental::detail::callAtomicOp<THierarchy>(
                addr,
                [&value](auto& ref) { return ref.fetch_sub(value); });
        }
    };

    // Min.
    //! The SYCL accelerator atomic operation.
    template<typename T, typename THierarchy>
    struct AtomicOp<AtomicMin, experimental::AtomicGenericSycl, T, THierarchy>
    {
        static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>, "SYCL atomics do not support this type");

        static auto atomicOp(experimental::AtomicGenericSycl const&, T* const addr, T const& value) -> T
        {
            return experimental::detail::callAtomicOp<THierarchy>(
                addr,
                [&value](auto& ref) { return ref.fetch_min(value); });
        }
    };

    // Max.
    //! The SYCL accelerator atomic operation.
    template<typename T, typename THierarchy>
    struct AtomicOp<AtomicMax, experimental::AtomicGenericSycl, T, THierarchy>
    {
        static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>, "SYCL atomics do not support this type");

        static auto atomicOp(experimental::AtomicGenericSycl const&, T* const addr, T const& value) -> T
        {
            return experimental::detail::callAtomicOp<THierarchy>(
                addr,
                [&value](auto& ref) { return ref.fetch_max(value); });
        }
    };

    // Exch.
    //! The SYCL accelerator atomic operation.
    template<typename T, typename THierarchy>
    struct AtomicOp<AtomicExch, experimental::AtomicGenericSycl, T, THierarchy>
    {
        static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>, "SYCL atomics do not support this type");

        static auto atomicOp(experimental::AtomicGenericSycl const&, T* const addr, T const& value) -> T
        {
            return experimental::detail::callAtomicOp<THierarchy>(
                addr,
                [&value](auto& ref) { return ref.exchange(value); });
        }
    };

    namespace detail
    {
        template<typename TRef, typename T, typename TEval>
        inline auto casWithCondition(T* const addr, TEval&& eval)
        {
            auto ref = TRef{*addr};

            auto old_val = ref.load();
            auto assumed = T{};

            do
            {
                assumed = old_val;
                auto const new_val = eval(old_val);
                old_val = ref.compare_exchange_strong(assumed, new_val);
            } while(assumed != old_val);


            return old_val;
        }
    } // namespace detail

    // Inc.
    //! The SYCL accelerator atomic operation.
    template<typename T, typename THierarchy>
    struct AtomicOp<AtomicInc, experimental::AtomicGenericSycl, T, THierarchy>
    {
        static_assert(std::is_unsigned_v<T>, "atomicInc only supported for unsigned types");

        static auto atomicOp(experimental::AtomicGenericSycl const&, T* const addr, T const& value) -> T
        {
            auto inc = [&value](auto old_val) { return (old_val >= value) ? static_cast<T>(0) : (old_val + 1u); };
            if(auto ptr = get_global_ptr(addr); ptr != nullptr)
                return detail::casWithCondition<experimental::detail::global_ref<T, THierarchy>>(addr, inc);
            else
                return detail::casWithCondition<experimental::detail::local_ref<T, THierarchy>>(addr, inc);
        }
    };

    // Dec.
    //! The SYCL accelerator atomic operation.
    template<typename T, typename THierarchy>
    struct AtomicOp<AtomicDec, experimental::AtomicGenericSycl, T, THierarchy>
    {
        static_assert(std::is_unsigned_v<T>, "atomicDec only supported for unsigned types");

        static auto atomicOp(experimental::AtomicGenericSycl const&, T* const addr, T const& value) -> T
        {
            auto dec
                = [&value](auto& old_val) { return ((old_val == 0) || (old_val > value)) ? value : (old_val - 1u); };
            if(auto ptr = get_global_ptr(addr); ptr != nullptr)
                return detail::casWithCondition<experimental::detail::global_ref<T, THierarchy>>(addr, dec);
            else
                return detail::casWithCondition<experimental::detail::local_ref<T, THierarchy>>(addr, dec);
        }
    };

    // And.
    //! The SYCL accelerator atomic operation.
    template<typename T, typename THierarchy>
    struct AtomicOp<AtomicAnd, experimental::AtomicGenericSycl, T, THierarchy>
    {
        static_assert(std::is_integral_v<T>, "Bitwise operations only supported for integral types.");

        static auto atomicOp(experimental::AtomicGenericSycl const&, T* const addr, T const& value) -> T
        {
            return experimental::detail::callAtomicOp<THierarchy>(
                addr,
                [&value](auto& ref) { return ref.fetch_and(value); });
        }
    };

    // Or.
    //! The SYCL accelerator atomic operation.
    template<typename T, typename THierarchy>
    struct AtomicOp<AtomicOr, experimental::AtomicGenericSycl, T, THierarchy>
    {
        static_assert(std::is_integral_v<T>, "Bitwise operations only supported for integral types.");

        static auto atomicOp(experimental::AtomicGenericSycl const&, T* const addr, T const& value) -> T
        {
            return experimental::detail::callAtomicOp<THierarchy>(
                addr,
                [&value](auto& ref) { return ref.fetch_or(value); });
        }
    };

    // Xor.
    //! The SYCL accelerator atomic operation.
    template<typename T, typename THierarchy>
    struct AtomicOp<AtomicXor, experimental::AtomicGenericSycl, T, THierarchy>
    {
        static_assert(std::is_integral_v<T>, "Bitwise operations only supported for integral types.");

        static auto atomicOp(experimental::AtomicGenericSycl const&, T* const addr, T const& value) -> T
        {
            return experimental::detail::callAtomicOp<THierarchy>(
                addr,
                [&value](auto& ref) { return ref.fetch_xor(value); });
        }
    };

    // Cas.
    //! The SYCL accelerator atomic operation.
    template<typename T, typename THierarchy>
    struct AtomicOp<AtomicCas, experimental::AtomicGenericSycl, T, THierarchy>
    {
        static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>, "SYCL atomics do not support this type");

        static auto atomicOp(experimental::AtomicGenericSycl const&, T* const addr, T const& compare, T const& value)
            -> T
        {
            auto cas = [&compare, &value](auto& ref)
            {
                // SYCL stores the value in *addr to the "compare" parameter if the values are not equal. Since
                // alpaka's interface does not expect this we need to copy "compare" to this function and forget it
                // afterwards.
                auto tmp = compare;

                // We always want to return the old value at the end.
                const auto old = ref.load();

                // This returns a bool telling us if the exchange happened or not. Useless in this case.
                ref.compare_exchange_strong(tmp, value);

                return old;
            };

            if(auto ptr = get_global_ptr(addr); ptr != nullptr)
            {
                auto ref = experimental::detail::global_ref<T, THierarchy>{*addr};
                return cas(ref);
            }
            else
            {
                auto ref = experimental::detail::local_ref<T, THierarchy>{*addr};
                return cas(ref);
            }
        }
    };
} // namespace alpaka::trait

#endif
