/* Copyright 2025 Andrea Bocci, Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/Tag.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/dim/DimIntegralConst.hpp"
#include "alpaka/dim/Traits.hpp"
#include "alpaka/elem/Traits.hpp"
#include "alpaka/exec/UniformElements.hpp"
#include "alpaka/extent/Traits.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/kernel/Traits.hpp"
#include "alpaka/mem/view/Traits.hpp"
#include "alpaka/vec/Vec.hpp"
#include "alpaka/workdiv/WorkDivHelpers.hpp"

#include <iterator>
#include <type_traits>

namespace alpaka
{

    namespace detail
    {

        template<typename TFn>
        struct TransformKernel
        {
            TFn fn;

            template<typename TAcc, typename T>
            ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* in_ptr, T* out_ptr, alpaka::Idx<TAcc> size) const
            {
                static_assert(std::is_invocable_r_v<T, TFn, T> or std::is_invocable_r_v<T, TFn, TAcc const&, T>);

                static_assert(alpaka::Dim<TAcc>::value == 1u);
                using Idx = alpaka::Idx<TAcc>;

                for(Idx i : alpaka::uniformElements(acc, size))
                {
                    if constexpr(std::is_invocable_r_v<T, TFn, T>)
                    {
                        // std::is_invocable_r_v<T, TFn, T>
                        out_ptr[i] = fn(in_ptr[i]);
                    }
                    else
                    {
                        // std::is_invocable_r_v<T, TFn, TAcc const&, T>
                        out_ptr[i] = fn(acc, in_ptr[i]);
                    }
                }
            }
        };

        template<typename TFn>
        struct TransformKernelND
        {
            TFn fn;

            template<typename TAcc, typename T>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                T const* in_ptr,
                alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> in_pithces,
                T* out_ptr,
                alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> out_pitches,
                alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> in_size) const
            {
                static_assert(std::is_invocable_r_v<T, TFn, T> or std::is_invocable_r_v<T, TFn, TAcc const&, T>);

                using Dim = alpaka::Dim<TAcc>;
                using Idx = alpaka::Idx<TAcc>;
                using Vec = alpaka::Vec<Dim, Idx>;

                for(Vec idx : alpaka::uniformElementsND(acc, in_size))
                {
                    auto p_in = reinterpret_cast<T const*>(
                        reinterpret_cast<uintptr_t>(in_ptr) + static_cast<uintptr_t>((idx * in_pithces).sum()));
                    auto p_out = reinterpret_cast<T*>(
                        reinterpret_cast<uintptr_t>(out_ptr) + static_cast<uintptr_t>((idx * out_pitches).sum()));
                    if constexpr(std::is_invocable_r_v<T, TFn, T>)
                    {
                        // std::is_invocable_r_v<T, TFn, T>
                        *p_out = fn(*p_in);
                    }
                    else
                    {
                        // std::is_invocable_r_v<T, TFn, TAcc const&, T>
                        *p_out = fn(acc, *p_in);
                    }
                }
            }
        };

    } // namespace detail

    /*
     * Applies asynchronously the given function `fn` to the elements of the input range starting at `in`,
     * and stores the result in the semi-open output range [`out_begin`,`out_end`), using the accelerator
     * back-end identified by `Tag`.
     */
    template<alpaka::concepts::Tag TTag, typename TQueue, typename T, typename TFn>
    void transform(TQueue& queue, T* out_begin, T* out_end, TFn&& fn, T* in)
    {
        using Idx = typename std::iterator_traits<T*>::difference_type;
        using Acc1D = alpaka::TagToAcc<TTag, alpaka::DimInt<1>, Idx>;

        static_assert(
            std::is_invocable_r_v<T, TFn, T> or std::is_invocable_r_v<T, TFn, Acc1D const&, T>,
            "TFn must accept either one argument (of type T) or two arguments (an accelerator and an argument of type "
            "T), and return a value of type T.");

        Idx size = std::distance(out_begin, out_end);
        detail::TransformKernel<TFn> kernel{fn};

        // Find a valid work division. This could be further optimised.
        auto const config
            = alpaka::KernelCfg<Acc1D>{size, Idx{1}, false, alpaka::GridBlockExtentSubDivRestrictions::Unrestricted};
        auto const grid = alpaka::getValidWorkDiv(config, alpaka::getDev(queue), kernel, in, out_begin, size);

        // Apply the fn function to all elements of the input range.
        alpaka::exec<Acc1D>(queue, grid, kernel, in, out_begin, size);
    }

    /*
     * Applies asynchronously the given function `fn` to the elements of the input buffer `in`,
     * and stores the result in the corresponding elements of the output buffer `out`,
     * using the accelerator back-end identified by `Tag`.
     */
    template<alpaka::concepts::Tag TTag, typename TQueue, typename TBuf, typename TFn, typename TConstBuf>
    void transform(TQueue& queue, TBuf& out, TFn&& fn, TConstBuf const& in)
    {
        // Check that the input and output buffers have compatible types.
        using Idx = alpaka::Idx<TConstBuf>;
        static_assert(
            std::is_same_v<alpaka::Idx<TBuf>, Idx>,
            "The input and output buffers must have the same index type.");
        using Dim = alpaka::Dim<TConstBuf>;
        static_assert(
            std::is_same_v<alpaka::Dim<TBuf>, Dim>,
            "The input and output buffers must have the same dimension.");
        using In = std::remove_const_t<alpaka::Elem<TConstBuf>>;
        using Out = alpaka::Elem<TBuf>;
        using Vec = alpaka::Vec<Dim, Idx>;
        using Acc = alpaka::TagToAcc<TTag, Dim, Idx>;

        static_assert(
            std::is_invocable_r_v<Out, TFn, In const> or std::is_invocable_r_v<Out, TFn, Acc const&, In const>,
            "TFn must accept either one argument (of the buffer's element type) or two arguments (an accelerator and "
            "the element type), and return a value of the buffer's element type.");

        // Check that the input and output buffers have the same size.
        Vec size = alpaka::getExtents(in);
        assert(alpaka::getExtents(out) == size and "The input and output buffers must have the same extents.");

        // Pass details of the input and output buffers to the kernel:
        //   - address of the first elements
        //   - pitches (in bytes) along all dimensions
        //   - number of elements along all dimensions
        detail::TransformKernelND<TFn> kernel{fn};

        // Find a valid work division. This could be further optimised.
        auto const config = alpaka::KernelCfg<Acc>{
            size,
            Vec::ones(),
            false,
            alpaka::GridBlockExtentSubDivRestrictions::Unrestricted};
        auto const grid = alpaka::getValidWorkDiv(
            config,
            alpaka::getDev(queue),
            kernel,
            in.data(),
            alpaka::getPitchesInBytes(in),
            out.data(),
            alpaka::getPitchesInBytes(out),
            size);

        // Apply the fn function to all elements of the input buffer.
        alpaka::exec<Acc>(
            queue,
            grid,
            kernel,
            in.data(),
            alpaka::getPitchesInBytes(in),
            out.data(),
            alpaka::getPitchesInBytes(out),
            size);
    }

} // namespace alpaka
