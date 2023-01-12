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

#    include <alpaka/acc/Traits.hpp>
#    include <alpaka/core/BoostPredef.hpp>
#    include <alpaka/core/STLTuple/STLTuple.hpp>
#    include <alpaka/core/Sycl.hpp>
#    include <alpaka/core/Tuple.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/dim/Traits.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/kernel/Traits.hpp>
#    include <alpaka/mem/buf/sycl/Accessor.hpp>
#    include <alpaka/pltf/Traits.hpp>
#    include <alpaka/queue/Traits.hpp>
#    include <alpaka/workdiv/WorkDivMembers.hpp>

#    include <CL/sycl.hpp>

#    include <cassert>
#    include <functional>
#    include <memory>
#    include <stdexcept>
#    include <tuple>
#    include <type_traits>
#    include <utility>

namespace alpaka::experimental::detail
{
    template<typename TAcc, typename TKernelFnObj, typename... TArgs>
    struct kernel
    {
    }; // SYCL kernel names must be globally visible

    // Helpers for assigning placeholder accessors to the command group of our kernel
    struct general
    {
    };
    struct special : general
    {
    };

    template<typename TElem, typename TIdx, std::size_t TDim, typename TAccessModes>
    inline auto require(
        sycl::handler& cgh,
        Accessor<detail::SyclAccessor<TElem, DimInt<TDim>::value, TAccessModes>, TElem, TIdx, TDim, TAccessModes> acc,
        special)
    {
        cgh.require(acc.m_accessor);
    }

    template<typename TParam>
    inline auto require(sycl::handler&, TParam&&, general)
    {
    }

    template<typename... TArgs>
    inline auto require(sycl::handler& cgh, core::Tuple<TArgs...> const& args)
    {
        core::apply([&](auto&&... ps) { (require(cgh, std::forward<decltype(ps)>(ps), special{}), ...); }, args);
    }
} // namespace alpaka::experimental::detail

namespace alpaka::experimental
{
    //! The SYCL accelerator execution task.
    template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelGenericSycl final : public WorkDivMembers<TDim, TIdx>
    {
    public:
        static_assert(TDim::value > 0 && TDim::value <= 3, "Invalid kernel dimensionality");

        template<typename TWorkDiv>
        TaskKernelGenericSycl(TWorkDiv&& workDiv, TKernelFnObj const& kernelFnObj, TArgs&&... args)
            : WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv))
            , m_kernelFnObj{kernelFnObj}
            , m_args{std::forward<TArgs>(args)...}
        {
        }

        auto operator()(sycl::handler& cgh, sycl::buffer<int, 1>& global_fence_buf) const -> void
        {
            // Assign placeholder accessors to this command group
            detail::require(cgh, m_args);

            auto const work_groups = WorkDivMembers<TDim, TIdx>::m_gridBlockExtent;
            auto const group_items = WorkDivMembers<TDim, TIdx>::m_blockThreadExtent;
            auto const item_elements = WorkDivMembers<TDim, TIdx>::m_threadElemExtent;

            auto const global_size = get_global_size(work_groups, group_items);
            auto const local_size = get_local_size(group_items);

            // allocate dynamic shared memory -- needs at least 1 byte to make the Xilinx Runtime happy
            auto const dyn_shared_mem_bytes = std::max(
                1ul,
                core::apply(
                    [&](std::decay_t<TArgs> const&... args) {
                        return getBlockSharedMemDynSizeBytes<TAcc>(m_kernelFnObj, group_items, item_elements, args...);
                    },
                    m_args));

            auto dyn_shared_accessor
                = sycl::accessor<std::byte, 1, sycl::access_mode::read_write, sycl::target::local>{
                    sycl::range<1>{dyn_shared_mem_bytes},
                    cgh};

            // allocate static shared memory -- value comes from the build system
            constexpr auto st_shared_mem_bytes = std::size_t{ALPAKA_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB * 1024};
            auto st_shared_accessor = sycl::accessor<std::byte, 1, sycl::access_mode::read_write, sycl::target::local>{
                sycl::range<1>{st_shared_mem_bytes},
                cgh};

            // register memory fence dummies
            auto global_fence_dummy = global_fence_buf.get_access(cgh); // Exists once per queue
            auto local_fence_dummy
                = sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::local>{sycl::range<1>{1}, cgh};

            // copy-by-value so we don't access 'this' on the device
            auto k_func = m_kernelFnObj;
            auto k_args = m_args;

#    ifdef ALPAKA_SYCL_IOSTREAM_ENABLED
            // Set up device-side printing with (user-chosen value) KiB per block for the output buffer.
            constexpr auto buf_size = std::size_t{ALPAKA_SYCL_IOSTREAM_KIB * 1024};
            auto buf_per_work_item = std::size_t{};
            if constexpr(TDim::value == 1)
                buf_per_work_item = buf_size / static_cast<std::size_t>(group_items[0]);
            else if constexpr(TDim::value == 2)
                buf_per_work_item = buf_size / static_cast<std::size_t>(group_items[0] * group_items[1]);
            else
                buf_per_work_item
                    = buf_size / static_cast<std::size_t>(group_items[0] * group_items[1] * group_items[2]);

            assert(buf_per_work_item > 0);

            auto output_stream = sycl::stream{buf_size, buf_per_work_item, cgh};
#    endif
            cgh.parallel_for<detail::kernel<TAcc, TKernelFnObj, TArgs...>>(
                sycl::nd_range<TDim::value>{global_size, local_size},
                [=](sycl::nd_item<TDim::value> work_item)
                {
#    ifdef ALPAKA_SYCL_IOSTREAM_ENABLED
                    auto acc = TAcc{
                        item_elements,
                        work_item,
                        dyn_shared_accessor,
                        st_shared_accessor,
                        global_fence_dummy,
                        local_fence_dummy,
                        output_stream};
#    else
                    auto acc = TAcc{
                        item_elements,
                        work_item,
                        dyn_shared_accessor,
                        st_shared_accessor,
                        global_fence_dummy,
                        local_fence_dummy};
#    endif
                    core::apply(
                        [k_func, &acc](typename std::decay_t<TArgs> const&... args) { k_func(acc, args...); },
                        k_args);
                });
        }

        static constexpr auto is_sycl_task = true;
        // Distinguish from other tasks
        static constexpr auto is_sycl_kernel = true;

    private:
        auto get_global_size(Vec<TDim, TIdx> const& work_groups, Vec<TDim, TIdx> const& group_items) const
        {
            if constexpr(TDim::value == 1)
                return sycl::range<1>{static_cast<std::size_t>(work_groups[0] * group_items[0])};
            else if constexpr(TDim::value == 2)
                return sycl::range<2>{
                    static_cast<std::size_t>(work_groups[1] * group_items[1]),
                    static_cast<std::size_t>(work_groups[0] * group_items[0])};
            else
                return sycl::range<3>{
                    static_cast<std::size_t>(work_groups[2] * group_items[2]),
                    static_cast<std::size_t>(work_groups[1] * group_items[1]),
                    static_cast<std::size_t>(work_groups[0] * group_items[0])};
        }

        auto get_local_size(Vec<TDim, TIdx> const& group_items) const
        {
            if constexpr(TDim::value == 1)
                return sycl::range<1>{static_cast<std::size_t>(group_items[0])};
            else if constexpr(TDim::value == 2)
                return sycl::range<2>{
                    static_cast<std::size_t>(group_items[1]),
                    static_cast<std::size_t>(group_items[0])};
            else
                return sycl::range<3>{
                    static_cast<std::size_t>(group_items[2]),
                    static_cast<std::size_t>(group_items[1]),
                    static_cast<std::size_t>(group_items[0])};
        }

    public:
        TKernelFnObj m_kernelFnObj;
        core::Tuple<std::decay_t<TArgs>...> m_args;
    };
} // namespace alpaka::experimental

namespace alpaka::trait
{
    //! The SYCL execution task accelerator type trait specialization.
    template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    struct AccType<experimental::TaskKernelGenericSycl<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
    {
        using type = TAcc;
    };

    //! The SYCL execution task device type trait specialization.
    template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    struct DevType<experimental::TaskKernelGenericSycl<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
    {
        using type = typename DevType<TAcc>::type;
    };

    //! The SYCL execution task platform type trait specialization.
    template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    struct PltfType<experimental::TaskKernelGenericSycl<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
    {
        using type = typename PltfType<TAcc>::type;
    };

    //! The SYCL execution task dimension getter trait specialization.
    template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    struct DimType<experimental::TaskKernelGenericSycl<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
    {
        using type = TDim;
    };

    //! The SYCL execution task idx type trait specialization.
    template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    struct IdxType<experimental::TaskKernelGenericSycl<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
    {
        using type = TIdx;
    };
} // namespace alpaka::trait

#endif
