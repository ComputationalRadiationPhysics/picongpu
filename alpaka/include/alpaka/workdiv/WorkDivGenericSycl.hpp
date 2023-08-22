/* Copyright 2023 Jan Stephan, Luca Ferragina, Andrea Bocci, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/idx/Traits.hpp"
#include "alpaka/vec/Vec.hpp"
#include "alpaka/workdiv/Traits.hpp"

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace alpaka
{
    //! The SYCL accelerator work division.
    template<typename TDim, typename TIdx>
    class WorkDivGenericSycl : public concepts::Implements<ConceptWorkDiv, WorkDivGenericSycl<TDim, TIdx>>
    {
        static_assert(TDim::value > 0, "The SYCL work division must have a dimension greater than zero.");

    public:
        using WorkDivBase = WorkDivGenericSycl;

        WorkDivGenericSycl(Vec<TDim, TIdx> const& threadElemExtent, sycl::nd_item<TDim::value> work_item)
            : m_threadElemExtent{threadElemExtent}
            , m_item_workdiv{work_item}
        {
        }

        Vec<TDim, TIdx> const& m_threadElemExtent;
        sycl::nd_item<TDim::value> m_item_workdiv;
    };
} // namespace alpaka

namespace alpaka::trait
{
    //! The SYCL accelerator work division dimension get trait specialization.
    template<typename TDim, typename TIdx>
    struct DimType<WorkDivGenericSycl<TDim, TIdx>>
    {
        using type = TDim;
    };

    //! The SYCL accelerator work division idx type trait specialization.
    template<typename TDim, typename TIdx>
    struct IdxType<WorkDivGenericSycl<TDim, TIdx>>
    {
        using type = TIdx;
    };

    //! The SYCL accelerator work division grid block extent trait specialization.
    template<typename TDim, typename TIdx>
    struct GetWorkDiv<WorkDivGenericSycl<TDim, TIdx>, origin::Grid, unit::Blocks>
    {
        //! \return The number of blocks in each dimension of the grid.
        static auto getWorkDiv(WorkDivGenericSycl<TDim, TIdx> const& workDiv) -> Vec<TDim, TIdx>
        {
            if constexpr(TDim::value == 0)
                return Vec<TDim, TIdx>{};
            else if constexpr(TDim::value == 1)
                return Vec<TDim, TIdx>{static_cast<TIdx>(workDiv.m_item_workdiv.get_group_range(0))};
            else if constexpr(TDim::value == 2)
            {
                return Vec<TDim, TIdx>{
                    static_cast<TIdx>(workDiv.m_item_workdiv.get_group_range(1)),
                    static_cast<TIdx>(workDiv.m_item_workdiv.get_group_range(0))};
            }
            else
            {
                return Vec<TDim, TIdx>{
                    static_cast<TIdx>(workDiv.m_item_workdiv.get_group_range(2)),
                    static_cast<TIdx>(workDiv.m_item_workdiv.get_group_range(1)),
                    static_cast<TIdx>(workDiv.m_item_workdiv.get_group_range(0))};
            }
        }
    };

    //! The SYCL accelerator work division block thread extent trait specialization.
    template<typename TDim, typename TIdx>
    struct GetWorkDiv<WorkDivGenericSycl<TDim, TIdx>, origin::Block, unit::Threads>
    {
        //! \return The number of threads in each dimension of a block.
        static auto getWorkDiv(WorkDivGenericSycl<TDim, TIdx> const& workDiv) -> Vec<TDim, TIdx>
        {
            if constexpr(TDim::value == 0)
                return Vec<TDim, TIdx>{};
            else if constexpr(TDim::value == 1)
                return Vec<TDim, TIdx>{static_cast<TIdx>(workDiv.m_item_workdiv.get_local_range(0))};
            else if constexpr(TDim::value == 2)
            {
                return Vec<TDim, TIdx>{
                    static_cast<TIdx>(workDiv.m_item_workdiv.get_local_range(1)),
                    static_cast<TIdx>(workDiv.m_item_workdiv.get_local_range(0))};
            }
            else
            {
                return Vec<TDim, TIdx>{
                    static_cast<TIdx>(workDiv.m_item_workdiv.get_local_range(2)),
                    static_cast<TIdx>(workDiv.m_item_workdiv.get_local_range(1)),
                    static_cast<TIdx>(workDiv.m_item_workdiv.get_local_range(0))};
            }
        }
    };

    //! The SYCL accelerator work division thread element extent trait specialization.
    template<typename TDim, typename TIdx>
    struct GetWorkDiv<WorkDivGenericSycl<TDim, TIdx>, origin::Thread, unit::Elems>
    {
        //! \return The number of elements in each dimension of the thread.
        static auto getWorkDiv(WorkDivGenericSycl<TDim, TIdx> const& workDiv) -> Vec<TDim, TIdx>
        {
            return workDiv.m_threadElemExtent;
        }
    };
} // namespace alpaka::trait

#endif
