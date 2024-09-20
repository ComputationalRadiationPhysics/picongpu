/* Copyright 2024 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/Tag.hpp"
#include "alpaka/acc/Traits.hpp"
#include "alpaka/dim/DimIntegralConst.hpp"
#include "alpaka/idx/Accessors.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/vec/Vec.hpp"

#include <type_traits>

namespace alpaka
{

    /* oncePerGrid
     *
     * `oncePerGrid(acc)` returns true for a single thread within the kernel execution grid.
     *
     * Usually the condition is true for block 0 and thread 0, but these indices should not be relied upon.
     */

    template<typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>
    ALPAKA_FN_ACC inline constexpr bool oncePerGrid(TAcc const& acc)
    {
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;
        using Vec = alpaka::Vec<Dim, Idx>;

        // Workaround for a weird bug in oneAPI 2024.x targetting the CPU backend and FPGA emulator.
        if constexpr(accMatchesTags<TAcc, TagCpuSycl, TagFpgaSyclIntel>)
        {
            // SYCL accelerator specific code
            return acc.m_item_workdiv.get_global_linear_id() == 0;
        }

        return getIdx<Grid, Threads>(acc) == Vec::zeros();
    }

    /* oncePerBlock
     *
     * `oncePerBlock(acc)` returns true for a single thread within the block.
     *
     * Usually the condition is true for thread 0, but this index should not be relied upon.
     */

    template<typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>
    ALPAKA_FN_ACC inline constexpr bool oncePerBlock(TAcc const& acc)
    {
        return getIdx<Block, Threads>(acc) == Vec<Dim<TAcc>, Idx<TAcc>>::zeros();
    }

} // namespace alpaka
