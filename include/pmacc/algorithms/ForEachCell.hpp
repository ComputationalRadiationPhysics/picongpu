/* Copyright 2025-2025 , Rene Widera, Edgar Marquardt
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/attribute/FunctionSpecifier.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/lockstep/ForEach.hpp"
#include "pmacc/lockstep/Kernel.hpp"
#include "pmacc/types.hpp"

#include <type_traits>
#include <utility>

#include "pmacc/math/vector/compile-time/Vector.hpp"

namespace pmacc::algorithms
{
    template<typename T_Func>
    struct DeviceLambda
    {
        T_Func const func;

        template<typename... T>
        DEVICEONLY auto operator()(T&&... args) const
        {
            return func(std::forward<T>(args)...);
        }

        template<typename... T>
        DEVICEONLY auto operator()(T&&... args)
        {
            return func(std::forward<T>(args)...);
        }
    };

    template<typename T_Func>
    DeviceLambda(T_Func const) -> DeviceLambda<T_Func>;
} // namespace pmacc::algorithms

namespace alpaka
{
    template<typename T>
    struct IsKernelArgumentTriviallyCopyable<pmacc::algorithms::DeviceLambda<T>, void> : std::true_type
    {
    };
} // namespace alpaka

namespace pmacc
{
    namespace algorithms
    {
        template<typename TMapper>
        struct ForEachCellKernel
        {
            TMapper mapper;

            DINLINE auto operator()(auto const& worker, auto const& func, auto outBox, auto... boxes) const -> void
            {
                constexpr auto simDim = TMapper::Dim;
                using SuperCellSize = typename TMapper::SuperCellSize;
                DataSpace<simDim> const superCellIdx(mapper.getSuperCellIndex(worker.blockDomIdxND()));
                DataSpace<simDim> superCellCellOffset = superCellIdx * SuperCellSize::toRT();

                constexpr uint32_t cellsPerSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value;

                auto forEachCellInSupercell = lockstep::makeForEach<cellsPerSuperCell>(worker);

                forEachCellInSupercell(
                    [&](int32_t const linearCellIdx)
                    {
                        DataSpace<simDim> const cellIdx = pmacc::math::mapToND(SuperCellSize::toRT(), linearCellIdx);
                        DataSpace<simDim> const dataCellOffset = superCellCellOffset + cellIdx;
                        outBox[dataCellOffset] = func(outBox[dataCellOffset], boxes[dataCellOffset]...);
                    });
            }
        };

        template<typename TMapper>
        inline void forEachCell(TMapper mapper, auto const& functor, auto& outBuffer, auto&... buffers)
        {
            PMACC_LOCKSTEP_KERNEL(ForEachCellKernel<TMapper>{mapper})
                .config(mapper.getGridDim(), typename TMapper::SuperCellSize{})(
                    DeviceLambda{functor},
                    outBuffer.getDataBox(),
                    buffers.getDataBox()...);
        }
    } // namespace algorithms
} // namespace pmacc
