/* Copyright 2025 Tapish Narwal
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/plugins/binning/utility.hpp"

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/lockstep/ForEach.hpp>
#include <pmacc/math/operation.hpp>
#include <pmacc/meta/errorHandlerPolicies/ReturnType.hpp>

#include <cstdint>

namespace picongpu
{
    namespace plugins::binning
    {
        /**
         * Functor to run on divide for time averaging
         */
        // @todo make this generic? apply a functor on all elements of a databox? take in functor?
        template<uint32_t blockSize, uint32_t nAxes>
        struct ProductKernel
        {
            using ResultType = void;

            HDINLINE ProductKernel() = default;

            template<typename T_Worker, typename T_DataSpace, typename T_DepositedQuantity, typename T_DataBox>
            HDINLINE void operator()(
                const T_Worker& worker,
                const T_DataSpace& extentsDataspace,
                const T_DepositedQuantity factor,
                T_DataBox dataBox) const
            {
                auto blockIdx = worker.blockDomIdxND().x() * blockSize;
                auto forEachElemInDataboxChunk = pmacc::lockstep::makeForEach<blockSize>(worker);
                forEachElemInDataboxChunk(
                    [&](int32_t const linearIdx)
                    {
                        int32_t const linearTid = blockIdx + linearIdx;
                        if(linearTid < extentsDataspace.productOfComponents())
                            dataBox(pmacc::DataSpace<1u>{static_cast<int>(linearTid)}) *= factor;
                    });
            }
        };


        /**
         * Functor to do volume normaliztion
         * Factor with picongpu units
         * User needs to deal with the units seperately
         */
        // @todo make this more generic? databox operations with another databox?
        // maybe store the normalization in a buffer rather than computing it at every output timestep (memory vs ops)
        // this style vs having the members as params of operator()
        template<uint32_t blockSize, uint32_t nAxes>
        struct BinNormalizationKernel
        {
            using ResultType = void;

            HDINLINE BinNormalizationKernel() = default;

            // @todo check if type stored in histBox is same as axisKernelTuple Type
            template<typename T_Worker, typename T_DataSpace, typename T_BinWidthKernelTuple, typename T_DataBox>
            HDINLINE void operator()(
                const T_Worker& worker,
                const T_DataSpace& extentsDataspace,
                const T_BinWidthKernelTuple& binWidthsKernelTuple,
                T_DataBox histBox) const
            {
                // @todo check normDataBox shape is same as histBox
                auto blockIdx = worker.blockDomIdxND().x() * blockSize;
                auto forEachElemInDataboxChunk = pmacc::lockstep::makeForEach<blockSize>(worker);
                forEachElemInDataboxChunk(
                    [&](int32_t const linearIdx)
                    {
                        int32_t const linearTid = blockIdx + linearIdx;
                        if(linearTid < extentsDataspace.productOfComponents())
                        {
                            pmacc::DataSpace<nAxes> const nDIdx = pmacc::math::mapToND(extentsDataspace, linearTid);
                            float_X factor = 1.;
                            binning::apply(
                                [&](auto const&... binWidthsKernel)
                                {
                                    // uses bin width for axes without dimensions as well should those be ignored?
                                    uint32_t i = 0;
                                    ((factor *= binWidthsKernel.getBinWidth(nDIdx[i++])), ...);
                                },
                                binWidthsKernelTuple);

                            histBox(linearTid) *= 1 / factor;
                        }
                    });
            }
        };

    } // namespace plugins::binning
} // namespace picongpu
