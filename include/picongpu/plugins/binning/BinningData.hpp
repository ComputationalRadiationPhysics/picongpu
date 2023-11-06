/* Copyright 2023 Tapish Narwal
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/plugins/binning/Axis.hpp"

#include <alpaka/core/Tuple.hpp>

namespace picongpu
{
    namespace plugins::binning
    {
        template<typename T_AxisTuple, typename T_SpeciesTuple, typename T_DepositionData>
        struct BinningData
        {
            // private:
            //     uint32_t n_axes;

        public:
            using DepositionFunctorType = typename T_DepositionData::FunctorType;
            using DepositedQuantityType = typename T_DepositionData::QuantityType;
            // @todo infer type from functor
            // using DepositedQuantityType = std::invoke_result_t<TDepositedQuantityFunctor, particle, worker>;
            // using DepositedQuantityType = typename decltype(std::function{quantityFunctor})::result_type;

            std::string binnerOutputName;
            T_AxisTuple axisTuple;
            T_SpeciesTuple speciesTuple;
            T_DepositionData depositionData;
            std::string notifyPeriod;
            uint32_t dumpPeriod;
            bool timeAveraging;
            bool normalizeByBinVolume;
            std::function<void(::openPMD::Series& series, ::openPMD::Iteration& iteration, ::openPMD::Mesh& mesh)>
                writeOpenPMDFunctor;
            DataSpace<std::tuple_size_v<T_AxisTuple>> axisExtentsND;

            BinningData(
                const std::string binnerName,
                const T_AxisTuple axes,
                const T_SpeciesTuple species,
                const T_DepositionData depositData,
                const std::string period,
                uint32_t dumpNNotifies,
                bool enableTimeAveraging,
                bool normalizeBinVol,
                std::function<void(::openPMD::Series& series, ::openPMD::Iteration& iteration, ::openPMD::Mesh& mesh)>
                    writeOpenPMD)
                : binnerOutputName{binnerName}
                , axisTuple{axes}
                , speciesTuple{species}
                , depositionData{depositData}
                , notifyPeriod{period}
                , dumpPeriod{dumpNNotifies}
                , timeAveraging{enableTimeAveraging}
                , normalizeByBinVolume{normalizeBinVol}
                , writeOpenPMDFunctor{writeOpenPMD}
            {
                static_assert(getNAxes() <= 3, "Only upto 3 binning axes are supported for now");
                std::apply(
                    [&](auto const&... tupleArgs)
                    {
                        uint32_t i = 0;
                        // This assumes getNBins() exists
                        ((axisExtentsND[i++] = tupleArgs.getNBins()), ...);
                    },
                    axisTuple);
            }

            static constexpr uint32_t getNAxes()
            {
                return std::tuple_size_v<T_AxisTuple>;
                // return utility::tuple::size(axisTuple);
                // return utility::tuple::size(declval<T_AxisTuple>());
            }
        };
    }; // namespace plugins::binning
} // namespace picongpu
