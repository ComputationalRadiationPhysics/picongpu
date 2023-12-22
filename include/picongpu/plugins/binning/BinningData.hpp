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
#include "picongpu/plugins/common/openPMDDefaultExtension.hpp"

#include <alpaka/core/Tuple.hpp>

#include <cstdint>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

namespace picongpu
{
    namespace plugins::binning
    {
        template<typename T_AxisTuple, typename T_SpeciesTuple, typename T_DepositionData>
        struct BinningData
        {
        public:
            using DepositionFunctorType = typename T_DepositionData::FunctorType;
            using DepositedQuantityType = typename T_DepositionData::QuantityType;
            // @todo infer type from functor
            // using DepositedQuantityType = std::invoke_result_t<TDepositedQuantityFunctor, particle, worker>;

            std::string binnerOutputName;
            T_AxisTuple axisTuple;
            T_SpeciesTuple speciesTuple;
            T_DepositionData depositionData;
            std::function<void(::openPMD::Series& series, ::openPMD::Iteration& iteration, ::openPMD::Mesh& mesh)>
                writeOpenPMDFunctor;
            DataSpace<std::tuple_size_v<T_AxisTuple>> axisExtentsND;

            /* Optional parameters not initialized by constructor.
             * Use the return value of addBinner() to modify them if needed. */
            bool timeAveraging = true;
            bool normalizeByBinVolume = true;
            std::string notifyPeriod = "1";
            uint32_t dumpPeriod = 0u;

            std::string openPMDInfix = "_%06T.";
            std::string openPMDExtension = openPMD::getDefaultExtension();

            std::string jsonCfg = "{}";

            BinningData(
                const std::string binnerName,
                const T_AxisTuple axes,
                const T_SpeciesTuple species,
                const T_DepositionData depositData,
                std::function<void(::openPMD::Series& series, ::openPMD::Iteration& iteration, ::openPMD::Mesh& mesh)>
                    writeOpenPMD)
                : binnerOutputName{binnerName}
                , axisTuple{axes}
                , speciesTuple{species}
                , depositionData{depositData}
                , writeOpenPMDFunctor{writeOpenPMD}
            {
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
            }

            /** @brief Time average the accumulated data when doing the dump. Defaults to true. */
            BinningData& setTimeAveraging(bool timeAv)
            {
                this->timeAveraging = timeAv;
                return *this;
            }
            /** @brief Defaults to true */
            BinningData& setNormalizeByBinVolume(bool normalize)
            {
                this->normalizeByBinVolume = normalize;
                return *this;
            }
            /** @brief The periodicity of the output. Defaults to 1 */
            BinningData& setNotifyPeriod(std::string notify)
            {
                this->notifyPeriod = std::move(notify);
                return *this;
            }
            /** @brief The number of notify steps to accumulate over. Dump at the end. Defaults to 1. */
            BinningData& setDumpPeriod(uint32_t dumpXNotifys)
            {
                this->dumpPeriod = dumpXNotifys;
                return *this;
            }
            BinningData& setOpenPMDExtension(std::string extension)
            {
                this->openPMDExtension = std::move(extension);
                return *this;
            }
            BinningData& setOpenPMDInfix(std::string infix)
            {
                this->openPMDInfix = std::move(infix);
                return *this;
            }

            BinningData& setJsonCfg(std::string cfg)
            {
                this->jsonCfg = std::move(cfg);
                return *this;
            }
        };
    }; // namespace plugins::binning
} // namespace picongpu
