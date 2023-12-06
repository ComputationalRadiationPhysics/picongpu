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

#include "picongpu/plugins/binning/Binner.hpp"
#include "picongpu/plugins/binning/BinningData.hpp"

#include <memory>

namespace picongpu
{
    namespace plugins::binning
    {
        /**
         * An object of this class is provided to the user to add their binning setups
         */
        class BinningCreator
        {
        public:
            MappingDesc* cellDescription;
            std::vector<std::unique_ptr<IPlugin>>& binnerVector;

        public:
            BinningCreator(std::vector<std::unique_ptr<IPlugin>>& binVec, MappingDesc* cellDesc)
                : cellDescription{cellDesc}
                , binnerVector{binVec}
            {
            }

            /**
             * Creates a binner from user input and adds it to the vector of all binners
             * @param binnerOutputName filename for openPMD output. It must be unique or will cause overwrites during
             * data dumps
             * @param axisTupleObject tuple holding the axes
             * @param speciesTupleObject tuple holding the species to do the binning with
             * @param depositionData functorDescription of the deposited quantity
             * @param notifyPeriod The periodicity of the output
             * @param dumpPeriod The number of notify steps to accumulate over. Dump at the end. Defaults to 1.
             * @param timeAveraging Time average the accumulated data when doing the dump. Defaults to true.
             * @param normalizeByBinVolume defaults to true
             * @param writeOpenPMDFunctor Functor to write out user specified openPMD data
             */
            template<typename TAxisTuple, typename TSpeciesTuple, typename TDepositionData>
            auto addBinner(
                std::string binnerOutputName,
                TAxisTuple axisTupleObject,
                TSpeciesTuple speciesTupleObject,
                TDepositionData depositionData,
                std::function<void(::openPMD::Series& series, ::openPMD::Iteration& iteration, ::openPMD::Mesh& mesh)>
                    writeOpenPMDFunctor
                = [](::openPMD::Series& series, ::openPMD::Iteration& iteration, ::openPMD::Mesh& mesh) {})
                -> BinningData<TAxisTuple, TSpeciesTuple, TDepositionData>&
            {
                auto bd = BinningData<TAxisTuple, TSpeciesTuple, TDepositionData>(
                    binnerOutputName,
                    axisTupleObject,
                    speciesTupleObject,
                    depositionData,
                    writeOpenPMDFunctor);
                auto binner = std::make_unique<Binner<BinningData<TAxisTuple, TSpeciesTuple, TDepositionData>>>(
                    bd,
                    cellDescription);
                auto& res = binner->binningData;
                binnerVector.emplace_back(std::move(binner));
                return res;
            }
        };
    } // namespace plugins::binning
} // namespace picongpu
