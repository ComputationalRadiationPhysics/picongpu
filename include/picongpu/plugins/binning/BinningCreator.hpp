/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Copyright 2023-2024 Tapish Narwal
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

#if (ENABLE_OPENPMD == 1)

#    include "picongpu/plugins/binning/BinningData.hpp"
#    include "picongpu/plugins/binning/binners/FieldBinner.hpp"
#    include "picongpu/plugins/binning/binners/ParticleBinner.hpp"
#    include "picongpu/plugins/binning/utility.hpp"

#    include <pmacc/math/operation/Add.hpp>

#    include <memory>
#    include <vector>

#    include <openPMD/Series.hpp>

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
             * Creates a particle binner from user input and adds it to the vector of all binners.
             *
             * The Particle binner will bin particle quantities to create histograms on a grid defined by axes.
             * The results will be written to openPMD files.
             *
             * @tparam T_BinaryOp The binary operation to be used for reduction. Optional. Defaults to addition.
             * @param binnerOutputName filename and dataset name for openPMD output. Must be unique to avoid overwrites
             * during data dumps and undefined behaviour during restarts.
             * @param axisTupleObject tuple holding the axes. Each element in the tuple describes an axis for binning.
             * @param speciesTupleObject tuple holding the species to bin. Each element specifies a species to be
             * included in the binning.
             * @param depositionData functor description of the deposited quantity. Defines which particle property to
             * bin.
             * @param extraData tuple holding extra data to be passed to the binner. Can be used for optional
             * configurations. Optional. Defaults to an empty tuple.
             * @return ParticleBinningData& reference to the created ParticleBinningData object.
             *         This can be used to further configure the binning setup if needed.
             */
            template<typename T_BinaryOp = pmacc::math::operation::Add, typename T_Extras = std::tuple<>>
            auto& addParticleBinner(
                std::string const& binnerOutputName,
                auto const& axisTupleObject,
                auto const& speciesTupleObject,
                auto const& depositionData,
                T_Extras const& extraData = {})
            {
                auto bd = makeParticleBinningData<T_BinaryOp>(
                    binnerOutputName,
                    axisTupleObject,
                    speciesTupleObject,
                    depositionData,
                    extraData);
                auto binner = binning::make_unique<ParticleBinner>(bd, cellDescription);
                auto& res = binner->binningData;
                binnerVector.emplace_back(std::move(binner));
                return res;
            }

            /**
             * Creates a field binner from user input and adds it to the vector of all binners
             *
             * The Field binner will bin field quantities to create histograms on a grid defined by axes.
             * The results will be written to openPMD files.
             *
             * @tparam T_BinaryOp The binary operation to be used for reduction. Optional. Defaults to addition.
             * @param binnerOutputName filename and dataset name for openPMD output. Must be unique to avoid overwrites
             * during data dumps and undefined behaviour during restarts.
             * @param axisTupleObject tuple holding the axes. Each element in the tuple describes an axis for binning.
             * @param fieldsTupleObject tuple holding the fields to bin. Each element specifies a field to be
             * included in the binning.
             * @param depositionData functor description of the deposited quantity. Defines which field property to
             * bin.
             * @param extraData tuple holding extra data to be passed to the binner. Can be used for optional
             * configurations. Optional. Defaults to an empty tuple.
             * @return FieldBinningData& reference to the created FieldBinningData object.
             *         This can be used to further configure the binning setup if needed.
             */
            template<typename T_BinaryOp = pmacc::math::operation::Add, typename T_Extras = std::tuple<>>
            auto& addFieldBinner(
                std::string const& binnerOutputName,
                auto const& axisTupleObject,
                auto const& fieldsTupleObject,
                auto const& depositionData,
                T_Extras const& extraData = {})
            {
                auto bd = makeFieldBinningData<T_BinaryOp>(
                    binnerOutputName,
                    axisTupleObject,
                    fieldsTupleObject,
                    depositionData,
                    extraData);
                auto binner = binning::make_unique<FieldBinner>(bd, cellDescription);
                auto& res = binner->binningData;
                binnerVector.emplace_back(std::move(binner));
                return res;
            }
        };
    } // namespace plugins::binning
} // namespace picongpu

#endif
