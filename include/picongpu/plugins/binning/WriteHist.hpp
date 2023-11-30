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

#if !ENABLE_OPENPMD
#    error The activated Binning plugin requires openPMD-api.
#endif

#pragma once

#include "picongpu/plugins/binning/UnitConversion.hpp"
#include "picongpu/plugins/binning/utility.hpp"
#include "picongpu/plugins/common/openPMDVersion.def"
#include "picongpu/plugins/common/stringHelpers.hpp"

#include <memory>
#include <optional>

#include <openPMD/openPMD.hpp>

namespace picongpu
{
    namespace plugins::binning
    {
        struct WriteOpenPMDParams
        {
            std::string const& extension;
            std::string const& jsonConfig;
        };

        /**
         * Write the N-D histogram with units
         * Write axis bin edges with units
         */
        class WriteHist
        {
        public:
            template<typename T_Type, typename T_BinningData>
            void operator()(
                std::optional<::openPMD::Series>& maybe_series,
                std::unique_ptr<HostBuffer<T_Type, 1u>> hReducedBuffer,
                T_BinningData binningData,
                std::string const& dir,
                std::array<double, 7> outputUnits,
                const uint32_t currentStep)
            {
                using Type = T_Type;

                if(!maybe_series.has_value())
                {
                    auto const& extension = binningData.openPMDExtension;
                    std::ostringstream filename;
                    if(std::any_of(extension.begin(), extension.end(), [](char const c) { return c == '.'; }))
                    {
                        filename << binningData.binnerOutputName << binningData.openPMDInfix << extension;
                    }
                    else
                    {
                        filename << binningData.binnerOutputName << binningData.openPMDInfix << '.' << extension;
                    }

                    maybe_series = ::openPMD::Series(dir + '/' + filename.str(), ::openPMD::Access::CREATE);
                }

                auto& series = *maybe_series;

                /* begin recommended openPMD global attributes */
                // series.setMeshesPath(meshesPathName);
                const std::string software("PIConGPU");
                std::stringstream softwareVersion;
                softwareVersion << PICONGPU_VERSION_MAJOR << "." << PICONGPU_VERSION_MINOR << "."
                                << PICONGPU_VERSION_PATCH;
                if(!std::string(PICONGPU_VERSION_LABEL).empty())
                    softwareVersion << "-" << PICONGPU_VERSION_LABEL;
                series.setSoftware(software, softwareVersion.str());

                std::string author = Environment<>::get().SimulationDescription().getAuthor();
                if(author.length() > 0)
                    series.setAuthor(author);

                std::string date = helper::getDateString("%F %T %z");
                series.setDate(date);
                /* end recommended openPMD global attributes */

                ::openPMD::Iteration iteration = series.writeIterations()[currentStep];

                /* begin required openPMD global attributes */
                iteration.setDt<float_X>(DELTA_T);
                const float_X time = float_X(currentStep) * DELTA_T;
                iteration.setTime(time);
                iteration.setTimeUnitSI(UNIT_TIME);
                /* end required openPMD global attributes */

                /**
                 * Write the histogram
                 */
                ::openPMD::Mesh mesh = iteration.meshes["Binning"];

                // Call the user defined OpenPMD
                binningData.writeOpenPMDFunctor(series, iteration, mesh);

                mesh.setGeometry(::openPMD::Mesh::Geometry::cartesian);
                mesh.setDataOrder(::openPMD::Mesh::DataOrder::C);

                std::apply(
                    [&](auto&... tupleArgs)
                    {
                        ((mesh.setAttribute(tupleArgs.label + "_bin_edges", tupleArgs.getBinEdgesSI())), ...);
                        ((mesh.setAttribute(tupleArgs.label + "_units", tupleArgs.units)), ...);
                        std::vector<std::string> labelVector;
                        ((labelVector.emplace_back(tupleArgs.label)), ...);
                        std::reverse(labelVector.begin(), labelVector.end());
                        mesh.setAxisLabels(labelVector);
                    },
                    binningData.axisTuple); // careful no const tupleArgs

                std::vector<double> gridSpacingVector;
                std::vector<double> gridOffsetVector;

                for(int i = binningData.getNAxes() - 1; i >= 0; i--)
                {
                    gridSpacingVector.emplace_back(1.); // How to deal with non fixed grid spacings?
                    gridOffsetVector.emplace_back(0.);
                }
                mesh.setGridSpacing(gridSpacingVector);
                mesh.setGridGlobalOffset(gridOffsetVector);

                {
                    using UD = ::openPMD::UnitDimension;
                    mesh.setUnitDimension(makeOpenPMDUnitMap(binningData.depositionData.units)); // charge density
                }

                ::openPMD::MeshRecordComponent record = mesh[::openPMD::RecordComponent::SCALAR];

                /*
                 * The value represents an aggregation over one cell, so any value is correct for the mesh position.
                 * Just use the center.
                 */
                record.setPosition(std::vector<float>{0.5, 0.5});

                ::openPMD::Offset histOffset;
                ::openPMD::Extent histExtent;
                auto bufferExtents = binningData.axisExtentsND;

                /** Z Y X - reverse order */
                for(int i = binningData.getNAxes() - 1; i >= 0; i--)
                {
                    histExtent.emplace_back(static_cast<size_t>(bufferExtents[i]));
                    histOffset.emplace_back(static_cast<size_t>(0));
                }

                record.setUnitSI(get_conversion_factor(outputUnits));

                record.resetDataset({::openPMD::determineDatatype<Type>(), histExtent});
#if OPENPMDAPI_VERSION_GE(0, 15, 0)
                auto base_ptr = hReducedBuffer->getBasePointer();
                ::openPMD::UniquePtrWithLambda<Type> data(
                    base_ptr,
                    [hReducedBuffer
                     = std::make_shared<decltype(hReducedBuffer)>(std::move(hReducedBuffer))](auto const*)
                    {
                        /* no-op, destroy data via destructor of captured hReducedBuffer */
                    });
                record.storeChunk<Type>(std::move(data), histOffset, histExtent);
#else
                openPMD::storeChunkRaw(record, hReducedBuffer->getBasePointer(), histOffset, histExtent);
#endif
                iteration.close();
            };
        };
    } // namespace plugins::binning
} // namespace picongpu
