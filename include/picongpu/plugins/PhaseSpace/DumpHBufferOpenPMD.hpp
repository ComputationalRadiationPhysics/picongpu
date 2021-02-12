/* Copyright 2013-2021 Axel Huebl, Rene Widera
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/plugins/PhaseSpace/AxisDescription.hpp"
#include <pmacc/communication/manager_common.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/mappings/simulation/SubGrid.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/cuSTL/container/HostBuffer.hpp>
#include <pmacc/math/vector/Int.hpp>
#include <pmacc/verify.hpp>

#include <string>
#include <fstream>
#include <sstream>
#include <utility>
#include <mpi.h>
#include <openPMD/openPMD.hpp>
#include <vector>

namespace picongpu
{
    class DumpHBuffer
    {
    private:
        using SuperCellSize = typename MappingDesc::SuperCellSize;

    public:
        /** Dump the PhaseSpace host Buffer
         *
         * \tparam Type the HBuffers element type
         * \tparam int the HBuffers dimension
         * \param hBuffer const reference to the hBuffer, including guard cells in spatial dimension
         * \param axis_element plot to create: e.g. py, x from momentum/spatial-coordinate
         * \param unit sim unit of the buffer
         * \param strSpecies unique short hand name of the species
         * \param filenameSuffix infix + extension part of openPMD filename
         * \param currentStep current time step
         * \param mpiComm communicator of the participating ranks
         */
        template<typename T_Type, int T_bufDim>
        void operator()(
            const pmacc::container::HostBuffer<T_Type, T_bufDim>& hBuffer,
            const AxisDescription axis_element,
            const std::pair<float_X, float_X> axis_p_range,
            const float_64 pRange_unit,
            const float_64 unit,
            const std::string strSpecies,
            const std::string filenameExtension,
            const std::string jsonConfig,
            const uint32_t currentStep,
            MPI_Comm mpiComm) const
        {
            using Type = T_Type;

            /** file name *****************************************************
             *    phaseSpace/PhaseSpace_xpy_timestep.h5                       */
            std::string fCoords("xyz");
            std::ostringstream openPMDFilename;
            openPMDFilename << "phaseSpace/PhaseSpace_" << strSpecies << "_" << fCoords.at(axis_element.space) << "p"
                            << fCoords.at(axis_element.momentum) << "_%T." << filenameExtension;

            /** get size of the fileWriter communicator ***********************/
            int size;
            MPI_CHECK(MPI_Comm_size(mpiComm, &size));

            /** create parallel domain collector ******************************/
            ::openPMD::Series series(openPMDFilename.str(), ::openPMD::Access::CREATE, mpiComm, jsonConfig);
            ::openPMD::Iteration iteration = series.iterations[currentStep];

            const std::string software("PIConGPU");

            std::stringstream softwareVersion;
            softwareVersion << PICONGPU_VERSION_MAJOR << "." << PICONGPU_VERSION_MINOR << "."
                            << PICONGPU_VERSION_PATCH;
            if(!std::string(PICONGPU_VERSION_LABEL).empty())
                softwareVersion << "-" << PICONGPU_VERSION_LABEL;
            series.setSoftware(software, softwareVersion.str());

            pmacc::GridController<simDim>& gc = pmacc::Environment<simDim>::get().GridController();

            /** calculate GUARD offset in the source hBuffer *****************/
            const uint32_t rGuardCells
                = SuperCellSize().toRT()[axis_element.space] * GuardSize::toRT()[axis_element.space];

            /** calculate local and global size of the phase space ***********/
            const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);
            const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
            const std::uint64_t rLocalOffset = subGrid.getLocalDomain().offset[axis_element.space];
            const std::uint64_t rLocalSize = int(hBuffer.size().y() - 2 * rGuardCells);
            const std::uint64_t rGlobalSize = subGrid.getGlobalDomain().size[axis_element.space];
            PMACC_VERIFY(int(rLocalSize) == subGrid.getLocalDomain().size[axis_element.space]);

            /* globalDomain of the phase space */
            ::openPMD::Extent globalPhaseSpace_extent{rGlobalSize, hBuffer.size().x()};

            /* global moving window meta information */
            ::openPMD::Offset globalPhaseSpace_offset{0, 0};
            std::uint64_t globalMovingWindowOffset = 0;
            std::uint64_t globalMovingWindowSize = rGlobalSize;
            if(axis_element.space == AxisDescription::y) /* spatial axis == y */
            {
                globalPhaseSpace_offset[0] = numSlides * rLocalSize;
                Window window = MovingWindow::getInstance().getWindow(currentStep);
                globalMovingWindowOffset = window.globalDimensions.offset[axis_element.space];
                globalMovingWindowSize = window.globalDimensions.size[axis_element.space];
            }

            /* localDomain: offset of it in the globalDomain and size */
            ::openPMD::Offset localPhaseSpace_offset{rLocalOffset, 0};
            ::openPMD::Extent localPhaseSpace_extent{rLocalSize, hBuffer.size().x()};

            /** Dataset Name **************************************************/
            std::ostringstream dataSetName;
            /* xpx or ypz or ... */
            dataSetName << strSpecies << "_" << fCoords.at(axis_element.space) << "p"
                        << fCoords.at(axis_element.momentum);

            /** debug log *****************************************************/
            int rank;
            MPI_CHECK(MPI_Comm_rank(mpiComm, &rank));
            {
                std::stringstream offsetAsString, localExtentAsString, globalExtentAsString;
                offsetAsString << "[" << localPhaseSpace_offset[0] << ", " << localPhaseSpace_offset[1] << "]";
                localExtentAsString << "[" << localPhaseSpace_extent[0] << ", " << localPhaseSpace_extent[1] << "]";
                globalExtentAsString << "[" << globalPhaseSpace_extent[0] << ", " << globalPhaseSpace_extent[1] << "]";
                log<picLog::INPUT_OUTPUT>(
                    "Dump buffer %1% to %2% at offset %3% with size %4% for total size %5% for rank %6% / %7%")
                    % (*(hBuffer.origin()(0, rGuardCells))) % dataSetName.str() % offsetAsString.str()
                    % localExtentAsString.str() % globalExtentAsString.str() % rank % size;
            }

            /** write local domain ********************************************/

            ::openPMD::Mesh mesh = iteration.meshes[dataSetName.str()];
            ::openPMD::MeshRecordComponent dataset = mesh[::openPMD::RecordComponent::SCALAR];

            dataset.resetDataset({::openPMD::determineDatatype<Type>(), globalPhaseSpace_extent});
            std::shared_ptr<Type> data(&(*hBuffer.origin()(0, rGuardCells)), [](auto const&) {});
            dataset.storeChunk<Type>(data, localPhaseSpace_offset, localPhaseSpace_extent);

            /** meta attributes for the data set: unit, range, moving window **/

            pmacc::Selection<simDim> globalDomain = subGrid.getGlobalDomain();
            pmacc::Selection<simDim> totalDomain = subGrid.getTotalDomain();
            // convert things to std::vector<> for the openPMD API to enjoy
            std::vector<int> globalDomainSize{&globalDomain.size[0], &globalDomain.size[0] + simDim};
            std::vector<int> globalDomainOffset{&globalDomain.offset[0], &globalDomain.offset[0] + simDim};
            std::vector<int> totalDomainSize{&totalDomain.size[0], &totalDomain.size[0] + simDim};
            std::vector<int> totalDomainOffset{&totalDomain.offset[0], &totalDomain.offset[0] + simDim};
            std::vector<std::string> globalDomainAxisLabels;
            if(simDim == DIM2)
            {
                globalDomainAxisLabels = {"y", "x"}; // 2D: F[y][x]
            }
            if(simDim == DIM3)
            {
                globalDomainAxisLabels = {"z", "y", "x"}; // 3D: F[z][y][x]
            }

            float_X const dr = cellSize[axis_element.space];

            mesh.setAttribute("globalDomainSize", globalDomainSize);
            mesh.setAttribute("globalDomainOffset", globalDomainOffset);
            mesh.setAttribute("totalDomainSize", totalDomainSize);
            mesh.setAttribute("totalDomainOffset", totalDomainOffset);
            mesh.setAttribute("globalDomainAxisLabels", globalDomainAxisLabels);
            mesh.setAttribute("totalDomainAxisLabels", globalDomainAxisLabels);
            mesh.setAttribute("_global_start", globalPhaseSpace_offset);
            mesh.setAttribute("_global_size", globalPhaseSpace_extent);
            mesh.setAxisLabels({axis_element.spaceAsString(), axis_element.momentumAsString()});
            mesh.setAttribute("sim_unit", unit);
            dataset.setUnitSI(unit);
            {
                using UD = ::openPMD::UnitDimension;
                mesh.setUnitDimension({{UD::I, 1.0}, {UD::T, 1.0}, {UD::L, -1.0}}); // charge density
            }
            mesh.setAttribute("p_unit", pRange_unit);
            mesh.setAttribute("p_min", axis_p_range.first);
            mesh.setAttribute("p_max", axis_p_range.second);
            mesh.setGridGlobalOffset({globalMovingWindowOffset * dr, axis_p_range.first});
            mesh.setAttribute("movingWindowOffset", globalMovingWindowOffset);
            mesh.setAttribute("movingWindowSize", globalMovingWindowSize);
            mesh.setAttribute("dr", dr);
            mesh.setAttribute("dV", CELL_VOLUME);
            mesh.setGridSpacing(std::vector<float_X>{dr, CELL_VOLUME / dr});
            mesh.setAttribute("dr_unit", UNIT_LENGTH);
            iteration.setDt(DELTA_T);
            iteration.setTimeUnitSI(UNIT_TIME);
            /*
             * The value represents an aggregation over one cell, so any value is correct for the mesh position.
             * Just use the center.
             */
            dataset.setPosition(std::vector<float>{0.5, 0.5});

            // avoid deadlock between not finished pmacc tasks and mpi calls in openPMD
            __getTransactionEvent().waitForFinished();

            /** close file ****************************************************/
            iteration.close();
        }
    };

} /* namespace picongpu */
