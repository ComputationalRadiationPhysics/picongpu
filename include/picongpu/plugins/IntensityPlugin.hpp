/* Copyright 2013-2021 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Benjamin Worpitz, Richard Pausch
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

#include <mpi.h>

#include "picongpu/simulation_defines.hpp"

#include "picongpu/plugins/ILightweightPlugin.hpp"
#include "picongpu/fields/FieldE.hpp"

#include <pmacc/memory/boxes/CachedBox.hpp>
#include <pmacc/dimensions/SuperCellDescription.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/memory/shared/Allocate.hpp>
#include <pmacc/memory/Array.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>

#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>


namespace picongpu
{
    using namespace pmacc;

    /* count particles in an area
     * is not optimized, it checks any particle position if it is really a particle
     */
    struct KernelIntensity
    {
        template<typename FieldBox, typename BoxMax, typename BoxIntegral, typename T_Acc>
        DINLINE void operator()(
            T_Acc const& acc,
            FieldBox field,
            DataSpace<simDim> cellsCount,
            BoxMax boxMax,
            BoxIntegral integralBox) const
        {
            typedef MappingDesc::SuperCellSize SuperCellSize;
            PMACC_SMEM(acc, s_integrated, memory::Array<float_X, SuperCellSize::y::value>);
            PMACC_SMEM(acc, s_max, memory::Array<float_X, SuperCellSize::y::value>);


            /*descripe size of a worker block for cached memory*/
            typedef SuperCellDescription<pmacc::math::CT::Int<SuperCellSize::x::value, SuperCellSize::y::value>>
                SuperCell2D;

            auto s_field = CachedBox::create<0, float_32>(acc, SuperCell2D());

            int y = cupla::blockIdx(acc).y * SuperCellSize::y::value + cupla::threadIdx(acc).y;
            int yGlobal = y + GuardSize::y::value * SuperCellSize::y::value;
            const DataSpace<DIM2> threadId(cupla::threadIdx(acc));

            if(threadId.x() == 0)
            {
                // clear destination arrays
                s_integrated[threadId.y()] = float_X(0.0);
                s_max[threadId.y()] = float_X(0.0);
            }
            cupla::__syncthreads(acc);

            // move cell-wise over z direction (without guarding cells)
            for(int z = GuardSize::z::value * SuperCellSize::z::value;
                z < cellsCount.z() - GuardSize::z::value * SuperCellSize::z::value;
                ++z)
            {
                // move supercell-wise over x direction without guarding
                for(int x = GuardSize::x::value * SuperCellSize::x::value + threadId.x();
                    x < cellsCount.x() - GuardSize::x::value * SuperCellSize::x::value;
                    x += SuperCellSize::x::value)
                {
                    const float3_X field_at_point(field(DataSpace<DIM3>(x, yGlobal, z)));
                    s_field(threadId) = pmacc::math::abs2(field_at_point);
                    cupla::__syncthreads(acc);
                    if(threadId.x() == 0)
                    {
                        // master thread moves cell-wise over 2D supercell
                        for(int x_local = 0; x_local < SuperCellSize::x::value; ++x_local)
                        {
                            DataSpace<DIM2> localId(x_local, threadId.y());
                            s_integrated[threadId.y()] += s_field(localId);
                            s_max[threadId.y()] = fmaxf(s_max[threadId.y()], s_field(localId));
                        }
                    }
                }
            }
            cupla::__syncthreads(acc);

            if(threadId.x() == 0)
            {
                /*copy result to global array*/
                integralBox[y] = s_integrated[threadId.y()];
                boxMax[y] = s_max[threadId.y()];
            }
        }
    };

    class IntensityPlugin : public ILightweightPlugin
    {
    private:
        typedef MappingDesc::SuperCellSize SuperCellSize;


        GridBuffer<float_32, DIM1>* localMaxIntensity;
        GridBuffer<float_32, DIM1>* localIntegratedIntensity;
        MappingDesc* cellDescription;
        std::string notifyPeriod;

        std::string pluginName;
        std::string pluginPrefix;

        std::ofstream outFileMax;
        std::ofstream outFileIntegrated;
        /*only rank 0 create a file*/
        bool writeToFile;

    public:
        /*! Calculate the max und integrated E-Field energy over laser propagation direction (in our case Y)
         * max is only the SI  value of the amplitude (V/m)
         * integrated is the integral of amplidude of X and Z on Y position (is V/m in cell volume)
         */
        IntensityPlugin()
            : pluginName("IntensityPlugin: calculate the maximum and integrated E-Field energy\nover laser "
                         "propagation direction")
            , pluginPrefix(FieldE::getName() + std::string("_intensity"))
            , localMaxIntensity(nullptr)
            , localIntegratedIntensity(nullptr)
            , cellDescription(nullptr)
            , writeToFile(false)
        {
            Environment<>::get().PluginConnector().registerPlugin(this);
        }

        virtual ~IntensityPlugin()
        {
        }

        void notify(uint32_t currentStep)
        {
            calcIntensity(currentStep);
            combineData(currentStep);
        }

        void pluginRegisterHelp(po::options_description& desc)
        {
            desc.add_options()(
                (pluginPrefix + ".period").c_str(),
                po::value<std::string>(&notifyPeriod),
                "enable plugin [for each n-th step]");
        }

        std::string pluginGetName() const
        {
            return pluginName;
        }

        void setMappingDescription(MappingDesc* cellDescription)
        {
            this->cellDescription = cellDescription;
        }

    private:
        void pluginLoad()
        {
            if(!notifyPeriod.empty())
            {
                writeToFile = Environment<simDim>::get().GridController().getGlobalRank() == 0;
                int yCells = cellDescription->getGridLayout().getDataSpaceWithoutGuarding().y();

                localMaxIntensity
                    = new GridBuffer<float_32, DIM1>(DataSpace<DIM1>(yCells)); // create one int on gpu und host
                localIntegratedIntensity
                    = new GridBuffer<float_32, DIM1>(DataSpace<DIM1>(yCells)); // create one int on gpu und host

                if(writeToFile)
                {
                    createFile(pluginPrefix + "_max.dat", outFileMax);
                    createFile(pluginPrefix + "_integrated.dat", outFileIntegrated);
                }

                Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);
            }
        }

        void pluginUnload()
        {
            if(!notifyPeriod.empty())
            {
                if(writeToFile)
                {
                    flushAndCloseFile(outFileIntegrated);
                    flushAndCloseFile(outFileMax);
                }
                __delete(localMaxIntensity);
                __delete(localIntegratedIntensity);
            }
        }

    private:
        /* reduce data from all gpus to one array
         * @param currentStep simulation step
         */
        void combineData(uint32_t currentStep)
        {
            const DataSpace<simDim> localSize(cellDescription->getGridLayout().getDataSpaceWithoutGuarding());
            Window window(MovingWindow::getInstance().getWindow(currentStep));

            const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();

            const int yGlobalSize = subGrid.getGlobalDomain().size.y();
            const int yLocalSize = localSize.y();

            const int gpus = Environment<simDim>::get().GridController().getGpuNodes().productOfComponents();


            /**\todo: fixme I cant work with not regular domains (use mpi_gatherv)*/
            DataSpace<simDim> globalRootCell(subGrid.getLocalDomain().offset);
            int yOffset = globalRootCell.y();
            int* yOffsetsAll = new int[gpus];
            float_32* maxAll = new float_32[yGlobalSize];
            float_32* maxAllTmp = new float_32[yLocalSize * gpus];
            memset(maxAll, 0, sizeof(float_32) * yGlobalSize);
            float_32* integretedAll = new float_32[yGlobalSize];
            float_32* integretedAllTmp = new float_32[yLocalSize * gpus];
            memset(integretedAll, 0, sizeof(float_32) * yGlobalSize);

            // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
            __getTransactionEvent().waitForFinished();
            MPI_CHECK(MPI_Gather(&yOffset, 1, MPI_INT, yOffsetsAll, 1, MPI_INT, 0, MPI_COMM_WORLD));

            MPI_CHECK(MPI_Gather(
                localMaxIntensity->getHostBuffer().getBasePointer(),
                yLocalSize,
                MPI_FLOAT,
                maxAllTmp,
                yLocalSize,
                MPI_FLOAT,
                0,
                MPI_COMM_WORLD));
            MPI_CHECK(MPI_Gather(
                localIntegratedIntensity->getHostBuffer().getBasePointer(),
                yLocalSize,
                MPI_FLOAT,
                integretedAllTmp,
                yLocalSize,
                MPI_FLOAT,
                0,
                MPI_COMM_WORLD));

            if(writeToFile)
            {
                for(int i = 0; i < gpus; ++i)
                {
                    int gOffset = yOffsetsAll[i];
                    int tmpOff = yLocalSize * i;
                    for(int y = 0; y < yLocalSize; ++y)
                    {
                        maxAll[gOffset + y] = std::max(maxAllTmp[tmpOff + y], maxAll[gOffset + y]);
                        integretedAll[gOffset + y] += integretedAllTmp[tmpOff + y];
                    }
                }

                const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);
                size_t physicelYCellOffset = numSlides * yLocalSize + window.globalDimensions.offset.y();
                writeFile(
                    currentStep,
                    maxAll + window.globalDimensions.offset.y(),
                    window.globalDimensions.size.y(),
                    physicelYCellOffset,
                    outFileMax,
                    UNIT_EFIELD);

                float_64 unit = UNIT_EFIELD * CELL_VOLUME * SI::EPS0_SI;
                for(uint32_t i = 0; i < simDim; ++i)
                    unit *= UNIT_LENGTH;

                writeFile(
                    currentStep,
                    integretedAll + window.globalDimensions.offset.y(),
                    window.globalDimensions.size.y(),
                    physicelYCellOffset,
                    outFileIntegrated,
                    unit);
            }

            __deleteArray(yOffsetsAll);
            __deleteArray(maxAll);
            __deleteArray(integretedAll);
            __deleteArray(maxAllTmp);
            __deleteArray(integretedAllTmp);
        }

        /* write data from array to a file
         * write current step to first column
         *
         * @param currentStep simulation step
         * @param array shifted source array (begin printing from first element)
         * @param count number of elements to print
         * @param physicalYOffset offset in cells to the absolute simulation begin
         * @param stream destination stream
         * @param unit unit to scale values from pic units to si units
         */
        void writeFile(
            size_t currentStep,
            float* array,
            size_t count,
            size_t physicalYOffset,
            std::ofstream& stream,
            float_64 unit)
        {
            stream << currentStep << " ";
            for(size_t i = 0; i < count; ++i)
            {
                stream << (physicalYOffset + i) * SI::CELL_HEIGHT_SI << " ";
            }
            stream << std::endl << currentStep << " ";
            for(size_t i = 0; i < count; ++i)
            {
                stream << sqrt((float_64)(array[i])) * unit << " ";
            }
            stream << std::endl;
        }

        /* run calculation of intensity
         * sync all result data to host side
         *
         * @param currenstep simulation step
         */
        void calcIntensity(uint32_t)
        {
            DataConnector& dc = Environment<>::get().DataConnector();

            auto fieldE = dc.get<FieldE>(FieldE::getName(), true);

            /*start only worker for any supercell in laser propagation direction*/
            DataSpace<DIM2> grid(
                1,
                cellDescription->getGridSuperCells().y() - cellDescription->getGuardingSuperCells().y());
            /*use only 2D slice XY for supercell handling*/
            typedef typename MappingDesc::SuperCellSize SuperCellSize;
            auto block = pmacc::math::CT::Vector<SuperCellSize::x, SuperCellSize::y>::toRT();

            PMACC_KERNEL(KernelIntensity{})
            (grid, block)(
                fieldE->getDeviceDataBox(),
                fieldE->getGridLayout().getDataSpace(),
                localMaxIntensity->getDeviceBuffer().getDataBox(),
                localIntegratedIntensity->getDeviceBuffer().getDataBox());

            dc.releaseData(FieldE::getName());

            localMaxIntensity->deviceToHost();
            localIntegratedIntensity->deviceToHost();
        }

        /*create a file with given filename
         * @param filename name of the output file
         * @param stream ref on a stream object
         */
        void createFile(std::string filename, std::ofstream& stream)
        {
            stream.open(filename.c_str(), std::ofstream::out | std::ostream::trunc);
            if(!stream)
            {
                std::cerr << "Can't open file [" << filename << "] for output, diasble plugin output. " << std::endl;
                writeToFile = false;
            }
            stream << "#step position_in_laser_propagation_direction" << std::endl;
            stream << "#step amplitude_data[*]" << std::endl;
        }

        /* close and flash a file stream object
         * @param stream stream which must closed
         */
        void flushAndCloseFile(std::ofstream& stream)
        {
            stream.flush();
            stream << std::endl; // now all data are written to file
            if(stream.fail())
                std::cerr << "Error on flushing file in IntensityPlugin. " << std::endl;
            stream.close();
        }
    };

} // namespace picongpu
