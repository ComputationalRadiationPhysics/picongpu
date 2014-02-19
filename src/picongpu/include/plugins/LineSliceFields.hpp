/**
 * Copyright 2013 Axel Huebl, Heiko Burau, Rene Widera
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
 


#ifndef LINESLICEFIELDS_HPP
#define	LINESLICEFIELDS_HPP

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "types.h"
#include "simulation_defines.hpp"
#include "simulation_types.hpp"

#include "simulation_classTypes.hpp"

#include "fields/FieldB.hpp"
#include "fields/FieldE.hpp"

#include "mappings/simulation/GridController.hpp"

#include "basicOperations.hpp"
#include "dimensions/DataSpaceOperations.hpp"
#include "plugins/IPluginModule.hpp"

namespace picongpu
{

    using namespace PMacc;

    namespace po = boost::program_options;

    typedef typename FieldB::DataBoxType B_DataBox;
    typedef typename FieldE::DataBoxType E_DataBox;

    template<class Mapping>
    __global__ void kernelLineSliceFields(E_DataBox fieldE, B_DataBox fieldB,
    float3_X* sliceDataField,
    DataSpace<simDim> globalCellIdOffset,
    DataSpace<simDim> globalNrOfCells,
    Mapping mapper)
    {

        typedef typename Mapping::SuperCellSize SuperCellSize;

        const DataSpace<simDim > threadIndex(threadIdx);
        const int linearThreadIdx = DataSpaceOperations<simDim>::template map<SuperCellSize > (threadIndex);

        const DataSpace<simDim> superCellIdx(mapper.getSuperCellIndex(DataSpace<simDim > (blockIdx)));


        __syncthreads();

        // GPU-local cell id with lower GPU-local guarding
        const DataSpace<simDim> localCell(superCellIdx * SuperCellSize() + threadIndex);

        const float3_X b = fieldB(localCell);
        const float3_X e = fieldE(localCell);

        // GPU-local cell id without lower GPU-local guarding
        const DataSpace<simDim> localCellWG(
                localCell
                - SuperCellSize::getDataSpace() * mapper.getGuardingSuperCells());
        // global cell id
        const DataSpace<simDim> globalCell = localCellWG + globalCellIdOffset;


        // slice out one cell along an axis
        if ((globalCell.x() == globalNrOfCells.x() / 2))
#if(SIMDIM==DIM3)
                if(globalCell.z() == globalNrOfCells.z() / 2)
#endif
            sliceDataField[localCellWG.y()] = e;

        __syncthreads();
    }

    class LineSliceFields : public ISimulationIO, public IPluginModule
    {
    private:
        FieldE* fieldE;
        FieldB* fieldB;

        MappingDesc *cellDescription;
        uint32_t notifyFrequency;

        GridBuffer<float3_X, DIM1> *sliceDataField;

        std::ofstream outfile;

    public:

        LineSliceFields() :
        fieldE(NULL),
        fieldB(NULL),
        cellDescription(NULL),
        notifyFrequency(0)
        {
            ModuleConnector::getInstance().registerModule(this);
        }

        virtual ~LineSliceFields()
        {

        }

        void notify(uint32_t currentStep)
        {
            typedef typename MappingDesc::SuperCellSize SuperCellSize;

            DataConnector& dc = DataConnector::getInstance();

            fieldE = &(dc.getData<FieldE > (FIELD_E, true));
            fieldB = &(dc.getData<FieldB > (FIELD_B, true));


            const int rank = GridController<simDim>::getInstance().getGlobalRank();
            getLineSliceFields < CORE + BORDER > ();
            
            PMACC_AUTO(simBox,SubGrid<simDim>::getInstance().getSimulationBox());
            

            // number of cells on the current CPU for each direction
            const DataSpace<simDim> nrOfGpuCells = cellDescription->getGridLayout().getDataSpaceWithoutGuarding();

            
            // global cell id offset (without guardings!)
            // returns the global id offset of the "first" border cell on a GPU
            const DataSpace<simDim> globalCellIdOffset(simBox.getGlobalOffset());

            // global number of cells for whole simulation: local cells on GPU * GPUs
            // (assumed same size on each gpu :-/  -> todo: provide interface!)
            //! \todo create a function for: global number of cells for whole simulation
            //!
            const DataSpace<simDim> globalNrOfCells = simBox.getGlobalSize();

            /*FORMAT OUTPUT*/
            /** \todo add float3_X with position of the cell to output*/
            // check if the current GPU contains the "middle slice" along
            // X_global / 2; Y_global / 2 over Z
            if (globalCellIdOffset.x() <= globalNrOfCells.x() / 2 &&
                    globalCellIdOffset.x() + nrOfGpuCells.x() > globalNrOfCells.x() / 2)
#if(SIMDIM==DIM3)
                if( globalCellIdOffset.z() <= globalNrOfCells.z() / 2 &&
                    globalCellIdOffset.z() + nrOfGpuCells.z() > globalNrOfCells.z() / 2)
#endif
                for (int i = 0; i < nrOfGpuCells.y(); ++i)
                {
                    const double xPos = double( i + globalCellIdOffset.y()) * SI::CELL_HEIGHT_SI;

                    outfile << currentStep << " " << rank << " ";
                    outfile << xPos << " "
                            /*<< sliceDataField->getHostBuffer().getDataBox()[i] */
                            << double(sliceDataField->getHostBuffer().getDataBox()[i].x()) * UNIT_EFIELD << " "
                            << double(sliceDataField->getHostBuffer().getDataBox()[i].y()) * UNIT_EFIELD << " "
                            << double(sliceDataField->getHostBuffer().getDataBox()[i].z()) * UNIT_EFIELD << " "
                            << "\n";
                }

            /* outfile << "[ANALYSIS] [" << rank << "] [COUNTER] [LineSliceFields] [" << currentStep << "] " <<
                    sliceDataField << "\n"; */

            // free line to separate timesteps in gnuplot via the "index" option
            outfile << std::endl;
        }

        void moduleRegisterHelp(po::options_description& desc)
        {
            desc.add_options()
                    ("lslice.period", po::value<uint32_t > (&notifyFrequency), "enable analyser [for each n-th step]");
        }

        std::string moduleGetName() const
        {
            return "LineSliceFields";
        }

        void setMappingDescription(MappingDesc *cellDescription)
        {
            this->cellDescription = cellDescription;
        }

    private:
        void moduleLoad()
        {
            if (notifyFrequency > 0)
            {
                // number of cells on the current CPU for each direction
                const DataSpace<simDim> nrOfGpuCells = SubGrid<simDim>::getInstance().getSimulationBox().getLocalSize();

                // create as much storage as cells in the direction we are interested in:
                // on gpu und host
                sliceDataField = new GridBuffer<float3_X, DIM1 >
                        (DataSpace<DIM1 > (nrOfGpuCells.y()));

                DataConnector::getInstance().registerObserver(this, notifyFrequency);

                const int rank = GridController<simDim>::getInstance().getGlobalRank();

                // open output file
                std::stringstream oFileName;
                oFileName << "lineSliceFields_" << rank << ".txt";

                outfile.open(oFileName.str().c_str(), std::ofstream::out | std::ostream::trunc);
                outfile.precision(8);
                outfile.setf(std::ios::scientific);
            }
        }

        void moduleUnload()
        {
            if (notifyFrequency > 0)
            {
                if (sliceDataField)
                    delete sliceDataField;

                // close the output file
                outfile.close();
            }
        }

        template< uint32_t AREA>
        void getLineSliceFields()
        {
            typedef typename MappingDesc::SuperCellSize SuperCellSize;

            const float3_X tmpFloat3(float3_X(float_X(0.0), float_X(0.0), float_X(0.0)));
            sliceDataField->getDeviceBuffer().setValue(tmpFloat3);
            dim3 block(SuperCellSize::getDataSpace());

            PMACC_AUTO(simBox,SubGrid<simDim>::getInstance().getSimulationBox());
            // global cell id offset (without guardings!)
            // returns the global id offset of the "first" border cell on a GPU
            const DataSpace<simDim> globalCellIdOffset(simBox.getGlobalOffset());

            // global number of cells for whole simulation: local cells on GPU * GPUs
            // (assumed same size on each gpu :-/  -> todo: provide interface!)
            //! \todo create a function for: global number of cells for whole simulation
            //!
            const DataSpace<simDim> localNrOfCells(simBox.getLocalSize());
            const DataSpace<simDim> globalNrOfCells (simBox.getGlobalSize());


            __picKernelArea(kernelLineSliceFields, *cellDescription, AREA)
                    (block)
                    (fieldE->getDeviceDataBox(),
                    fieldB->getDeviceDataBox(),
                    sliceDataField->getDeviceBuffer().getBasePointer(),
                    globalCellIdOffset,
                    globalNrOfCells
                    );
            sliceDataField->deviceToHost();
            //return sliceDataField->getHostBuffer().getDataBox()[0];
        }

    };

} // namespace picongpu close


#endif	/* LINESLICEFIELDS_HPP */

