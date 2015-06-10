/**
 * Copyright 2013-2015 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch
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

#include <string>

/*pic default*/
#include "types.h"
#include "simulation_defines.hpp"
#include "simulation_classTypes.hpp"

#include "Fields.def"
#include "fields/SimulationFieldHelper.hpp"
#include "dataManagement/ISimulationData.hpp"

/*libPMacc*/
#include "memory/buffers/GridBuffer.hpp"
#include "mappings/simulation/GridController.hpp"
#include "memory/boxes/DataBox.hpp"
#include "memory/boxes/PitchedBox.hpp"

#include "math/Vector.hpp"
#include "particles/Particles.hpp"

namespace picongpu
{
using namespace PMacc;

// The fieldJ saves the current density j
//
// j = current / area
// To obtain the current which goes out of a cell in the 3 directions,
// calculate J = float3_X( j.x() * cellSize.y() * cellSize.z(),
//                            j.y() * cellSize.x() * cellSize.z(),
//                            j.z() * cellSize.x() * cellSize.y())
//

class FieldJ : public SimulationFieldHelper<MappingDesc>, public ISimulationData
{
public:

    typedef float3_X ValueType;
    typedef typename promoteType<float_64, ValueType>::type UnitValueType;
    static const int numComponents = ValueType::dim;

    typedef DataBox<PitchedBox<ValueType, simDim> > DataBoxType;

    FieldJ(MappingDesc cellDescription);

    virtual ~FieldJ();

    virtual EventTask asyncCommunication(EventTask serialEvent);

    void init(FieldE &fieldE, FieldB &fieldB);

    GridLayout<simDim> getGridLayout();

    void reset(uint32_t currentStep);

    void clear();

    HDINLINE static UnitValueType getUnit();

    static std::string getName();

    static uint32_t getCommTag();

    template<uint32_t AREA, class ParticlesClass>
    void computeCurrent(ParticlesClass &parClass, uint32_t currentStep);

    template<uint32_t AREA, class T_CurrentInterpolation>
    void addCurrentToEMF( T_CurrentInterpolation& myCurrentInterpolation );

    SimulationDataId getUniqueId();

    void synchronize();

    void syncToDevice()
    {
        ValueType tmp = float3_X(0., 0., 0.);
        fieldJ.getDeviceBuffer().setValue(tmp);
    }

    DataBoxType getDeviceDataBox()
    {
        return fieldJ.getDeviceBuffer().getDataBox();
    }

    DataBoxType getHostDataBox()
    {
        return fieldJ.getHostBuffer().getDataBox();
    }

    GridBuffer<ValueType, simDim> &getGridBuffer();

    /* Bash particles in a direction.
     * Copy all particles from the guard of a direction to the device exchange buffer
     */
    void bashField(uint32_t exchangeType);

    /* Insert all particles which are in device exchange buffer
     */
    void insertField(uint32_t exchangeType);

private:

    GridBuffer<ValueType, simDim> fieldJ;
    GridBuffer<ValueType, simDim>* fieldJrecv;

    FieldE *fieldE;
    FieldB *fieldB;
};

template<typename T_SpeciesName, typename T_Area>
struct ComputeCurrent
{

    template<typename T_StorageTuple>
    HINLINE void operator()( FieldJ* fieldJ,
                            T_StorageTuple& tuple,
                            const uint32_t currentStep) const
    {
        typedef T_SpeciesName SpeciesName;
        typedef typename SpeciesName::type SpeciesType;

        PMACC_AUTO(speciesPtr, tuple[SpeciesName()]);
        fieldJ->computeCurrent<T_Area::value, SpeciesType> (*speciesPtr, currentStep);
    }
};

} // namespace picongpu
