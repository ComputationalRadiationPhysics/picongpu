/* Copyright 2013-2019 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch,
 *                     Benjamin Worpitz
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
#include <vector>

/*pic default*/
#include <pmacc/types.hpp>
#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/Fields.def"
#include <pmacc/fields/SimulationFieldHelper.hpp>
#include <pmacc/dataManagement/ISimulationData.hpp>

/*PMacc*/
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/memory/boxes/DataBox.hpp>
#include <pmacc/memory/boxes/PitchedBox.hpp>

#include <pmacc/math/Vector.hpp>
#include "picongpu/particles/Particles.hpp"

namespace picongpu
{
using namespace pmacc;

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
    typedef promoteType<float_64, ValueType>::type UnitValueType;
    static constexpr int numComponents = ValueType::dim;

    typedef DataBox<PitchedBox<ValueType, simDim> > DataBoxType;

    FieldJ(MappingDesc cellDescription);

    virtual ~FieldJ();

    virtual EventTask asyncCommunication(EventTask serialEvent);

    GridLayout<simDim> getGridLayout();

    void reset(uint32_t currentStep);

    /** Assign a value to all cells
     *
     * Example usage:
     * ```C++
     *   FieldJ::ValueType zeroJ( FieldJ::ValueType::create(0.) );
     *   fieldJ->assign( zeroJ );
     * ```
     *
     * \param value date to fill all cells with
     */
    void assign(ValueType value);

    HDINLINE static UnitValueType getUnit();

    /** powers of the 7 base measures
     *
     * characterizing the record's unit in SI
     * (length L, mass M, time T, electric current I,
     *  thermodynamic temperature theta, amount of substance N,
     *  luminous intensity J) */
    HINLINE static std::vector<float_64> getUnitDimension();

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

template<typename T_SpeciesType, typename T_Area>
struct ComputeCurrent
{
    using SpeciesType = T_SpeciesType;
    using FrameType = typename SpeciesType::FrameType;

    HINLINE void operator()( const uint32_t currentStep ) const
    {
        DataConnector &dc = Environment<>::get().DataConnector();
        auto species = dc.get< SpeciesType >( FrameType::getName(), true );
        auto fieldJ = dc.get< FieldJ >( FieldJ::getName(), true );

        fieldJ->computeCurrent< T_Area::value, SpeciesType >( *species, currentStep );

        dc.releaseData( FrameType::getName() );
        dc.releaseData( FieldJ::getName() );
    }
};

} // namespace picongpu
