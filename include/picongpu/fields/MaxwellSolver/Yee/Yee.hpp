/* Copyright 2013-2019 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz
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
#include "picongpu/fields/MaxwellSolver/Yee/Yee.def"
#include "picongpu/fields/MaxwellSolver/Yee/Curl.hpp"
#include "picongpu/fields/FieldManipulator.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldManipulator.hpp"
#include "picongpu/fields/MaxwellSolver/Yee/Yee.kernel"
#include "picongpu/fields/numericalCellTypes/NumericalCellTypes.hpp"
#include "picongpu/fields/LaserPhysics.hpp"

#include <pmacc/nvidia/functors/Assign.hpp>
#include <pmacc/mappings/threads/ThreadCollective.hpp>
#include <pmacc/memory/boxes/CachedBox.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>


namespace picongpu
{
namespace fields
{
namespace maxwellSolver
{

    template<
        typename T_CurrentInterpolation,
        class CurlE,
        class CurlB
    >
    class Yee
    {
    private:
        typedef MappingDesc::SuperCellSize SuperCellSize;


        std::shared_ptr< FieldE > fieldE;
        std::shared_ptr< FieldB > fieldB;
        MappingDesc m_cellDescription;

        template<uint32_t AREA>
        void updateE()
        {
            /* Courant-Friedrichs-Levy-Condition for Yee Field Solver: */
            PMACC_CASSERT_MSG(Courant_Friedrichs_Levy_condition_failure____check_your_grid_param_file,
                (SPEED_OF_LIGHT*SPEED_OF_LIGHT*DELTA_T*DELTA_T*INV_CELL2_SUM)<=1.0);

            typedef SuperCellDescription<
                    SuperCellSize,
                    typename CurlB::LowerMargin,
                    typename CurlB::UpperMargin
                    > BlockArea;

            AreaMapping<AREA, MappingDesc> mapper(m_cellDescription);

            constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
                pmacc::math::CT::volume< SuperCellSize >::type::value
            >::value;

            PMACC_KERNEL(yee::KernelUpdateE< numWorkers, BlockArea >{ })
                ( mapper.getGridDim(), numWorkers )(
                    CurlB( ),
                    this->fieldE->getDeviceDataBox(),
                    this->fieldB->getDeviceDataBox(),
                    mapper
                );
        }

        template<uint32_t AREA>
        void updateBHalf()
        {
            typedef SuperCellDescription<
                    SuperCellSize,
                    typename CurlE::LowerMargin,
                    typename CurlE::UpperMargin
                    > BlockArea;

            AreaMapping<AREA, MappingDesc> mapper(m_cellDescription);

            constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
                pmacc::math::CT::volume< SuperCellSize >::type::value
            >::value;

            PMACC_KERNEL(yee::KernelUpdateBHalf< numWorkers, BlockArea >{ })
                ( mapper.getGridDim(), numWorkers )(
                    CurlE( ),
                    this->fieldB->getDeviceDataBox(),
                    this->fieldE->getDeviceDataBox(),
                    mapper
                );
        }

    public:

        using NummericalCellType = picongpu::numericalCellTypes::YeeCell;
        using CurrentInterpolation = T_CurrentInterpolation;

        Yee(MappingDesc cellDescription) : m_cellDescription(cellDescription)
        {
            DataConnector &dc = Environment<>::get().DataConnector();

            this->fieldE = dc.get< FieldE >( FieldE::getName(), true );
            this->fieldB = dc.get< FieldB >( FieldB::getName(), true );
        }

        void update_beforeCurrent(uint32_t)
        {
            updateBHalf < CORE+BORDER >();
            EventTask eRfieldB = fieldB->asyncCommunication(__getTransactionEvent());

            updateE<CORE>();
            __setTransactionEvent(eRfieldB);
            updateE<BORDER>();
        }

        void update_afterCurrent(uint32_t currentStep)
        {
            FieldManipulator::absorbBorder(currentStep,this->m_cellDescription, this->fieldE->getDeviceDataBox());
            if (laserProfiles::Selected::INIT_TIME > float_X(0.0))
                LaserPhysics{}(currentStep);

            EventTask eRfieldE = fieldE->asyncCommunication(__getTransactionEvent());

            updateBHalf < CORE> ();
            __setTransactionEvent(eRfieldE);
            updateBHalf < BORDER > ();

            FieldManipulator::absorbBorder(currentStep,this->m_cellDescription, fieldB->getDeviceDataBox());

            EventTask eRfieldB = fieldB->asyncCommunication(__getTransactionEvent());
            __setTransactionEvent(eRfieldB);
        }

        static pmacc::traits::StringProperty getStringProperties()
        {
            pmacc::traits::StringProperty propList( "name", "Yee" );
            return propList;
        }
    };

} // namespace maxwellSolver
} // namespace fields
} // picongpu
