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
#include "picongpu/fields/MaxwellSolver/Yee/Yee.hpp"
#include "picongpu/fields/MaxwellSolver/Yee/Curl.hpp"
#include "picongpu/fields/FieldManipulator.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldManipulator.hpp"
#include "picongpu/fields/MaxwellSolver/Yee/YeePML.kernel"
#include "picongpu/fields/numericalCellTypes/NumericalCellTypes.hpp"
#include "picongpu/fields/LaserPhysics.hpp"

#include <pmacc/nvidia/functors/Assign.hpp>
#include <pmacc/mappings/threads/ThreadCollective.hpp>
#include <pmacc/memory/boxes/CachedBox.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>

#include <cstdint>


namespace picongpu
{
namespace fields
{
namespace maxwellSolver
{

    // Yee solver with PML absorber
    // With the current design of field solvers, PML has to be specific for each solver
    // (so it is not possible to do it the same universal way the current absorber is done)
    // How to better organize it is ofc a question, maybe it is better to use trait-like design
    //
    // There probably is a way to to provide the universal PML for all solvers with a single
    // implementation, but it would require a significant redesign. Idea: elevate the current Curl
    // approach to even a higher level of abstraction, by making each solver provide the whole
    // discretization used as a type. Then a PML could be built on top of this discretization.
    // However, from the pure solver point of view this would probably look somewhat unnatural. 
    // (Probably not that relevant now.)
    template<
        typename T_CurrentInterpolation,
        class CurlE,
        class CurlB
    >
    class YeePML : public Yee< T_CurrentInterpolation, CurlE, CurlB >
    {
    private:

        // Polynomial order of the absorber strength growth towards borders
        // (often denoted 'n' in the literature)
        uint32_t pmlOrder;

        // These parameters are specific for each side,
        // as a temporary solution sides are encoded in the absorber format (to be fixed)
        uint32_t thickness[3][2]; // unit: cells
        float_X maxAbsorbtion[3][2]; // often denoted 'sigma_max' in the literature

        template<uint32_t AREA>
        void updateE()
        {
            // In the border area apply PML when necessary
            typedef SuperCellDescription<
                    SuperCellSize,
                    typename CurlB::LowerMargin,
                    typename CurlB::UpperMargin
                    > BlockArea;
            using Difference = typename CurlB::Difference;

            AreaMapping<AREA, MappingDesc> mapper(m_cellDescription);

            constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
                pmacc::math::CT::volume< SuperCellSize >::type::value
            >::value;

            auto splitFieldE = this->fieldB->getDeviceDataBox(); /// placeholder for now
            PMACC_KERNEL(yeePML::KernelUpdateE< numWorkers, BlockArea >{ })
                ( mapper.getGridDim(), numWorkers )(
                    CurlB( ),
                    this->fieldE->getDeviceDataBox(),
                    splitFieldE,
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

            auto splitFieldB = this->fieldB->getDeviceDataBox(); /// placeholder for now
            PMACC_KERNEL(yeePML::KernelUpdateBHalf< numWorkers, BlockArea >{ })
                (mapper.getGridDim(), numWorkers)(
                    CurlE(),
                    this->fieldB->getDeviceDataBox(),
                    splitFieldB,
                    this->fieldE->getDeviceDataBox(),
                    mapper
                );
        }

    public:

        YeePML(MappingDesc cellDescription) : Yee(cellDescription)
        {
            // for now reuse parameters of the old absorber
            // WARNING: this is only correct for a single process,
            // TODO: for the general case needs to take into account domain decomposition
            // so that thickness is 0 for "internal" boundaries between domains
            for (int axis = 0; axis < 3; axis++)
                for (int direction = 0; direction < 2; direction++)
                    thickness[axis][direction] = ABSORBER_CELLS[axis][direction];

            pmlOrder = 4; // for now hardcoded with a good default value
            constexpr float_X baseReflectionLevel = 1e-8; // for now hardcoded with a good default value
            for (int axis = 0; axis < 3; axis++)
                for (int direction = 0; direction < 2; direction++)
                    if (thickness[axis][direction])
                        maxAbsorbtion[axis][direction] = -log(baseReflectionLevel) * (pmlOrder + 1);
                    else
                        maxAbsorbtion[axis][direction] = 0;
        }

        // TODO: the following update_beforeCurrent() and update_afterCurrent()
        // are exactly the same as in the Yee solver (but internally called
        // updateBHalf, updateE are not). It would be ofc much better to not
        // duplicate this logic, and this should be possible rather easily
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
            if (laserProfiles::Selected::INIT_TIME > float_X(0.0))
                LaserPhysics{}(currentStep);

            EventTask eRfieldE = fieldE->asyncCommunication(__getTransactionEvent());

            updateBHalf < CORE> ();
            __setTransactionEvent(eRfieldE);
            updateBHalf < BORDER > ();

            EventTask eRfieldB = fieldB->asyncCommunication(__getTransactionEvent());
            __setTransactionEvent(eRfieldB);
        }

        static pmacc::traits::StringProperty getStringProperties()
        {
            pmacc::traits::StringProperty propList( "name", "YeePML" );
            return propList;
        }
    };

} // namespace maxwellSolver
} // namespace fields
} // picongpu
