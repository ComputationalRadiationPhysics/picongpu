/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt,
 *                     Richard Pausch, Benjamin Worpitz, Pawel Ordyna
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

#include "picongpu/fields/FieldTmp.hpp"

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/FieldTmp.kernel"
#include "picongpu/fields/MaxwellSolver/Solvers.hpp"
#include "picongpu/param/fileOutput.param"
#include "picongpu/particles/filter/filter.hpp"
#include "picongpu/particles/traits/GetInterpolation.hpp"
#include "picongpu/traits/GetMargin.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/dimensions/SuperCellDescription.hpp>
#include <pmacc/fields/operations/AddExchangeToBorder.hpp>
#include <pmacc/fields/operations/CopyGuardToExchange.hpp>
#include <pmacc/fields/tasks/FieldFactory.hpp>
#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/traits/GetUniqueTypeId.hpp>

#include <memory>
#include <string>

namespace picongpu
{
    using namespace pmacc;

    template<typename A, typename B>
    using SpeciesLowerMarginOp =
        typename pmacc::math::CT::max<A, typename GetLowerMargin<typename GetInterpolation<B>::type>::type>::type;
    template<typename A, typename B>
    using SpeciesUpperMarginOp =
        typename pmacc::math::CT::max<A, typename GetUpperMargin<typename GetInterpolation<B>::type>::type>::type;

    template<typename A, typename B>
    using FieldTmpLowerMarginOp = typename pmacc::math::CT::max<A, typename GetLowerMargin<B>::type>::type;
    template<typename A, typename B>
    using FieldTmpUpperMarginOp = typename pmacc::math::CT::max<A, typename GetUpperMargin<B>::type>::type;

    FieldTmp::FieldTmp(MappingDesc const& cellDescription, uint32_t slotId)
        : SimulationFieldHelper<MappingDesc>(cellDescription)
        , m_slotId(slotId)
    {
        /* Since this class is instantiated for each temporary field slot,
         * use getUniqueId( ) directly to get unique tags for each instance.
         *
         * Warning: this usage relies on the same order of calls to getNextId() on all MPI ranks
         */
        m_commTagScatter = pmacc::traits::getUniqueId();
        m_commTagGather = pmacc::traits::getUniqueId();

        using Buffer = GridBuffer<ValueType, simDim>;
        fieldTmp = std::make_unique<Buffer>(cellDescription.getGridLayout());

        if(fieldTmpSupportGatherCommunication)
            fieldTmpRecv = std::make_unique<Buffer>(fieldTmp->getDeviceBuffer(), cellDescription.getGridLayout());

        /** \todo The exchange has to be resetted and set again regarding the
         *  temporary "Fill-"Functor we want to use.
         *
         *  Problem: buffers don't allow "bigger" exchange during run time.
         *           so let's stay with the maximum guards.
         */
        const DataSpace<simDim> coreBorderSize = cellDescription.getGridLayout().sizeWithoutGuardND();

        using VectorSpeciesWithInterpolation =
            typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, interpolation<>>::type;

        /* ------------------ lower margin  ----------------------------------*/
        using SpeciesLowerMargin = pmacc::mp_fold<
            VectorSpeciesWithInterpolation,
            typename pmacc::math::CT::make_Int<simDim, 0>::type,
            SpeciesLowerMarginOp>;

        using FieldTmpLowerMargin = pmacc::
            mp_fold<FieldTmpSolvers, typename pmacc::math::CT::make_Int<simDim, 0>::type, FieldTmpLowerMarginOp>;

        using SpeciesFieldTmpLowerMargin = pmacc::math::CT::max<SpeciesLowerMargin, FieldTmpLowerMargin>::type;

        using FieldSolverLowerMargin = GetLowerMargin<fields::Solver>::type;

        using LowerMargin = pmacc::math::CT::max<SpeciesFieldTmpLowerMargin, FieldSolverLowerMargin>::type;


        /* ------------------ upper margin  -----------------------------------*/

        using SpeciesUpperMargin = pmacc::mp_fold<
            VectorSpeciesWithInterpolation,
            typename pmacc::math::CT::make_Int<simDim, 0>::type,
            SpeciesUpperMarginOp>;

        using FieldTmpUpperMargin = pmacc::
            mp_fold<FieldTmpSolvers, typename pmacc::math::CT::make_Int<simDim, 0>::type, FieldTmpUpperMarginOp>;

        using SpeciesFieldTmpUpperMargin = pmacc::math::CT::max<SpeciesUpperMargin, FieldTmpUpperMargin>::type;

        using FieldSolverUpperMargin = GetUpperMargin<fields::Solver>::type;

        using UpperMargin = pmacc::math::CT::max<SpeciesFieldTmpUpperMargin, FieldSolverUpperMargin>::type;

        const DataSpace<simDim> originGuard(LowerMargin().toRT());
        const DataSpace<simDim> endGuard(UpperMargin().toRT());

        /*go over all directions*/
        for(uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i)
        {
            DataSpace<simDim> relativMask = Mask::getRelativeDirections<simDim>(i);
            /*guarding cells depend on direction
             */
            DataSpace<simDim> guardingCells;
            for(uint32_t d = 0; d < simDim; ++d)
            {
                /*originGuard and endGuard are switch because we send data
                 * e.g. from left I get endGuardingCells and from right I originGuardingCells
                 */
                switch(relativMask[d])
                {
                    // receive from negativ side to positiv (end) guarding cells
                case -1:
                    guardingCells[d] = endGuard[d];
                    break;
                    // receive from positiv side to negativ (origin) guarding cells
                case 1:
                    guardingCells[d] = originGuard[d];
                    break;
                case 0:
                    guardingCells[d] = coreBorderSize[d];
                    break;
                };
            }

            fieldTmp->addExchangeBuffer(i, guardingCells, m_commTagScatter);

            if(fieldTmpRecv)
            {
                /* guarding cells depend on direction
                 * for negative direction use originGuard else endGuard (relative direction ZERO is ignored)
                 * don't switch end and origin because this is a read buffer and not send buffer
                 */
                for(uint32_t d = 0; d < simDim; ++d)
                    guardingCells[d] = (relativMask[d] == -1 ? originGuard[d] : endGuard[d]);
                fieldTmpRecv->addExchange(GUARD, i, guardingCells, m_commTagGather);
            }
        }
    }

    SimulationDataId FieldTmp::getUniqueId(uint32_t slotId)
    {
        return getName() + std::to_string(slotId);
    }

    SimulationDataId FieldTmp::getUniqueId()
    {
        return getUniqueId(m_slotId);
    }

    void FieldTmp::synchronize()
    {
        fieldTmp->deviceToHost();
    }

    void FieldTmp::syncToDevice()
    {
        fieldTmp->hostToDevice();
    }

    EventTask FieldTmp::asyncCommunication(EventTask serialEvent)
    {
        EventTask ret;
        eventSystem::startTransaction(serialEvent + m_gatherEv + m_scatterEv);
        FieldFactory::getInstance().createTaskFieldReceiveAndInsert(*this);
        ret = eventSystem::endTransaction();

        eventSystem::startTransaction(serialEvent + m_gatherEv + m_scatterEv);
        FieldFactory::getInstance().createTaskFieldSend(*this);
        ret += eventSystem::endTransaction();
        m_scatterEv = ret;
        return ret;
    }

    EventTask FieldTmp::asyncCommunicationGather(EventTask serialEvent)
    {
        PMACC_VERIFY_MSG(
            fieldTmpSupportGatherCommunication == true,
            "fieldTmpSupportGatherCommunication in memory.param must be set to true");

        if(fieldTmpRecv != nullptr)
            m_gatherEv = fieldTmpRecv->asyncCommunication(serialEvent + m_scatterEv + m_gatherEv);
        return m_gatherEv;
    }

    void FieldTmp::bashField(uint32_t exchangeType)
    {
        pmacc::fields::operations::CopyGuardToExchange{}(*fieldTmp, SuperCellSize{}, exchangeType);
    }

    void FieldTmp::insertField(uint32_t exchangeType)
    {
        pmacc::fields::operations::AddExchangeToBorder{}(*fieldTmp, SuperCellSize{}, exchangeType);
    }

    FieldTmp::DataBoxType FieldTmp::getDeviceDataBox()
    {
        return fieldTmp->getDeviceBuffer().getDataBox();
    }

    FieldTmp::DataBoxType FieldTmp::getHostDataBox()
    {
        return fieldTmp->getHostBuffer().getDataBox();
    }

    GridBuffer<typename FieldTmp::ValueType, simDim>& FieldTmp::getGridBuffer()
    {
        return *fieldTmp;
    }

    GridLayout<simDim> FieldTmp::getGridLayout()
    {
        return cellDescription.getGridLayout();
    }

    void FieldTmp::reset(uint32_t)
    {
        fieldTmp->getHostBuffer().reset(true);
        fieldTmp->getDeviceBuffer().reset(false);
    }

    std::string FieldTmp::getName()
    {
        return "FieldTmp";
    }

} // namespace picongpu
