/* Copyright 2019-2023 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz,
 *                     Sergei Bastrakov, Klaus Steiniger
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/FieldJ.hpp"
#include "picongpu/fields/MaxwellSolver/FDTD/FDTD.def"
#include "picongpu/fields/MaxwellSolver/FDTD/FDTDBase.hpp"
#include "picongpu/fields/YeeCell.hpp"
#include "picongpu/traits/GetMargin.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/dataManagement/ISimulationData.hpp>
#include <pmacc/traits/GetStringProperties.hpp>

#include <cstdint>

namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            template<typename T_CurlE, typename T_CurlB>
            class FDTD
                : public fdtd::FDTDBase<T_CurlE, T_CurlB>
                , public ISimulationData
            {
            public:
                //! Base type
                using Base = fdtd::FDTDBase<T_CurlE, T_CurlB>;

                //! Cell type
                using CellType = fields::YeeCell;

                /** Create FDTD solver instance
                 *
                 * @param cellDescription mapping description for kernels
                 */
                FDTD(MappingDesc const cellDescription) : Base(cellDescription)
                {
                }

                /** Perform the first part of E and B propagation by a PIC time step.
                 *
                 * Does not account for the J term, which will be added by addCurrent().
                 * Together with addCurrent() and update_afterCurrent() forms the full propagation by a PIC time step.
                 *
                 * @param currentStep index of the current time iteration
                 */
                void update_beforeCurrent(uint32_t const currentStep)
                {
                    this->updateBeforeCurrent(static_cast<float_X>(currentStep));
                }

                /** Add contribution of FieldJ in the given area according to Ampere's law
                 *
                 * @tparam T_area area to operate on
                 */
                template<uint32_t T_area>
                void addCurrent()
                {
                    DataConnector& dc = Environment<>::get().DataConnector();
                    auto& fieldJ = *dc.get<FieldJ>(FieldJ::getName());
                    // Coefficient in front of J in Ampere's law
                    constexpr float_X coeff = -(1.0_X / EPS0) * sim.pic.getDt();
                    this->template addCurrentImpl<T_area>(fieldJ.getDeviceDataBox(), coeff);
                }

                /** Perform the last part of E and B propagation by a PIC time step
                 *
                 * Does not account for the J term, which has been added by addCurrent().
                 * Together with addCurrent() and update_beforeCurrent() forms the full propagation by a PIC time step.
                 *
                 * @param currentStep index of the current time iteration
                 */
                void update_afterCurrent(uint32_t const currentStep)
                {
                    this->updateAfterCurrent(static_cast<float_X>(currentStep));
                }

                //! Get string properties
                static pmacc::traits::StringProperty getStringProperties()
                {
                    pmacc::traits::StringProperty propList("name", "FDTD");
                    return propList;
                }

                /** Name of the solver which can be used to share this class via DataConnector */
                static std::string getName()
                {
                    return "FieldSolverFDTD";
                }

                /**
                 * Synchronizes simulation data, meaning accessing (host side) data
                 * will return up-to-date values.
                 */
                void synchronize() override{};

                /**
                 * Return the globally unique identifier for this simulation data.
                 *
                 * @return globally unique identifier
                 */
                SimulationDataId getUniqueId() override
                {
                    return getName();
                }
            };

        } // namespace maxwellSolver
    } // namespace fields

    namespace traits
    {
        /** Get margin for B field access in the FDTD solver
         *
         * @tparam T_CurlE functor to compute curl of E
         * @tparam T_CurlB functor to compute curl of B
         */
        template<typename T_CurlE, typename T_CurlB>
        struct GetMargin<fields::maxwellSolver::FDTD<T_CurlE, T_CurlB>, FieldB>
        {
            using LowerMargin = typename T_CurlB::LowerMargin;
            using UpperMargin = typename T_CurlB::UpperMargin;
        };

        /** Get margin for E field access in the FDTD solver
         *
         * @tparam T_CurlE functor to compute curl of E
         * @tparam T_CurlB functor to compute curl of B
         */
        template<typename T_CurlE, typename T_CurlB>
        struct GetMargin<fields::maxwellSolver::FDTD<T_CurlE, T_CurlB>, FieldE>
        {
            using LowerMargin = typename T_CurlE::LowerMargin;
            using UpperMargin = typename T_CurlE::UpperMargin;
        };

        /** Get margin for both fields access in the FDTD solver
         *
         * @tparam T_CurlE functor to compute curl of E
         * @tparam T_CurlB functor to compute curl of B
         */
        template<typename T_CurlE, typename T_CurlB>
        struct GetMargin<fields::maxwellSolver::FDTD<T_CurlE, T_CurlB>>
        {
        private:
            using Solver = fields::maxwellSolver::FDTD<T_CurlE, T_CurlB>;

        public:
            using LowerMargin = typename pmacc::math::CT::max<
                typename GetLowerMargin<Solver, FieldB>::type,
                typename GetLowerMargin<Solver, FieldE>::type>::type;
            using UpperMargin = typename pmacc::math::CT::max<
                typename GetUpperMargin<Solver, FieldB>::type,
                typename GetUpperMargin<Solver, FieldE>::type>::type;
        };


    } // namespace traits
} // namespace picongpu
