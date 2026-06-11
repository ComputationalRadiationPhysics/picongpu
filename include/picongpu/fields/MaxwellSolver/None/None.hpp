/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/fields/MaxwellSolver/CFLChecker.hpp"
#include "picongpu/fields/MaxwellSolver/None/None.def"
#include "picongpu/fields/YeeCell.hpp"
#include "picongpu/traits/GetMargin.hpp"

#include <pmacc/dataManagement/ISimulationData.hpp>
#include <pmacc/types.hpp>

#include <cstdint>
#include <limits>

namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            class None : public ISimulationData
            {
            private:
                using SuperCellSize = MappingDesc::SuperCellSize;

            public:
                using CellType = fields::YeeCell;

                None(MappingDesc)
                {
                }

                void update_beforeCurrent(uint32_t)
                {
                }

                template<uint32_t T_area>
                void addCurrent()
                {
                }

                void update_afterCurrent(uint32_t)
                {
                }

                static pmacc::traits::StringProperty getStringProperties()
                {
                    pmacc::traits::StringProperty propList("name", "none");
                    return propList;
                }

                static std::string getName()
                {
                    return "FieldSolverNone";
                }

                /**
                 * Synchronizes simulation data, meaning accessing (host side) data
                 * will return up-to-date values.
                 */
                void synchronize() override {};

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

            /** Specialization of the CFL condition checker for the None solver
             *
             * @tparam T_Defer technical parameter to defer evaluation
             */
            template<typename T_Defer>
            struct CFLChecker<None, T_Defer>
            {
                /** No limitations for this solver, allow any dt
                 *
                 * @return upper bound on `c * dt` due to chosen cell size according to CFL condition
                 */
                float_X operator()() const
                {
                    return std::numeric_limits<float_X>::infinity();
                }
            };

        } // namespace maxwellSolver
    } // namespace fields

    namespace traits
    {
        /** Get margin for any field access in the None solver
         *
         * @tparam T_Field field type
         */
        template<typename T_Field>
        struct GetMargin<picongpu::fields::maxwellSolver::None, T_Field>
        {
            using LowerMargin = typename pmacc::math::CT::make_Int<simDim, 0>::type;
            using UpperMargin = LowerMargin;
        };
    } // namespace traits

} // namespace picongpu
