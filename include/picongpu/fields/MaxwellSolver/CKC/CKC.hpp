/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera, Remi Lehe, Sergei Bastrakov, Lennert Sprenger
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/fields/MaxwellSolver/CFLChecker.hpp"
#include "picongpu/fields/MaxwellSolver/CKC/CKC.def"
#include "picongpu/fields/MaxwellSolver/CKC/Derivative.hpp"
#include "picongpu/fields/MaxwellSolver/DispersionRelation.hpp"
#include "picongpu/fields/MaxwellSolver/FDTD/FDTD.hpp"

#include <pmacc/algorithms/math/defines/pi.hpp>
#include <pmacc/traits/GetStringProperties.hpp>

namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            /** Specialization of the CFL condition checker for CKC solver
             *
             * @tparam T_Defer technical parameter to defer evaluation
             */
            template<typename T_Defer>
            struct CFLChecker<CKC, T_Defer>
            {
                /** Check the CFL condition according to the paper, doesn't compile when failed
                 *
                 * @return upper bound on `c * dt` due to chosen cell size according to CFL condition
                 */
                float_X operator()() const
                {
                    // cellSize is not constexpr currently, so make an own constexpr array
                    constexpr float_X step[3]
                        = {sim.pic.getCellSize().x(), sim.pic.getCellSize().y(), sim.pic.getCellSize().z()};
                    constexpr float_X cdt = sim.pic.getSpeedOfLight() * getTimeStep(); // c * dt

                    constexpr float_64 delta = std::min({step[0], step[1], step[2]});

                    // Dependence on T_Defer is required, otherwise this check would have been enforced for each setup
                    PMACC_CASSERT_MSG(
                        Courant_Friedrichs_Lewy_condition_failure____check_your_simulation_param_file,
                        (cdt <= delta) && sizeof(T_Defer*) != 0);

                    return delta;
                }
            };

        } // namespace maxwellSolver
    } // namespace fields
} // namespace picongpu

namespace pmacc
{
    namespace traits
    {
        template<>
        struct StringProperties<::picongpu::fields::maxwellSolver::CKC>
        {
            static StringProperty get()
            {
                auto propList = ::picongpu::fields::maxwellSolver::CKC::getStringProperties();
                // overwrite the name of the solver (inherit all other properties)
                propList["name"].value = "CK";
                return propList;
            }
        };

    } // namespace traits
} // namespace pmacc
