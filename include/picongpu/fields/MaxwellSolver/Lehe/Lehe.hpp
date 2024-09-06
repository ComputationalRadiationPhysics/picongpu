/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Remi Lehe,
 *                     Sergei Bastrakov
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

#include "picongpu/fields/MaxwellSolver/CFLChecker.hpp"
#include "picongpu/fields/MaxwellSolver/DispersionRelation.hpp"
#include "picongpu/fields/MaxwellSolver/Lehe/Derivative.hpp"
#include "picongpu/fields/MaxwellSolver/Lehe/Lehe.def"

#include <pmacc/algorithms/math/defines/pi.hpp>
#include <pmacc/traits/GetStringProperties.hpp>

#include <cstdint>


namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            /** Specialization of the CFL condition checker for Lehe solver
             *
             * @tparam T_CherenkovFreeDir the direction (axis) which should be free of cherenkov radiation
             *                            0 = x, 1 = y, 2 = z
             * @tparam T_Defer technical parameter to defer evaluation
             */
            template<uint32_t T_cherenkovFreeDir, typename T_Defer>
            struct CFLChecker<Lehe<T_cherenkovFreeDir>, T_Defer>
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
                    constexpr auto stepFreeDirection = step[T_cherenkovFreeDir];

                    // Dependance on T_Defer is required, otherwise this check would have been enforced for each setup
                    constexpr auto dt = getTimeStep();
                    PMACC_CASSERT_MSG(
                        Courant_Friedrichs_Lewy_condition_failure____check_your_grid_param_file,
                        (SPEED_OF_LIGHT * dt) <= stepFreeDirection && sizeof(T_Defer*) != 0);

                    return stepFreeDirection;
                }
            };

            /** Specialization of the dispersion relation for Lehe solver
             *
             * @tparam T_CherenkovFreeDir the direction (axis) which should be free of cherenkov radiation
             *                            0 = x, 1 = y, 2 = z
             */
            template<uint32_t T_cherenkovFreeDir>
            class DispersionRelation<Lehe<T_cherenkovFreeDir>> : public DispersionRelationBase
            {
            private:
                /** Generalized directions: Cherenkov-free along dir0, dir1 and dir2 are the two other directions
                 *
                 * @{
                 */

                static constexpr uint32_t dir0 = T_cherenkovFreeDir;
                static constexpr uint32_t dir1 = (dir0 + 1) % 3;
                static constexpr uint32_t dir2 = (dir0 + 2) % 3;

                /** }@ */

                //! Component-wise squared grid steps
                float3_64 const stepSquared = step * step;

                //! Inverse of Courant factor for the Cherenkov-free direction
                float_64 const stepRatio = step[dir0] / static_cast<float_64>(SPEED_OF_LIGHT * timeStep);

                //! Helper to calculate delta
                float_64 const coeff = stepRatio
                    * math::sin(pmacc::math::Pi<float_64>::halfValue * static_cast<float_64>(SPEED_OF_LIGHT * timeStep)
                                / step[dir0]);

                /** delta_x0 from eq. (10) in Lehe et al., generalized for any direction
                 *
                 * We use delta_x = delta_x0 and the other two deltas = 0 following eq. (11).
                 */
                float_64 const delta = 0.25 * (1.0 - coeff * coeff);

                //! Helper to calculate betaDir1 uniformly in 2d and 3d
                float_64 const stepRatio1 = dir1 < simDim ? step[dir0] / step[dir1] : 0.0;

                //! Helper to calculate betaDir2 uniformly in 2d and 3d
                float_64 const stepRatio2 = dir2 < simDim ? step[dir0] / step[dir2] : 0.0;

                //! beta_yx = beta_zx from eq. (11), generalized for any direction
                float_64 const betaDir0 = 0.125;

                //! beta_xy from eq. (11), generalized for any direction
                float_64 const betaDir1 = 0.125 * stepRatio1 * stepRatio1;

                //! beta_xz from eq. (11), generalized for any direction
                float_64 const betaDir2 = 0.125 * stepRatio2 * stepRatio2;

            public:
                /** Create a functor with the given parameters
                 *
                 * @param omega angular frequency = 2pi * c / lambda
                 * @param direction normalized propagation direction
                 */
                DispersionRelation(float_64 const omega, float3_64 const direction)
                    : DispersionRelationBase(omega, direction)
                {
                }

                /** Calculate f(absK) in the dispersion relation, see comment to the main template
                 *
                 * @param absK absolute value of the (angular) wave number
                 */
                float_64 relation(float_64 const absK) const
                {
                    /* Eq. (8) in Lehe et al., using coefficients from eq. (11).
                     * Generalized for arbitrary T_cherenkovFreeDir and expressed as rhs - lhs = 0
                     */
                    auto sSquared = sSquaredValues(absK);
                    auto rhs = 0.0;
                    for(uint32_t d = 0; d < simDim; d++)
                        rhs += sSquared[d] / stepSquared[d];
                    rhs -= 4.0 * delta * sSquared[dir0] * sSquared[dir0] / stepSquared[dir0];
                    rhs -= 4.0 * (betaDir1 / stepSquared[dir1] + betaDir0 / stepSquared[dir0]) * sSquared[dir0]
                        * sSquared[dir1];
                    rhs -= 4.0 * (betaDir2 / stepSquared[dir2] + betaDir0 / stepSquared[dir0]) * sSquared[dir0]
                        * sSquared[dir2];
                    auto const lhsTerm = math::sin(0.5 * omega * timeStep) / (SPEED_OF_LIGHT * timeStep);
                    auto const lhs = lhsTerm * lhsTerm;
                    return rhs - lhs;
                }

                /** Calculate df(absK)/d(absK) in the dispersion relation, see comment to the main template
                 *
                 * @param absK absolute value of the (angular) wave number
                 */
                float_64 relationDerivative(float_64 const absK) const
                {
                    /* Due to the structure of the relation, it is more convenient to operate with variables
                     * sSquared(absK) from function relation() rather than s. So e.g. for calculating d(s^4)/d(absK) we
                     * use d(sSquared^2)/d(absK). Precalculate d(sSquared(absK))/d(absK) for x, y, z components.
                     */
                    auto sSquaredDerivative = float3_64::create(0.0);
                    for(uint32_t d = 0; d < simDim; d++)
                    {
                        auto const arg = 0.5 * absK * direction[d] * step[d];
                        auto const s = math::sin(arg);
                        auto const sDerivative = 0.5 * direction[d] * step[d] * math::cos(arg);
                        sSquaredDerivative[d] = 2.0 * s * sDerivative;
                    }

                    // Term-wise derivative in same order as in relation()
                    auto sSquared = sSquaredValues(absK);
                    auto result = 0.0;
                    for(uint32_t d = 0; d < simDim; d++)
                        result += sSquaredDerivative[d] / stepSquared[d];
                    result -= 4.0 * delta * sSquared[dir0] * sSquaredDerivative[dir0] / stepSquared[dir0];
                    result -= 4.0 * (betaDir1 / stepSquared[dir1] + betaDir0 / stepSquared[dir0])
                        * (sSquared[dir0] * sSquaredDerivative[dir1] + sSquaredDerivative[dir0] * sSquared[dir1]);
                    result -= 4.0 * (betaDir1 / stepSquared[dir1] + betaDir0 / stepSquared[dir0])
                        * (sSquared[dir0] * sSquaredDerivative[dir2] + sSquaredDerivative[dir0] * sSquared[dir2]);
                    return result;
                }

            private:
                /** Get vector [s_x^2, s_y^2, s_z^2] described under eq. (8) in Lehe et al.
                 *
                 * @param absK absolute value of the (angular) wave number
                 */
                float3_64 sSquaredValues(float_64 const absK) const
                {
                    auto result = float3_64::create(0.0);
                    for(uint32_t d = 0; d < simDim; d++)
                    {
                        auto const arg = 0.5 * absK * direction[d] * step[d];
                        auto const s = math::sin(arg);
                        result[d] = s * s;
                    }
                    return result;
                }
            };
        } // namespace maxwellSolver
    } // namespace fields
} // namespace picongpu

namespace pmacc
{
    namespace traits
    {
        template<uint32_t T_cherenkovFreeDir>
        struct StringProperties<::picongpu::fields::maxwellSolver::Lehe<T_cherenkovFreeDir>>
        {
            static StringProperty get()
            {
                auto propList = ::picongpu::fields::maxwellSolver::Lehe<T_cherenkovFreeDir>::getStringProperties();
                // overwrite the name of the solver (inherit all other properties)
                propList["name"].value = "Lehe";
                return propList;
            }
        };

    } // namespace traits
} // namespace pmacc
