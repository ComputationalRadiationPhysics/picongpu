/* Copyright 2022 Pawel Ordyna
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

#include "picongpu/particles/collision/kernels.def"
#include "picongpu/particles/collision/relativistic/RelativisticCollision.hpp"
#include "picongpu/particles/collision/relativistic/RelativisticCollisionDynamicLog.def"

#include <cmath>
namespace picongpu
{
    namespace particles
    {
        namespace collision
        {
            namespace relativistic
            {
                namespace acc
                {
                    //! Coulomb log functor for dynamic log calculation
                    struct DynamicLog
                    {
                        DINLINE float_COLL operator()(Variables const& v) const
                        {
                            // [Perez2012] formula (22):
                            const float_COLL factor1 = math::abs((v.charge0 * v.charge1))
                                / (4._COLL * pmacc::math::Pi<float_COLL>::value
                                   * static_cast<float_COLL>(EPS0 * SPEED_OF_LIGHT * SPEED_OF_LIGHT));
                            const float_COLL factor2 = v.gammaComs / (v.mass0 * v.gamma0 + v.mass1 * v.gamma1);
                            const float_COLL factor3
                                = (v.coeff0 * v.coeff1 / v.comsMomentum0Norm
                                       * static_cast<float_COLL>(SPEED_OF_LIGHT * SPEED_OF_LIGHT)
                                   + 1._COLL);
                            /*
                             * factor3 is not squared following a note in smilei documentation. According to smilei
                             * developers the square in the original formula was a typo. the 1/weight factor is needed
                             * because we are mixing charge and mass in one formula an these are not canceling it each
                             * other out correctly.
                             */
                            const float_COLL twoRadImpactParam
                                = factor1 * factor2 * factor3 / precision::WEIGHT_NORM_COLL;

                            // formula in line above eq. (22) in [Perez2012]:
                            const float_COLL minImpactParam = math::max(
                                static_cast<float_COLL>(HBAR) * pmacc::math::Pi<float_COLL>::doubleValue
                                    / (2._COLL * math::sqrt(v.comsMomentum0Norm) / precision::WEIGHT_NORM_COLL),
                                twoRadImpactParam);

                            // eq. (23) in [Perez2012]:
                            const float_COLL coulombLog = 0.5_COLL
                                * math::log(1.0_COLL
                                            + static_cast<float_COLL>(screeningLengthSquared_m)
                                                / (minImpactParam * minImpactParam));
                            return math::max(2._COLL, coulombLog);
                        }
                        PMACC_ALIGN(screeningLengthSquared_m, float_X);
                    };

                } // namespace acc

                template<bool ifDebug>
                struct RelativisticCollisionDynamicLogImpl
                {
                    template<typename T_Species0, typename T_Species1>
                    struct apply
                    {
                        using type = RelativisticCollisionDynamicLogImpl<ifDebug>;
                    };

                    // set the kernel to provide the Debye length to the acc functor
                    using CallingInterKernel = InterCollision<true>;
                    using CallingIntraKernel = IntraCollision<true>;

                    static constexpr bool ifDebug_m = ifDebug;
                    HINLINE RelativisticCollisionDynamicLogImpl(uint32_t currentStep){};

                    using AccFunctorImpl = acc::RelativisticCollision<acc::DynamicLog, ifDebug>;
                    using AccFunctor = collision::acc::IBinary<AccFunctorImpl>;

                    /** create device manipulator functor
                     *
                     * @param worker lockstep worker
                     * @param offset (in supercells, without any guards) to the origin of the local domain
                     * @param density0 cell density of the 1st species
                     * @param density1 cell density of the 2nd species
                     * @param potentialPartners number of potential collision partners for a macro particle in
                     *   the cell.
                     * @param coulombLog Coulomb logarithm
                     */
                    template<typename T_Worker>
                    HDINLINE auto operator()(
                        T_Worker const& worker,
                        DataSpace<simDim> const& offset,
                        float_X const& density0,
                        float_X const& density1,
                        uint32_t const& potentialPartners) const
                    {
                        using namespace picongpu::particles::collision::precision;
                        return AccFunctor{AccFunctorImpl{
                            math::pow(precisionCast<float_COLL>(density0), 2.0_COLL / 3.0_COLL),
                            math::pow(precisionCast<float_COLL>(density1), 2.0_COLL / 3.0_COLL),
                            potentialPartners}};
                    }

                    //! get the name of the functor
                    static HINLINE std::string getName()
                    {
                        return "RelativisticCollisionDynamicLog";
                    }
                };
            } // namespace relativistic
        } // namespace collision
    } // namespace particles
} // namespace picongpu
