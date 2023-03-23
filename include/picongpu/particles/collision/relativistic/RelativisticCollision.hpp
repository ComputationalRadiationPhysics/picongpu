/* Copyright 2015-2022 Rene Widera, Pawel Ordyna
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

#include <pmacc/random/distributions/Uniform.hpp>

#include <cmath>
#include <cstdio>
#include <type_traits>
#include <utility>

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
                    using namespace pmacc;
                    using namespace picongpu::particles::collision::precision;
                    constexpr float_COLL c = static_cast<float_COLL>(SPEED_OF_LIGHT);


                    /* Calculate @f[ \gamma^* m @f]
                     *
                     * Returns particle mass times its Lorentz factor in the COM frame.
                     *
                     * @param labMomentum particle momentum in the labFrame
                     * @param mass particle mass
                     * @param gamma
                     * @param gammaComs Lorentz factor of the COM frame in the lab frame
                     * @param comsVelocity COM system velocity in the lab frame
                     */
                    DINLINE float_COLL coeff(
                        float3_COLL labMomentum,
                        float_COLL mass,
                        float_COLL gamma,
                        float_COLL gammaComs,
                        float3_COLL comsVelocity)
                    {
                        float3_COLL labVelocity = labMomentum / gamma / mass;
                        float_COLL dot = pmacc::math::dot(comsVelocity, labVelocity);

                        float_COLL val = gammaComs * gamma - dot * (gammaComs * gamma / (c * c));
                        val *= mass;
                        return val;
                    }

                    /* Convert momentum from the lab frame into the COM frame.
                     *
                     * @param labMomentum momentum in the lab frame
                     * @param mass particle mass
                     * @param gamma particle Lorentz factor in the lab frame
                     * @param gammaComs @f[ \gamma_C @f] Lorentz factor of the COM frame in the lab frame
                     * @param factorA @f[ \frac{\gamma_C - 1}{v_C^2} @f]
                     * @param comsVelocity @f[ v_C @f] COM system velocity in the lab frame
                     */
                    DINLINE float3_COLL labToComs(
                        float3_COLL labMomentum,
                        float_COLL mass,
                        float_COLL gamma,
                        float_COLL gammaComs,
                        float_COLL factorA,
                        float3_COLL comsVelocity)
                    {
                        float3_COLL labVelocity = labMomentum / gamma / mass;
                        float_COLL dot = pmacc::math::dot(comsVelocity, labVelocity);
                        float_COLL factor = (factorA * dot - gammaComs);
                        factor *= mass * gamma;
                        float3_COLL diff = factor * comsVelocity;
                        return labMomentum + diff;
                    }

                    /* Calculate relative velocity in the COM system
                     *
                     * @param comsMementumMag0 1st particle momentum (in the COM system) magnitude
                     * @param mass0 1st particle mass
                     * @param mass1 2nd particle mass
                     * @param gamma0 1st particle Lorentz factor
                     * @param gamma1 2nd particle Lorentz factor
                     * @param gammaComs Lorentz factor of the COM frame in the lab frame
                     */
                    DINLINE float_COLL calcRelativeComsVelocity(
                        float_COLL comsMomentumMag0,
                        float_COLL mass0,
                        float_COLL mass1,
                        float_COLL gamma0,
                        float_COLL gamma1,
                        float_COLL coeff0,
                        float_COLL coeff1,
                        float_COLL gammaComs)
                    {
                        float_COLL val = (mass0 * gamma0 + mass1 * gamma1) * comsMomentumMag0;
                        val = val / (coeff0 * coeff1 * gammaComs); // TODO: null division?
                        // TODO:
                        // this is actually not the relative velocity! since we are missing (1 + v1v2/c^2).
                        return val;
                    }

                    /* Convert momentum from the COM frame into the lab frame.
                     *
                     * @param labMomentum momentum in the COM frame
                     * @param mass particle mass
                     * @param gamma particle Lorentz factor in the lab frame
                     * @param gammaComs @f[ \gamma_C @f] Lorentz factor of the COM frame in the lab frame
                     * @param factorA @f[ \frac{\gamma_C - 1}{v_C^2} @f]
                     * @param comsVelocity @f[ v_C @f] COM system velocity in the lab frame
                     */
                    DINLINE float3_COLL comsToLab(
                        float3_COLL comsMomentum,
                        float_COLL mass,
                        float_COLL coeff,
                        float_COLL gammaComs,
                        float_COLL factorA,
                        float3_COLL comsVelocity)
                    {
                        // (13) in [Perez 2012]
                        float_COLL dot = pmacc::math::dot(comsVelocity, comsMomentum);
                        float_COLL factor = (factorA * dot + coeff * gammaComs);
                        float3_COLL diff = factor * comsVelocity;

                        return comsMomentum + diff;
                    }

                    /* Calculate the cosine of the scattering angle.
                     *
                     * The probability distribution for the cosine depends on @f[ s_{12} @f]. The returned vale
                     * is determined by a float value between 0 and 1.
                     *
                     * @param s12 @f[ s_{12} @f] parameter. See [Perez 2012]. It should be >= 0.
                     * @param u a random generated float between 0 and 1
                     */
                    DINLINE float_COLL calcCosXi(float_COLL const s12, float_COLL const u)
                    {
                        // new fit from smilei implementation:
                        if(s12 < 4._COLL)
                        {
                            const float_COLL s2 = s12 * s12;
                            const float_COLL alpha = 0.37_COLL * s12 - 0.005_COLL * s2 - 0.0064_COLL * s2 * s12;
                            const float_COLL sin2X2 = alpha * u / math::sqrt((1._COLL - u) + alpha * alpha * u);
                            return 1._COLL - 2.0_COLL * sin2X2;
                        }
                        else
                            return 2._COLL * u - 1._COLL;
                    }


                    /* Calculate the momentum after the collision in the COM frame
                     *
                     * @param p momentum in the COM frame
                     * @param cosXi cosine of the scattering angle
                     * @param phi azimuthal scattering angle from [0, 2pi]
                     */
                    DINLINE float3_COLL
                    calcFinalComsMomentum(float3_COLL const p, float_COLL const cosXi, float_COLL const phi)
                    {
                        float_COLL sinPhi, cosPhi;
                        pmacc::math::sincos(phi, sinPhi, cosPhi);
                        float_COLL sinXi = math::sqrt(1.0_COLL - cosXi * cosXi);

                        // (12) in [Perez 2012]
                        float3_COLL finalVec;
                        float_COLL const pNorm2 = math::sqrt(pmacc::math::l2norm2(p));
                        float_COLL const pPerp = math::sqrt(p.x() * p.x() + p.y() * p.y());
                        // TODO chose a better limit?
                        // limit px->0 py=0. this also covers the pPerp = pAbs = 0 case. An alternative would
                        // be to let the momentum unchanged in that case.
                        if(pPerp <= math::max(std::numeric_limits<float_COLL>::epsilon(), 1.0e-10_COLL) * pNorm2)
                        {
                            finalVec[0] = pNorm2 * sinXi * cosPhi;
                            finalVec[1] = pNorm2 * sinXi * sinPhi;
                            finalVec[2] = pNorm2 * cosXi;
                        }
                        else // normal case
                        {
                            finalVec[0] = (p.x() * p.z() * sinXi * cosPhi - p.y() * pNorm2 * sinXi * sinPhi) / pPerp
                                + p.x() * cosXi;
                            finalVec[1] = (p.y() * p.z() * sinXi * cosPhi + p.x() * pNorm2 * sinXi * sinPhi) / pPerp
                                + p.y() * cosXi;
                            finalVec[2] = -1.0_COLL * pPerp * sinXi * cosPhi + p.z() * cosXi;
                        }
                        return finalVec;
                    }

                    //! Stores some precalculated values used in the collision algorithm
                    struct Variables
                    {
                        PMACC_ALIGN(normalizedWeight0, float_COLL);
                        PMACC_ALIGN(normalizedWeight1, float_COLL);
                        PMACC_ALIGN(labMomentum0, float3_COLL);
                        PMACC_ALIGN(labMomentum1, float3_COLL);
                        PMACC_ALIGN(mass0, float_COLL);
                        PMACC_ALIGN(mass1, float_COLL);
                        PMACC_ALIGN(charge0, float_COLL);
                        PMACC_ALIGN(charge1, float_COLL);
                        PMACC_ALIGN(gamma0, float_COLL);
                        PMACC_ALIGN(gamma1, float_COLL);
                        PMACC_ALIGN(comsVelocity, float3_COLL);

                        PMACC_ALIGN(comsMomentum0, float3_COLL);
                        PMACC_ALIGN(comsMomentum0Norm2, float_COLL);
                        PMACC_ALIGN(gammaComs, float_COLL);
                        PMACC_ALIGN(factorA, float_COLL);
                        PMACC_ALIGN(coeff0, float_COLL);
                        PMACC_ALIGN(coeff1, float_COLL);

                        template<typename T_Par0, typename T_Par1>
                        DINLINE Variables(T_Par0 const& par0, T_Par1 const& par1)
                            : normalizedWeight0(precisionCast<float_COLL>(par0[weighting_]) / WEIGHT_NORM_COLL)
                            , normalizedWeight1(precisionCast<float_COLL>(par1[weighting_]) / WEIGHT_NORM_COLL)
                            , labMomentum0(precisionCast<float_COLL>(par0[momentum_]) / normalizedWeight0)
                            , labMomentum1(precisionCast<float_COLL>(par1[momentum_]) / normalizedWeight1)
                            , mass0(precisionCast<float_COLL>(
                                  picongpu::traits::attribute::getMass(WEIGHT_NORM_COLL, par0)))
                            , mass1(precisionCast<float_COLL>(
                                  picongpu::traits::attribute::getMass(WEIGHT_NORM_COLL, par1)))
                            , charge0(precisionCast<float_COLL>(
                                  picongpu::traits::attribute::getCharge(WEIGHT_NORM_COLL, par0)))
                            , charge1(precisionCast<float_COLL>(
                                  picongpu::traits::attribute::getCharge(WEIGHT_NORM_COLL, par1)))
                            , gamma0(picongpu::gamma<float_COLL>(labMomentum0, mass0))
                            , gamma1(picongpu::gamma<float_COLL>(labMomentum1, mass1))
                            , comsVelocity((labMomentum0 + labMomentum1) / (mass0 * gamma0 + mass1 * gamma1))
                        {
                            float_COLL const comsVelocityNorm2 = pmacc::math::l2norm2(comsVelocity);

                            if(comsVelocityNorm2 != 0.0_COLL)
                            {
                                float_COLL const comsVelocityAbs = math::sqrt(comsVelocityNorm2);
                                // written as (1-v)(1+v) rather than (1-v^2) for better performance when v close to
                                // c
                                gammaComs = 1.0_COLL
                                    / math::sqrt((1.0_COLL - comsVelocityAbs / c) * (1.0_COLL + comsVelocityAbs / c));
                                // used later for comsToLab:
                                factorA = (gammaComs - 1.0_COLL) / comsVelocityNorm2;

                                // Stared gamma times mass, from [Perez 2012].
                                coeff0 = coeff(labMomentum0, mass0, gamma0, gammaComs, comsVelocity);
                                // gamma^* . mass
                                coeff1 = coeff(labMomentum1, mass1, gamma1, gammaComs, comsVelocity);
                                // (2) in [Perez 2012]
                                comsMomentum0
                                    = labToComs(labMomentum0, mass0, gamma0, gammaComs, factorA, comsVelocity);
                            }
                            else
                            {
                                // Lab frame is the same as the COMS frame
                                gammaComs = 1.0_COLL;
                                // used later for comsToLab:
                                // lim v_coms-->0 for (gamma_coms -1 / v_coms^2) is 1/(2c^2)
                                factorA = 1.0_COLL / (2.0_COLL * c * c);
                                // Stared gamma times mass, from [Perez 2012].
                                coeff0 = mass0 * gamma0;
                                // gamma^* . mass
                                coeff1 = mass1 * gamma1;
                                comsMomentum0 = labMomentum0;
                            }
                            comsMomentum0Norm2 = pmacc::math::l2norm2(comsMomentum0);
                        }
                    };

                    //! Base class for relativistic collision that is used to extend the algorithm with debug features
                    template<bool ifDebug>
                    struct RelativisticCollisionBase
                    {
                        DINLINE void processDebugValues(
                            float_COLL const& sumCoulombLog_p,
                            float_COLL const& sumSParam_p)
                        {
                        }
                    };

                    template<>
                    struct RelativisticCollisionBase<true>
                    {
                        PMACC_ALIGN(sumCoulombLog = 0._COLL, float_COLL);
                        PMACC_ALIGN(sumSParam = 0._COLL, float_COLL);
                        PMACC_ALIGN(timesUsed = 0u, uint32_t);

                        DINLINE void processDebugValues(
                            float_COLL const& sumCoulombLog_p,
                            float_COLL const& sumSParam_p)
                        {
                            sumCoulombLog += sumCoulombLog_p;
                            sumSParam += sumSParam_p;
                            timesUsed++;
                        }
                    };


                    /* Perform a single binary collision between two macro particles. (Device side functor)
                     *
                     * This algorithm was described in [Perez 2012] @url www.doi.org/10.1063/1.4742167.
                     * And it incorporates changes suggested in [Higginson 2020]
                     * @url www.doi.org/10.1016/j.jcp.2020.109450
                     */
                    template<typename T_CoulombLogFunctor, bool ifDebug>
                    struct RelativisticCollision : public RelativisticCollisionBase<ifDebug>
                    {
                        /* Initialize device side functor.
                         *
                         * @param p_densitySqCbrt0 @f[ n_0^{2/3} @f] where @f[ n_0 @f] is the 1st species density.
                         * @param p_densitySqCbrt1 @f[ n_1^{2/3} @f] where @f[ n_1 @f] is the 2nd species density.
                         * @param p_potentialPartners number of potential collision partners for a macro particle in
                         *   the cell.
                         * @param p_coulombLog coulomb logarithm
                         */
                        HDINLINE RelativisticCollision(
                            float_COLL p_densitySqCbrt0,
                            float_COLL p_densitySqCbrt1,
                            uint32_t p_potentialPartners)
                            : densitySqCbrt0(p_densitySqCbrt0)
                            , densitySqCbrt1(p_densitySqCbrt1)
                            , duplicationCorrection(1u)
                            , potentialPartners(p_potentialPartners){};

                        PMACC_ALIGN(coulombLogFunctor, T_CoulombLogFunctor);
                        PMACC_ALIGN(densitySqCbrt0, float_COLL);
                        PMACC_ALIGN(densitySqCbrt1, float_COLL);
                        PMACC_ALIGN(duplicationCorrection, float_COLL);
                        PMACC_ALIGN(potentialPartners, uint32_t);

                    private:
                        //! Calculates the s parameter from the algorithm
                        DINLINE float_COLL normalizedPathLength(Variables const& v, float_COLL const& coulombLog) const
                        {
                            // const float_COLL coulombLog = 10._COLL;
                            //  f0 * f1 * f2^2
                            //  is equal  s12 * (n12/(n1*n2)) from [Perez 2012]
                            float_COLL s12Factor0
                                = (DELTA_T_COLL * coulombLog * v.charge0 * v.charge0 * v.charge1 * v.charge1)
                                / (4.0_COLL * pmacc::math::Pi<float_COLL>::value * EPS0_COLL * EPS0_COLL * c * c * c
                                   * c * v.mass0 * v.gamma0 * v.mass1 * v.gamma1);
                            s12Factor0 *= 1.0_COLL / WEIGHT_NORM_COLL / WEIGHT_NORM_COLL;
                            float_COLL const s12Factor1 = v.gammaComs * math::sqrt(v.comsMomentum0Norm2)
                                / (v.mass0 * v.gamma0 + v.mass1 * v.gamma1);
                            float_COLL const s12Factor2
                                = v.coeff0 * v.coeff1 * c * c / v.comsMomentum0Norm2 + 1.0_COLL;
                            // Statistical part from [Higginson 2020],
                            // corresponds to n1*n2/n12 in [Perez 2012]:
                            float_COLL const s12Factor3 = potentialPartners
                                * math::max(v.normalizedWeight0, v.normalizedWeight1) * WEIGHT_NORM_COLL
                                / static_cast<float_COLL>(duplicationCorrection) / CELL_VOLUME_COLL;
                            float_COLL const s12n = s12Factor0 * s12Factor1 * s12Factor2 * s12Factor2 * s12Factor3;

                            // low Temeprature correction:
                            // [Perez 2012] (8)
                            // TODO: should we check for the non-relativistic condition? Which gamma should we look at?
                            float_COLL relativeComsVelocity = calcRelativeComsVelocity(
                                math::sqrt(v.comsMomentum0Norm2),
                                v.mass0,
                                v.mass1,
                                v.gamma0,
                                v.gamma1,
                                v.coeff0,
                                v.coeff1,
                                v.gammaComs);
                            // [Perez 2012] (21) ( without n1*n2/n12 )
                            float_COLL s12Max = math::pow(
                                                    4.0_COLL * pmacc::math::Pi<float_COLL>::value / 3._COLL,
                                                    1.0_COLL / 3.0_COLL)
                                * DELTA_T_COLL * (v.mass0 + v.mass1)
                                / math::max(v.mass0 * densitySqCbrt0, v.mass1 * densitySqCbrt1) * relativeComsVelocity;
                            s12Max *= s12Factor3;
                            return math::min(s12n, s12Max);
                        }

                        //! sample scattering angles and apply the new momenta
                        template<typename T_Context, typename T_Par0, typename T_Par1>
                        DINLINE void lastPart(
                            Variables const& v,
                            T_Context const& ctx,
                            T_Par0& par0,
                            T_Par1& par1,
                            float_COLL const& s12) const
                        {
                            // Get a random float value from 0,1
                            auto const& worker = *ctx.m_worker;
                            auto& rngHandle = *ctx.m_hRng;
                            using UniformFloat = pmacc::random::distributions::Uniform<
                                pmacc::random::distributions::uniform::ExcludeZero<float_COLL>>;
                            auto rng = rngHandle.template applyDistribution<UniformFloat>();
                            float_COLL rngValue = rng(worker);

                            float_COLL const cosXi = calcCosXi(s12, rngValue);
                            float_COLL const phi = 2.0_COLL * PI * rng(worker);
                            float3_COLL const finalComs0 = calcFinalComsMomentum(v.comsMomentum0, cosXi, phi);

                            float3_COLL finalLab0, finalLab1;
                            if(v.normalizedWeight0 > v.normalizedWeight1)
                            {
                                finalLab1 = comsToLab(
                                    -1.0_COLL * finalComs0,
                                    v.mass1,
                                    v.coeff1,
                                    v.gammaComs,
                                    v.factorA,
                                    v.comsVelocity);


                                par1[momentum_] = precisionCast<float_X>(finalLab1 * v.normalizedWeight1);
                                if((v.normalizedWeight1 / v.normalizedWeight0) - rng(worker) > 0)
                                {
                                    finalLab0 = comsToLab(
                                        finalComs0,
                                        v.mass0,
                                        v.coeff0,
                                        v.gammaComs,
                                        v.factorA,
                                        v.comsVelocity);
                                    par0[momentum_] = precisionCast<float_X>(finalLab0 * v.normalizedWeight0);
                                }
                            }
                            else
                            {
                                finalLab0
                                    = comsToLab(finalComs0, v.mass0, v.coeff0, v.gammaComs, v.factorA, v.comsVelocity);
                                par0[momentum_] = precisionCast<float_X>(finalLab0 * v.normalizedWeight0);
                                if((v.normalizedWeight0 / v.normalizedWeight1) - rng(worker) >= 0.0_COLL)
                                {
                                    finalLab1 = comsToLab(
                                        -1.0_COLL * finalComs0,
                                        v.mass1,
                                        v.coeff1,
                                        v.gammaComs,
                                        v.factorA,
                                        v.comsVelocity);
                                    par1[momentum_] = precisionCast<float_X>(finalLab1 * v.normalizedWeight1);
                                }
                            }
                        }


                    public:
                        /** Execute the collision functor
                         *
                         * @param ctx collision context
                         * @param par0 1st colliding macro particle
                         * @param par1 2nd colliding macro particle
                         */
                        template<typename T_Context, typename T_Par0, typename T_Par1>
                        DINLINE void operator()(T_Context const& ctx, T_Par0& par0, T_Par1& par1)
                        {
                            if((par0[momentum_] == float3_X{0.0_X, 0.0_X, 0.0_X})
                               && (par1[momentum_] == float3_X{0.0_X, 0.0_X, 0.0_X}))
                                return;
                            const Variables v{par0, par1};
                            if(v.comsMomentum0Norm2 == 0.0_COLL)
                                return;
                            const float_COLL coulombLog = coulombLogFunctor(v);
                            const float_COLL s12 = normalizedPathLength(v, coulombLog);
                            RelativisticCollisionBase<ifDebug>::processDebugValues(coulombLog, s12);
                            lastPart(v, ctx, par0, par1, s12);
                        }
                    };
                } // namespace acc
            } // namespace relativistic
        } // namespace collision
    } // namespace particles
} // namespace picongpu
