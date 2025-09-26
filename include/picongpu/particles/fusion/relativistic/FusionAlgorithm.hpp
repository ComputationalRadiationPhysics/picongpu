/* Copyright 2015-2024 Rene Widera, Filip Optolowicz
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

#include "picongpu/defines.hpp"
#include "picongpu/particles/fusion/param.hpp"

#include <pmacc/random/distributions/Uniform.hpp>

#include <cmath>
#include <cstdio>
#include <type_traits>
#include <utility>

namespace picongpu::particles::fusion::relativistic
{
    namespace acc
    {
        using namespace pmacc;
        using namespace picongpu::particles::fusion::precision;
        constexpr float_COLL c = static_cast<float_COLL>(sim.pic.getSpeedOfLight());
        constexpr float_COLL c2 = c * c;
        constexpr float_COLL c3 = c2 * c;
        constexpr float_COLL c4 = c3 * c;

        /**
         * @brief Calculates the total relativistic energy of a particle.
         *
         * @tparam T_Float The floating point type for the calculation.
         * @tparam T_Vec The type of the 3-momentum vector.
         * @tparam T_Mass The type of the rest mass.
         * @param momentum The relativistic 3-momentum vector of the particle.
         * @param mass The rest mass of the particle.
         * @return The total relativistic energy E.
         */
        template<typename T_Float, typename T_Vec, typename T_Mass>
        DINLINE T_Float energy(T_Vec const& momentum, T_Mass const& mass)
        {
            // Using the formula E = sqrt((pc)^2 + (mc^2)^2)
            // which is E = sqrt(p^2 * c^2 + m^2 * c^4)
            // where p is the magnitude of the 3-momentum vector.

            // T_Float const p_sq = pmacc::math::l2norm2(momentum);
            // return math::sqrt(p_sq * c2 + mass * mass * c4);

            // gamma*m*c^2 - probably more numerically stable
            return picongpu::gamma<float_COLL>(momentum, mass) * mass * c2;
        }

        //! Stores some precalculated values used in the fusion algorithm - also:
        //! calculates momenta of outgoing particles if needed
        struct FusionKinematics
        {
            PMACC_ALIGN(labMomentum0, float3_COLL);
            PMACC_ALIGN(labMomentum1, float3_COLL);
            PMACC_ALIGN(mass0, float_COLL);
            PMACC_ALIGN(mass1, float_COLL);

            PMACC_ALIGN(V_cm, float3_COLL);
            PMACC_ALIGN(gamma_cm, float_COLL);
            PMACC_ALIGN(factorA, float_COLL);
            PMACC_ALIGN(E_cm_tot_sq, float_COLL); // total CM energy squared
            PMACC_ALIGN(E_cm_tot, float_COLL); // total CM energy
            PMACC_ALIGN(E_r, float_COLL); // CM 'relative' kinetic energy
            PMACC_ALIGN(V_rel_mag, float_COLL); // Magnitude of relative velocity for cross-section

            /**
             * @brief Constructs the FusionKinematics struct and calculates key values for a fusion reaction.
             *
             * @tparam T_Par0 The type of the first reactant particle.
             * @tparam T_Par1 The type of the second reactant particle.
             * @param par0 The first reactant particle.
             * @param par1 The second reactant particle.
             */
            template<typename T_Par0, typename T_Par1>
            DINLINE FusionKinematics(T_Par0 const& par0, T_Par1 const& par1)
                : labMomentum0(precisionCast<float_COLL>(par0[momentum_]) / par0[weighting_])
                , labMomentum1(precisionCast<float_COLL>(par1[momentum_]) / par1[weighting_])
                , mass0(precisionCast<float_COLL>(picongpu::traits::attribute::getMass(1, par0)))
                , mass1(precisionCast<float_COLL>(picongpu::traits::attribute::getMass(1, par1)))
            {
                // --- Calculate total 4-momentum in Lab Frame ---
                // In PIC units, non weighted
                float_COLL const E0_lab = energy<float_COLL>(labMomentum0, mass0);
                float_COLL const E1_lab = energy<float_COLL>(labMomentum1, mass1);
                float_COLL const E_tot_lab = E0_lab + E1_lab;
                float3_COLL const p_tot_lab = labMomentum0 + labMomentum1;

                // --- Invariant CM energy (s) and CM velocity ---
                E_cm_tot_sq = E_tot_lab * E_tot_lab - pmacc::math::l2norm2(p_tot_lab) * c2;
                E_cm_tot_sq = math::max(E_cm_tot_sq, 0.0_COLL); // guard
                E_cm_tot = math::sqrt(E_cm_tot_sq);

                // Velocity of the CM frame in the Lab frame
                V_cm = p_tot_lab * c2 / E_tot_lab;
                float_COLL const beta2 = pmacc::math::l2norm2(V_cm) / c2;
                gamma_cm = 1.0_COLL / math::sqrt(1.0_COLL - beta2);

                // (gamma-1)/V^2 = gamma^2 / [c^2 (gamma+1)]  (stable as V->0)
                factorA = (gamma_cm * gamma_cm) / (c2 * (gamma_cm + 1.0_COLL));

                // --- Invariants for v_rel and E_r ---
                //(Cannoni) = https://arxiv.org/abs/1605.00569v2
                float_COLL const s = E_cm_tot_sq;
                float_COLL const m0c2 = mass0 * c2;
                float_COLL const m1c2 = mass1 * c2;

                // Lorentz factor of the relative motion (Cannoni eq. (42))
                float_COLL const gamma_r = (s - m0c2 * m0c2 - m1c2 * m1c2) / (2.0_COLL * m0c2 * m1c2);
                float_COLL const gamma_r_clamped = math::max(gamma_r, 1.0_COLL);
                // Relative velocity magnitude |v_rel| (Cannoni inv eq. (32))
                V_rel_mag = c * math::sqrt(1.0_COLL - 1.0_COLL / (gamma_r_clamped * gamma_r_clamped));

                // CM 'relative' kinetic energy E_r = mu c^2 (gamma_r - 1)
                float_COLL const mu = (mass0 * mass1) / (mass0 + mass1);
                E_r = mu * c2 * (gamma_r_clamped - 1.0_COLL);
            }

            /**
             * @brief Calculates the momenta of the product particles in the lab frame.
             *
             * This function performs a Lorentz boost from the center-of-mass (CM) frame back to the
             * laboratory (Lab) frame to determine the final momenta of the two product particles.
             *
             * @tparam T_Product0Box The type of the first product particle box.
             * @tparam T_Product1Box The type of the second product particle box.
             * @param dir The direction vector of the first product particle's momentum in the CM frame.
             */
            template<typename T_Product0Box, typename T_Product1Box>
            DINLINE void P(float3_COLL const& dir)
            {
                // --- Define reaction properties ---
                float_COLL const mP0 = picongpu::traits::frame::getMass<typename T_Product0Box::FrameType>();
                float_COLL const mP1 = picongpu::traits::frame::getMass<typename T_Product1Box::FrameType>();
                // float_COLL const Q = (mass0 + mass1 - mP0 - mP1) * c2; // Q-value of the reaction - no
                // need because we have E_cm_tot

                // debug or necessary? -> it can happen only for endothermic reactions and the cross
                // sections should be zero than?
                float_COLL const mP0c2 = mP0 * c2;
                float_COLL const mP1c2 = mP1 * c2;
                if(E_cm_tot < (mP0c2 + mP1c2))
                {
                    printf(
                        "Warning: Fusion reaction energy is too low for product masses. E_cm_tot: %f, "
                        "mP0c2: %f, mP1c2: %f\n",
                        E_cm_tot,
                        mP0c2,
                        mP1c2);
                    labMomentum0 = float3_COLL::create(0);
                    labMomentum1 = float3_COLL::create(0);
                    return;
                }

                // Relativistic formula for the TOTAL energy of product 0 in the CM frame
                float_COLL const E_p0_tot_cm
                    = (E_cm_tot * E_cm_tot + (mP0 * mP0 - mP1 * mP1) * c4) / (2.0_COLL * E_cm_tot);
                // TOTAL energy of product 1
                float_COLL const E_p1_tot_cm = E_cm_tot - E_p0_tot_cm;


                // --- Calculate Product Momenta (CM Frame) ---
                // Magnitude of momentum for product 0, from p = sqrt(E_tot^2 - (mc^2)^2) / c
                float_COLL const p0_mag_cm = math::sqrt(E_cm_tot_sq - mP0 * mP0 * c4) / c;
                // Momentum vectors in the CM frame
                float3_COLL const p0_cm = p0_mag_cm * dir;
                float3_COLL const p1_cm = -p0_cm;

                // --- Inverse Lorentz Boost back to Lab Frame ---
                // We apply the reverse transformation using the pre-calculated V_cm and gamma_cm.
                // The structure is similar to the forward boost, but the sign of the velocity-dependent
                // term is flipped.

                // For Product 0:
                float_COLL const gamma_p0_cm = E_p0_tot_cm / (mP0 * c2);
                float3_COLL const u0_cm = p0_cm / mP0;
                float3_COLL const u0_lab = u0_cm + (math::dot(V_cm, u0_cm) * factorA + gamma_cm * gamma_p0_cm) * V_cm;
                labMomentum0 = u0_lab * mP0;

                // For Product 1:
                float_COLL const gamma_p1_cm = E_p1_tot_cm / (mP1 * c2);
                float3_COLL const u1_cm = p1_cm / mP1;
                float3_COLL const u1_lab = u1_cm + (math::dot(V_cm, u1_cm) * factorA + gamma_cm * gamma_p1_cm) * V_cm;
                labMomentum1 = u1_lab * mP1;
                if constexpr(debugFusion)
                {
                    printf(
                        "  Product 1: mass: %f, momentum: %f, %f, %f, energy: %f\n",
                        mP0,
                        labMomentum0[0],
                        labMomentum0[1],
                        labMomentum0[2],
                        energy<float_X>(labMomentum0, mP0));
                    printf(
                        "  Product 2: mass: %f, momentum: %f, %f, %f, energy: %f\n",
                        mP1,
                        labMomentum1[0],
                        labMomentum1[1],
                        labMomentum1[2],
                        energy<float_X>(labMomentum1, mP1));
                }
            }

            /**
             * @brief Returns the momentum of the first product particle in the lab frame.
             *
             * @return A `float3_X` vector representing the momentum of the first product.
             */
            DINLINE float3_X P0() const
            {
                return precisionCast<float_X>(labMomentum0);
            }

            /**
             * @brief Returns the momentum of the second product particle in the lab frame.
             *
             * @return A `float3_X` vector representing the momentum of the second product.
             */
            DINLINE float3_X P1() const
            {
                return precisionCast<float_X>(labMomentum1);
            }
        };

        /**
         * @brief Defines the fusion algorithm for calculating the probability and outcome of a fusion reaction.
         *
         * @tparam T_CrossSection A callable object that provides the cross-section of the reaction.
         */
        template<typename T_CrossSection>
        struct FusionAlg
        {
            HDINLINE FusionAlg() {};
            PMACC_ALIGN(crossSection, T_CrossSection);


        public:
            /**
             * @brief Attempts to fuse two reactant particles and calculates the momenta of the products.
             * The momenta of the products are returned via reference parameters.
             *
             * This function calculates the probability of a fusion event based on the relative kinetic energy
             * of the reactants and the fusion cross-section. If a fusion event occurs, it calculates the
             * new momenta of the resulting product particles.
             *
             * @tparam T_Product0Box The type of the first product particle box.
             * @tparam T_Product1Box The type of the second product particle box.
             * @tparam T_Worker The worker object.
             * @tparam T_Par0 The type of the first reactant particle.
             * @tparam T_Par1 The type of the second reactant particle.
             * @tparam T_RngHandle The type of the random number generator handle.
             * @param worker The worker object.
             * @param par0 The first reactant particle.
             * @param par1 The second reactant particle.
             * @param weightingR1 The weighting factor of the first reactant.
             * @param weightingR2 The weighting factor of the second reactant.
             * @param probabilityFactor A factor to adjust the fusion probability.
             * @param mom0 The output parameter for the momentum of the first product.
             * @param mom1 The output parameter for the momentum of the second product.
             * @param rngHandle The random number generator handle.
             */
            template<
                typename T_Product0Box,
                typename T_Product1Box,
                typename T_Worker,
                typename T_Par0,
                typename T_Par1,
                typename T_RngHandle>
            DINLINE void fuse(
                T_Worker const& worker,
                T_Par0 par0,
                T_Par1 par1,
                float_X weightingR1,
                float_X weightingR2,
                float_X probabilityFactor,
                float3_X& mom0,
                float3_X& mom1,
                T_RngHandle& rngHandle)
            {
                if((par0[momentum_] == float3_X{0.0_X, 0.0_X, 0.0_X})
                   && (par1[momentum_] == float3_X{0.0_X, 0.0_X, 0.0_X}))
                    return;


                // Get a random float value from 0,1
                using UniformFloat = pmacc::random::distributions::Uniform<
                    pmacc::random::distributions::uniform::ExcludeOne<float_COLL>::Reduced>;
                auto rng = rngHandle.template applyDistribution<UniformFloat>();
                float_COLL rngValue1 = rng(worker);


                // calculate relative velocity and relative energy
                FusionKinematics fusionVar{par0, par1};


                // Convert energy from PIC units to keV
                constexpr float_COLL picEnergy_to_Joule = sim.unit.energy();
                constexpr float_COLL joule_to_eV = 1.0 / sim.si.get_eV();
                constexpr float_COLL eV_to_keV = 1e-3;
                constexpr float_COLL convToKeV = picEnergy_to_Joule * joule_to_eV * eV_to_keV;

                // Convert cross section from millibarns to PIC area units
                constexpr float_COLL millibarn_to_m2 = 1e-31; // 1 millibarn = 1e-31 m²
                constexpr float_COLL picLength_to_m = sim.unit.length(); // [m/UNIT_LENGTH]
                constexpr float_COLL m2_to_picArea = 1.0 / (picLength_to_m * picLength_to_m);
                constexpr float_COLL millibarn_to_picArea = millibarn_to_m2 * m2_to_picArea;

                float_X sigma_picArea = crossSection(fusionVar.E_r * convToKeV) * millibarn_to_picArea;
                float_X P = probabilityFactor * sigma_picArea * fusionVar.V_rel_mag * fusionVar.gamma_cm;


                // print with probability 1e-8
                if constexpr(debugFusion)
                    if(rng(worker) < 1e-8)
                    {
                        // get pic velocity to m/s -> debug
                        constexpr float_COLL picVelocity_to_m_per_s = sim.unit.length() / sim.unit.time();

                        // Apply conversions
                        float_X sigma_milibarns = crossSection(fusionVar.E_r * convToKeV);
                        // print particle id
                        printf("Particle 1 ID: %lu, Particle 2 ID: %lu\n", par0[particleId_], par1[particleId_]);

                        printf(
                            "Worker %d,millibarn_to_picArea: %e, sigma_milibarns: %e, probabilityFactor: %e, "
                            "fusionVar.V_rel [m/s]: %e, fusionVar.gamma_cm: %e, P: %e\n",
                            worker.workerIdx(),
                            millibarn_to_picArea,
                            sigma_milibarns,
                            probabilityFactor,
                            fusionVar.V_rel_mag * picVelocity_to_m_per_s,
                            fusionVar.gamma_cm,
                            P);
                        printf("E_r [keV]: %f, sigma [mb]: %f\n", fusionVar.E_r * convToKeV, sigma_milibarns);
                        // print momenta
                        printf(
                            "  Reactant 1: weight: %f, mass: %f, momentum: %f, %f, %f, energy: %f\n",
                            weightingR1,
                            fusionVar.mass0,
                            par0[momentum_][0],
                            par0[momentum_][1],
                            par0[momentum_][2],
                            energy<float_X>(par0[momentum_], fusionVar.mass0));
                        printf(
                            "  Reactant 2: weight: %f, mass: %f, momentum: %f, %f, %f, energy: %f\n",
                            weightingR2,
                            fusionVar.mass1,
                            par1[momentum_][0],
                            par1[momentum_][1],
                            par1[momentum_][2],
                            energy<float_X>(par1[momentum_], fusionVar.mass1));
                    }

                if constexpr(alwaysFuseQ)
                    P = 1.0_COLL; // always fuse if this is set to true


                if(fusionVar.E_r <= 0.0_COLL)
                {
                    // print Er
                    if constexpr(debugFusion)
                        if(worker.workerIdx() == 0 && rng(worker) < 1e-8)
                        {
                            printf("Warning: Relative kinetic energy E_r is non-positive: %f\n", fusionVar.E_r);
                        }
                    // No relative kinetic energy, no reaction possible
                    return;
                }

                if(rngValue1 < P)
                {
                    float_COLL rngValue2 = rng(worker);
                    float_COLL rngValue3 = rng(worker);

                    // rngValues 2 and 3 are used to generate the scattering angle
                    float_COLL x1 = 2.0_COLL * rngValue2 - 1.0_COLL; // [-1,1]
                    float_COLL x2 = 2.0_COLL * rngValue3 - 1.0_COLL; // [-1,1]
                    while(x1 * x1 + x2 * x2 > 1.0_COLL)
                    {
                        // rejection sampling
                        rngValue2 = rng(worker);
                        rngValue3 = rng(worker);
                        x1 = 2.0_COLL * rngValue2 - 1.0_COLL; // [-1,1]
                        x2 = 2.0_COLL * rngValue3 - 1.0_COLL; // [-1,1]
                    }
                    float_COLL s = math::sqrt(1.0_COLL - x1 * x1 - x2 * x2);
                    float_COLL x = 2.0_COLL * x1 * s;
                    float_COLL y = 2.0_COLL * x2 * s;
                    float_COLL z = 1.0_COLL - 2 * (x1 * x1 + x2 * x2);


                    // returns momentum of one particle - not multiplied by weighting.
                    // Multiplication by weighting is later in creation of particles
                    float3_COLL const dir = float3_COLL(x, y, z);
                    fusionVar.P<T_Product0Box, T_Product1Box>(dir);
                    mom0 = fusionVar.P0();
                    mom1 = fusionVar.P1();
                }
                else
                {
                    mom0 = float3_X{0.0_X, 0.0_X, 0.0_X};
                    mom1 = float3_X{0.0_X, 0.0_X, 0.0_X};
                }
            }
        };
    } // namespace acc
} // namespace picongpu::particles::fusion::relativistic
