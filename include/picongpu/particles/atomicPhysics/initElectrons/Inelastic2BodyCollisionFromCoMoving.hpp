/* Copyright 2023 Brian Marre
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

//! @file implements init of macro electron as inelastic collision of co-moving electron with ion

#pragma once

#include "picongpu/simulation_defines.hpp"
// need physicalConstants.param

#include "picongpu/particles/atomicPhysics/ConvertEnum.hpp"
#include "picongpu/particles/atomicPhysics/initElectrons/CloneAdditionalAttributes.hpp"

#include <pmacc/algorithms/math/PowerFunction.hpp>
#include <pmacc/math/Matrix.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics::initElectrons
{
    // momenta are always 3-dimensional in picongpu
    using Matrix_3x3 = pmacc::math::Matrix<float_64, pmacc::math::CT::UInt32<3u, 3u>>;
    using MatrixVector = pmacc::math::Matrix<float_64, pmacc::math::CT::UInt32<3u, 1u>>;

    namespace detail
    {
        struct LorentzBoost
        {
            float_64 m_gamma;
            MatrixVector m_beta;
            Matrix_3x3 spaceMatrix;

            HDINLINE LorentzBoost(float_64 const gamma, MatrixVector const& beta, Matrix_3x3 const& lorentzMatrix)
                : m_gamma(gamma)
                , m_beta(beta)
                , spaceMatrix(lorentzMatrix)
            {
            }
        };
    } // namespace detail

    struct Inelastic2BodyCollisionFromCoMoving
    {
    private:
        /** fill space components of a Lorentz boots matrix
         *
         * @attention normBetaSquared, beta and gamma must be consistent!
         */
        HDINLINE static Matrix_3x3 fillLorentzMatrix(
            MatrixVector const& beta,
            float_64 const gamma,
            float_64 const normBetaSquared)
        {
            Matrix_3x3 lorentzMatrix(0.);

            /// @detail split between general and diagonal contributions, not that readable
            ///     but faster than checking for every component

            // general contributions
            if(normBetaSquared != 0.0)
            {
                // standard case
                for(uint32_t j = 0u; j < 3u; j++)
                {
                    auto const jTerm = (gamma - 1.) * beta(j, u32(0u)) / normBetaSquared;
                    for(uint32_t i = 0u; i < 3u; i++)
                    {
                        lorentzMatrix(i, j) = jTerm * beta(i, u32(0u));
                    }
                }
            }

            // diagonal only contributions
            for(uint32_t i = 0u; i < 3u; i++)
                lorentzMatrix(i, i) += 1.;

            return lorentzMatrix;
        }

        /** get LorentzBoost instance for boost from ion frame of reference to lab
         *
         * @param ion ion defining ion frame of reference
         */
        template<typename T_IonParticle>
        HDINLINE static detail::LorentzBoost getLorentzBoostFromIonToLab(T_IonParticle& ion)
        {
            // sim.unit.mass() * sim.unit.length() / sim.unit.time(), **weighted**
            float3_64 const momentum_Lab = static_cast<float3_64>(ion[momentum_]);
            // unitless
            float_64 const weighting = static_cast<float_64>(ion[weighting_]);

            // sim.unit.mass()^2 * sim.unit.length()^2 / sim.unit.time()^2, not weighted
            float_64 const momentumSquared_Lab = pmacc::math::l2norm2(momentum_Lab) / (weighting * weighting);

            // sim.unit.mass(), not weighted
            float_64 const mass
                = static_cast<float_64>(picongpu::traits::frame::getMass<typename T_IonParticle::FrameType>());

            // sim.unit.mass()^2 * sim.unit.length()^2 / sim.unit.time()^2, not weighted
            float_64 const m2c2 = pmacc::math::cPow(
                mass * picongpu::sim.si.getSpeedOfLight() / (sim.unit.length() / sim.unit.time()),
                2u);

            // (sim.unit.mass() * sim.unit.length() / sim.unit.time(), weighted) / weighting
            //      / sqrt( (sim.unit.mass()^2 * sim.unit.length()^2 / sim.unit.time()^2, not weighted)
            //              + (sim.unit.mass()^2 * sim.unit.length()^2 / sim.unit.time()^2, not weighted) )
            // unitless, not weighted
            auto const beta = MatrixVector(
                /* inverse direction in ion system */ (-1.) * momentum_Lab
                / (weighting * math::sqrt(momentumSquared_Lab + m2c2)));

            // ((sim.unit.mass()^2 * sim.unit.length()^2 / sim.unit.time()^2, not weighted)
            //  / (sim.unit.mass()^2 * sim.unit.length()^2 / sim.unit.time()^2, not weighted)) + unitless
            // unitless, not weighted
            float_64 const gamma = math::sqrt(momentumSquared_Lab / m2c2 + 1.0);

            // (sim.unit.mass()^2 * sim.unit.length()^2 / sim.unit.time()^2, not weighted)
            //  / ((sim.unit.mass()^2 * sim.unit.length()^2 / sim.unit.time()^2, not weighted) * (unitless, not
            //  weighted)^2)
            // unitless, not weighted
            float_64 const normBetaSquared = momentumSquared_Lab / (m2c2 * gamma * gamma);

            // lower 3x3 block of Lorentz boost matrix for transformation from Ion-System to Lab-System
            // unitless
            return detail::LorentzBoost(gamma, beta, fillLorentzMatrix(beta, gamma, normBetaSquared));
        }

    public:
        /** init electron according to inelastic relativistic collision of initially
         *  co-moving particles with deltaEnergy as break-up/decay energy
         *
         * @param ion Particle, view of ion frame slot
         * @param electron Particle, view of electron frame slot
         * @param deltaEnergy in eV, energy difference between initial and
         *  final state of transition
         * @param rngFactory factory for uniformly distributed random number generator
         *
         * @attention numerically unstable for highly relativistic ion/electrons (and MeV+ deltaEnergys)
         * @attention assumes that ion and electron have equal weight
         *
         * see Brian Marre, notebook 01.06.2022-?, p.110-178 + p.84-85 + p.101-102 for full derivation
         *
         * Naming Legend:
         * - Def.: Ion-system ... frame of reference co-moving with ion before scattering
         * - Def.: Lab-system ... frame of reference of PIC-simulation
         * - *Star* ... after inelastic collision, otherwise before
         */
        template<typename T_Worker, typename T_IonParticle, typename T_ElectronParticle, typename T_RngGeneratorFloat>
        HDINLINE static void initElectron(
            T_Worker const& worker,
            T_IonParticle& ion,
            T_ElectronParticle& electron,
            IdGenerator& idGen,
            // eV
            float_X const deltaEnergy,
            /// const?, @todo Brian Marre, 2023
            T_RngGeneratorFloat& rngGenerator)
        {
            CloneAdditionalAttributes::init(worker, ion, electron, idGen);

            // sim.unit.mass(), not weighted
            float_X const massElectron = picongpu::traits::frame::getMass<typename T_ElectronParticle::FrameType>();
            // sim.unit.mass(), not weighted
            float_X const massIon = picongpu::traits::frame::getMass<typename T_IonParticle::FrameType>();

            // init+update momentum
            if(deltaEnergy <= 0._X)
            {
                if constexpr(picongpu::atomicPhysics::debug::initIonizationElectrons::
                                 CHECK_DELTA_ENERGY_INIT_FROM_COMOVING_POSITIVE)
                    if(deltaEnergy < 0._X)
                        printf(
                            "atomicPhysics ERROR: init as inelastic collision from coMoving reference frame"
                            "is unphysical for deltaEnergy < 0, deltaEnergy = %.8f !\n",
                            deltaEnergy);

                electron[momentum_] = ion[momentum_] * massElectron / massIon;
                return;
            }

            /// calculate electron gamma after inelastic scattering in ion system @{
            // sim.unit.mass()/sim.unit.mass() = unitless
            float_64 const mE_mI = static_cast<float_64>(massElectron / massIon);
            // kg * m^2/s^2 * keV/J * 1e3 = J/J * eV = eV
            float_64 const mc2_Ion = static_cast<float_64>(massIon) * sim.unit.mass()
                * pmacc::math::cPow(picongpu::sim.si.getSpeedOfLight(), 2u) * sim.si.conv.joule2ev(1.0);

            //  eV / eV + sim.unit.mass() / sim.unit.mass() = unitless
            float_64 const A_E = deltaEnergy / mc2_Ion + mE_mI;
            // (unitless + unitless)/ unitless = unitless
            float_64 const gammaStarElectron_IonSystem
                = (A_E * (A_E + 2.) + (mE_mI * mE_mI)) / ((A_E + 1.) * 2. * mE_mI);
            /// @}

            /// get momentum norm from gamma @{
            // sim.unit.mass() * sim.unit.length() / sim.unit.time(), not weighted
            float_64 const mc_Electron
                = massElectron * picongpu::sim.si.getSpeedOfLight() / (sim.unit.length() / sim.unit.time());

            // norm of electron momentum after inelastic collision in Ion-System
            // weight * (unitless^2 - unitless) * (sim.unit.mass() * sim.unit.length()/sim.unit.time())
            // = sim.unit.mass() * sim.unit.length() / sim.unit.time(), weighted
            float_64 const normMomentumStarElectron_IonSystem = static_cast<float_64>(electron[weighting_])
                * mc_Electron * math::sqrt(pmacc::math::cPow(gammaStarElectron_IonSystem, u8(2u)) - 1.);
            ///@}

            /// choose scattering direction @{
            float_X const u = rngGenerator();
            float_X const v = rngGenerator();

            float_X const cosTheta = 1._X - 2._X * v;
            float_X const sinTheta = math::sqrt(v * (2._X - v));
            float_X const phi = 2._X * static_cast<float_X>(picongpu::PI) * u;

            float_X sinPhi;
            float_X cosPhi;
            pmacc::math::sincos(phi, sinPhi, cosPhi);

            /// @note momenta are always 3-dimensional in picongpu!
            float3_64 const directionVector = float3_64(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
            /// @}

            // sim.unit.mass() * sim.unit.length() / sim.unit.time(), weighted
            auto momentumStarElectron_IonSystem = MatrixVector(normMomentumStarElectron_IonSystem * directionVector);

            /// Lorentz transformation from IonSystem to LabSystem
            detail::LorentzBoost lorentzBoost = getLorentzBoostFromIonToLab(ion);
            MatrixVector momentumStarElectron_LabSystem
                = lorentzBoost.spaceMatrix.mMul(momentumStarElectron_IonSystem);

            // sim.unit.mass()/sim.unit.mass() = unitless
            float_64 const mI_mE = static_cast<float_64>(massIon / massElectron);
            // kg * m^2/s^2 * keV/J * 1e3 = J/J * eV = eV
            float_64 const mc2_Electron = static_cast<float_64>(massElectron) * sim.unit.mass()
                * pmacc::math::cPow(picongpu::sim.si.getSpeedOfLight(), 2u) * sim.si.conv.joule2ev(1.0);

            float_64 const A_I = deltaEnergy / mc2_Electron + mI_mE;
            float_64 const gammaStarIon_IonSystem = (A_I * (A_I + 2.) + (mI_mE * mI_mE)) / ((A_I + 1.) * 2. * mI_mE);

            // sim.unit.mass() * sim.unit.length() / sim.unit.time(), not weighted
            float_64 const mc_Ion
                = massIon * picongpu::sim.si.getSpeedOfLight() / (sim.unit.length() / sim.unit.time());

            // sim.unit.mass() * sim.unit.length()/sim.unit.time(), weighted
            MatrixVector momentumStarIon_LabSystem = momentumStarElectron_LabSystem /* ion = - electron */ * -1.
                + lorentzBoost.m_beta * (lorentzBoost.m_gamma * gammaStarIon_IonSystem * mc_Ion);
            // sim.unit.mass() * sim.unit.length() / sim.unit.time(), weighted
            momentumStarElectron_LabSystem = momentumStarElectron_LabSystem
                + lorentzBoost.m_beta * (lorentzBoost.m_gamma * gammaStarElectron_IonSystem * mc_Electron);

            // set to particle, @attention momentum is always 3D in picongpu!
            for(uint32_t i = 0u; i < 3u; i++)
                ion[momentum_][i] = static_cast<float_X>(momentumStarIon_LabSystem(i, 0u));

            // set to particle
            for(uint32_t i = 0u; i < 3u; i++)
                electron[momentum_][i] = static_cast<float_X>(momentumStarElectron_LabSystem(i, 0u));
        }
    };
} // namespace picongpu::particles::atomicPhysics::initElectrons
