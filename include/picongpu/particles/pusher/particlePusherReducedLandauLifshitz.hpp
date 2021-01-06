/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch
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
#include "picongpu/traits/attribute/GetMass.hpp"
#include "picongpu/traits/attribute/GetCharge.hpp"
#include "picongpu/particles/interpolationMemoryPolicy/ShiftToValidRange.hpp"
#include <pmacc/math/Vector.hpp>
#include <pmacc/math/RungeKutta.hpp>


namespace picongpu
{
    namespace particlePusherReducedLandauLifshitz
    {
        /* This pusher uses the Lorentz force and a reduced
         * Landau Lifshitz term to push particles based on the
         * Runge Kutta solver 4th order. It takes into account
         * the energy loss due to radiation.
         *
         * More details on this approach can be found in
         * Marija Vranic's paper: Classical Radiation Reaction
         * in Particle-In-Cell Simulations
         * http://arxiv.org/abs/1502.02432
         */
        template<class Velocity, class Gamma>
        struct Push
        {
            /* this is an optional extention for sub-sampling pushes that enables grid to particle interpolation
             * for particle positions outside the super cell in one push
             */
            using LowerMargin = typename pmacc::math::CT::make_Int<simDim, 1>::type;
            using UpperMargin = typename pmacc::math::CT::make_Int<simDim, 1>::type;

            template<typename T_FunctorFieldE, typename T_FunctorFieldB, typename T_Particle, typename T_Pos>
            HDINLINE void operator()(
                const T_FunctorFieldB functorBField, /* at t=0 */
                const T_FunctorFieldE functorEField, /* at t=0 */
                T_Particle& particle,
                T_Pos& pos, /* at t=0 */
                const uint32_t)
            {
                float_X const weighting = particle[weighting_];
                float_X const mass = attribute::getMass(weighting, particle);
                float_X const charge = attribute::getCharge(weighting, particle);

                using TypeBFieldFunctor = T_FunctorFieldB;
                using TypeEFieldFunctor = T_FunctorFieldE;
                using TypePosition = position_pic::type;
                using TypeMomentum = momentum::type;
                using TypeMass = float_X;
                using TypeCharge = float_X;
                using TypeWeighting = weighting::type;

                TypeMomentum mom = particle[momentum_];

                const float_X deltaT = DELTA_T;
                const uint32_t dimMomentum = GetNComponents<TypeMomentum>::value;
                // the transver data type adjust to 3D3V, 2D3V, 2D2V, ...
                using VariableType = pmacc::math::Vector<picongpu::float_X, simDim + dimMomentum>;
                VariableType var;

                // transfer position
                for(uint32_t i = 0; i < picongpu::simDim; ++i)
                    var[i] = pos[i];

                // transfer momentum
                for(uint32_t i = 0; i < dimMomentum; ++i)
                    var[simDim + i] = mom[i];

                using DiffEqType = DiffEquation<
                    VariableType,
                    float_X,
                    TypeEFieldFunctor,
                    TypeBFieldFunctor,
                    TypePosition,
                    TypeMomentum,
                    TypeMass,
                    TypeCharge,
                    TypeWeighting,
                    Velocity,
                    Gamma>;
                DiffEqType diffEq(functorEField, functorBField, mass, charge, weighting);

                VariableType varNew = pmacc::math::RungeKutta4()(diffEq, var, float_X(0.0), deltaT);

                // transfer position
                for(uint32_t i = 0; i < picongpu::simDim; ++i)
                    pos[i] = varNew[i];

                // transfer momentum
                for(uint32_t i = 0; i < dimMomentum; ++i)
                    mom[i] = varNew[simDim + i];

                particle[momentum_] = mom;
            }

            template<
                typename T_Var,
                typename T_Time,
                typename T_FieldEFunc,
                typename T_FieldBFunc,
                typename T_Pos,
                typename T_Mom,
                typename T_Mass,
                typename T_Charge,
                typename T_Weighting,
                typename T_Velocity,
                typename T_Gamma>
            struct DiffEquation
            {
                // alias for types to  follow coding guide line
                using VariableType = T_Var;
                using TimeType = T_Time;
                using EFieldFuncType = T_FieldEFunc;
                using BFieldFuncType = T_FieldBFunc;
                using PositionType = T_Pos;
                using MomentumType = T_Mom;
                using MassType = T_Mass;
                using ChargeType = T_Charge;
                using WeightingType = T_Weighting;
                using VelocityType = T_Velocity;
                using GammaType = T_Gamma;


                HDINLINE DiffEquation(
                    EFieldFuncType funcE,
                    BFieldFuncType funcB,
                    MassType m,
                    ChargeType q,
                    WeightingType w)
                    : fieldEFunc(funcE)
                    , fieldBFunc(funcB)
                    , mass(m)
                    , charge(q)
                    , weighting(w)
                {
                }

                HDINLINE VariableType operator()(TimeType time, VariableType var) const
                {
                    PositionType pos;
                    PositionType posInterpolation;
                    MomentumType mom;
                    // transfer position
                    for(uint32_t i = 0; i < picongpu::simDim; ++i)
                    {
                        posInterpolation[i] = var[i];
                        pos[i] = var[i] * cellSize[i];
                    }

                    auto fieldE = fieldEFunc(
                        posInterpolation,
                        picongpu::particles::interpolationMemoryPolicy::ShiftToValidRange());
                    auto fieldB = fieldBFunc(
                        posInterpolation,
                        picongpu::particles::interpolationMemoryPolicy::ShiftToValidRange());

                    // transfer momentum
                    const uint32_t dimMomentum = GetNComponents<MomentumType>::value;
                    for(uint32_t i = 0; i < dimMomentum; ++i)
                        mom[i] = var[simDim + i];

                    VelocityType velocityCalc;
                    GammaType gammaCalc;
                    const float_X c = SPEED_OF_LIGHT;
                    const float3_X velocity = velocityCalc(mom, mass);
                    const float_X gamma = gammaCalc(mom, mass);
                    const float_X conversionMomentum2Beta = 1.0 / (gamma * mass * c);

                    const float_X c2 = c * c;
                    const float_X charge2 = charge * charge;
                    const float3_X beta = velocity / c;

                    const float_X prefactorRR = 2. / 3. * charge2 * charge2 / (4. * PI * EPS0 * mass * mass * c2 * c2);
                    const float3_X lorentz = fieldE + conversionMomentum2Beta * c * pmacc::math::cross(mom, fieldB);
                    const float_X fieldETimesBeta = pmacc::math::dot(fieldE, mom) * conversionMomentum2Beta;
                    const float3_X radReactionVec = c
                            * (pmacc::math::cross(fieldE, fieldB)
                               + c * conversionMomentum2Beta
                                   * pmacc::math::cross(fieldB, pmacc::math::cross(fieldB, mom)))
                        + conversionMomentum2Beta * fieldE * pmacc::math::dot(mom, fieldE)
                        - gamma * gamma * conversionMomentum2Beta
                            * (mom * (pmacc::math::dot(lorentz, lorentz) - fieldETimesBeta * fieldETimesBeta));

                    const float3_X diffMom = charge * lorentz + (prefactorRR / weighting) * radReactionVec;
                    const float3_X diffPos = velocity;

                    VariableType returnVar;
                    for(uint32_t i = 0; i < picongpu::simDim; ++i)
                        returnVar[i] = diffPos[i] / cellSize[i];

                    for(uint32_t i = 0; i < dimMomentum; ++i)
                        returnVar[simDim + i] = diffMom[i];

                    return returnVar;
                }


            private:
                EFieldFuncType fieldEFunc; /* functor E field interpolation */
                BFieldFuncType fieldBFunc; /* functor B field interpolation */
                MassType mass; /* mass of the macro particle */
                ChargeType charge; /* charge of the macro particle */
                WeightingType weighting; /* weighting of the macro particle */
            };

            static pmacc::traits::StringProperty getStringProperties()
            {
                pmacc::traits::StringProperty propList("name", "other");
                propList["param"] = "reduced Landau-Lifshitz pusher via RK4 and "
                                    "classical radiation reaction, Marija Vranic (2015)";
                return propList;
            }
        };
    } // namespace particlePusherReducedLandauLifshitz
} // namespace picongpu
