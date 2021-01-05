/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch,
 *                     Benjamin Worpitz
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


// That is a sum over two out of 3 coordinates, as described in the script
// above. (See Ref.!)
#define FOR_JK_NOT_I(I, J, K, code) (code(I, J, K)) + (code(I, K, J))

#include <pmacc/types.hpp>

namespace picongpu
{
    namespace particlePusherAxel
    {
        template<class Velocity, class Gamma>
        struct Push
        {
            /* this is an optional extension for sub-sampling pushes that enables grid to particle interpolation
             * for particle positions outside the super cell in one push
             */
            using LowerMargin = typename pmacc::math::CT::make_Int<simDim, 0>::type;
            using UpperMargin = typename pmacc::math::CT::make_Int<simDim, 0>::type;

            enum coords
            {
                x = 0,
                y = 1,
                z = 2
            };

            HDINLINE float_X levichivita(const unsigned int i, const unsigned int j, const unsigned int k)
            {
                if(i == j || j == k || i == k)
                    return float_X(0.0);

                if(i == x && j == y)
                    return float_X(1.0);
                if(i == z && j == x)
                    return float_X(1.0);
                if(i == y && j == z)
                    return float_X(1.0);

                return float_X(-1.0);
            }

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

                using MomType = momentum::type;
                MomType mom = particle[momentum_];

                auto bField = functorBField(pos);
                auto eField = functorEField(pos);

                Gamma gammaCalc;
                Velocity velocityCalc;
                const float_X epsilon = 1.0e-6;
                const float_X deltaT = DELTA_T;

                // const float3_X velocity_atMinusHalf = velocity(mom, mass);
                const float_X gamma = gammaCalc(mom, mass);

                const MomType mom_old = mom;

                const float_X B2 = pmacc::math::abs2(bField);
                const float_X B = math::abs(bField);

                if(B2 > epsilon)
                {
                    trigo_X sinres;
                    trigo_X cosres;
                    trigo_X arg = B * charge * deltaT / gamma;
                    pmacc::math::sincos(arg, sinres, cosres);

                    mom.x() = bField.x() * bField.x() * (eField.x() * charge * deltaT + mom_old.x());
                    mom.y() = bField.y() * bField.y() * (eField.y() * charge * deltaT + mom_old.y());
                    mom.z() = bField.z() * bField.z() * (eField.z() * charge * deltaT + mom_old.z());

#define SUM_PLINE1(I, J, K)                                                                                           \
    bField.J()                                                                                                        \
        * (-levichivita(I, J, K) * gamma * eField.K() + bField.I() * (eField.J() * charge * deltaT + mom_old.J()))
#define SUM_PLINE2(I, J, K)                                                                                           \
    -bField.J() * (-levichivita(I, J, K) * gamma * eField.K() + bField.I() * mom_old.J() - bField.J() * mom_old.I())
#define SUM_PLINE3(I, J, K)                                                                                           \
    bField.J() * bField.J() * gamma* eField.I() - bField.I() * bField.J() * gamma* eField.J()                         \
        + levichivita(I, J, K) * mom_old.J() * bField.K() * B2

                    mom.x() += FOR_JK_NOT_I(x, y, z, SUM_PLINE1);
                    mom.x() += float_X(cosres) * (FOR_JK_NOT_I(x, y, z, SUM_PLINE2));
                    mom.x() += float_X(sinres) / B * (FOR_JK_NOT_I(x, y, z, SUM_PLINE3));

                    mom.y() += FOR_JK_NOT_I(y, z, x, SUM_PLINE1);
                    mom.y() += float_X(cosres) * (FOR_JK_NOT_I(y, z, x, SUM_PLINE2));
                    mom.y() += float_X(sinres) / B * (FOR_JK_NOT_I(y, z, x, SUM_PLINE3));

                    mom.z() += FOR_JK_NOT_I(z, x, y, SUM_PLINE1);
                    mom.z() += float_X(cosres) * (FOR_JK_NOT_I(z, x, y, SUM_PLINE2));
                    mom.z() += float_X(sinres) / B * (FOR_JK_NOT_I(z, x, y, SUM_PLINE3));

                    mom *= float_X(1.0) / B2;
                }
                else
                {
                    mom += eField * charge * deltaT;
                }

                particle[momentum_] = mom;

                float3_X dr(float3_X::create(0.0));

                // old spacial change calculation: linear step
                if(TrajectoryInterpolation == LINEAR)
                {
                    const float3_X vel = velocityCalc(mom, mass);
                    dr = float3_X(
                        vel.x() * deltaT / CELL_WIDTH,
                        vel.y() * deltaT / CELL_HEIGHT,
                        vel.z() * deltaT / CELL_DEPTH);
                }

                // new spacial change calculation
                if(TrajectoryInterpolation == NONLINEAR)
                {
                    const float3_X vel_old = velocityCalc(mom_old, mass);
                    const float_X QoM = charge / mass;
                    const float_X B4 = B2 * B2;
                    float3_X r = pos;

                    if(B4 > epsilon)
                    {
                        trigo_X sinres;
                        trigo_X cosres;
                        trigo_X arg = B * QoM * deltaT / SPEED_OF_LIGHT;
                        pmacc::math::sincos(arg, sinres, cosres);

                        r.x() = bField.x() * bField.x() * bField.x() * bField.x() * QoM
                            * (eField.x() * QoM * deltaT * deltaT + 2.0f * (deltaT * vel_old.x() + pos.x()));

#define SUM_RLINE1(I, J, K)                                                                                           \
    2.0 * bField.J() * bField.J() * bField.J() * bField.J() * QoM* pos.x()                                            \
        + 2.0 * bField.J() * bField.J() * bField.K() * bField.K() * QoM* pos.x()                                      \
        + bField.J() * bField.J() * bField.J()                                                                        \
            * (-levichivita(I, J, K) * 2.0 * SPEED_OF_LIGHT * (eField.K() * QoM * deltaT + vel_old.K())               \
               + bField.I() * QoM * deltaT * (eField.J() * QoM * deltaT + 2.0 * vel_old.J()))                         \
        + bField.J() * bField.J()                                                                                     \
            * (2.0 * SPEED_OF_LIGHT * SPEED_OF_LIGHT * eField.I()                                                     \
               + bField.I() * bField.I() * QoM                                                                        \
                   * (eField.I() * QoM * deltaT * deltaT + 2.0 * deltaT * vel_old.I() + 4.0 * pos.I())                \
               + levichivita(I, J, K) * 2.0 * SPEED_OF_LIGHT * bField.K() * vel_old.J()                               \
               + bField.K() * QoM                                                                                     \
                   * (levichivita(I, J, K) * 2.0 * eField.J() * SPEED_OF_LIGHT * deltaT                               \
                      + bField.I() * bField.K() * QoM * deltaT * deltaT))                                             \
        + bField.I() * bField.J()                                                                                     \
            * (bField.I() * bField.I() * QoM * deltaT * (eField.J() * QoM * deltaT + 2.0 * vel_old.J())               \
               - levichivita(I, J, K) * 2.0 * bField.I() * SPEED_OF_LIGHT * (eField.K() * QoM * deltaT + vel_old.K()) \
               - 2.0 * SPEED_OF_LIGHT * SPEED_OF_LIGHT * eField.J())

#define SUM_RLINE2(I, J, K)                                                                                           \
    -bField.J()                                                                                                       \
        * (SPEED_OF_LIGHT * eField.I() * bField.J() - levichivita(I, J, K) * bField.J() * bField.J() * vel_old.K()    \
           - bField.I() * SPEED_OF_LIGHT * eField.J()                                                                 \
           - levichivita(I, J, K) * bField.J() * vel_old.K() * (bField.I() * bField.I() + bField.K() * bField.K()))

#define SUM_RLINE3(I, J, K)                                                                                           \
    levichivita(I, J, K) * bField.J()                                                                                 \
        * (SPEED_OF_LIGHT * eField.K()                                                                                \
           + levichivita(I, J, K) * (bField.J() * vel_old.I() - bField.I() * vel_old.J()))

                        r.x() += FOR_JK_NOT_I(x, y, z, SUM_RLINE1);
                        r.x() += float_X(cosres) * 2.0 * SPEED_OF_LIGHT * (FOR_JK_NOT_I(x, y, z, SUM_RLINE2));
                        r.x() += float_X(sinres) * 2.0 * SPEED_OF_LIGHT * B * (FOR_JK_NOT_I(x, y, z, SUM_RLINE3));

                        r.y() += FOR_JK_NOT_I(y, z, x, SUM_RLINE1);
                        r.y() += float_X(cosres) * 2.0 * SPEED_OF_LIGHT * (FOR_JK_NOT_I(y, z, x, SUM_RLINE2));
                        r.y() += float_X(sinres) * 2.0 * SPEED_OF_LIGHT * B * (FOR_JK_NOT_I(y, z, x, SUM_RLINE3));

                        r.z() += FOR_JK_NOT_I(z, x, y, SUM_RLINE1);
                        r.z() += float_X(cosres) * 2.0 * SPEED_OF_LIGHT * (FOR_JK_NOT_I(z, x, y, SUM_RLINE2));
                        r.z() += float_X(sinres) * 2.0 * SPEED_OF_LIGHT * B * (FOR_JK_NOT_I(z, x, y, SUM_RLINE3));

                        r *= float_X(0.5) / B4 / QoM;
                    }
                    else
                    {
                        r += eField * QoM * deltaT * deltaT + vel_old * deltaT;
                    }
                    dr = r - pos;

                    dr *= float3_X::create(1.0) / cellSize;
                }

                pos += dr;
            }

            static pmacc::traits::StringProperty getStringProperties()
            {
                pmacc::traits::StringProperty propList("name", "other");
                propList["param"] = "semi analytical, Axel Huebl (2011)";
                return propList;
            }
        };
    } // namespace particlePusherAxel
} // namespace picongpu
