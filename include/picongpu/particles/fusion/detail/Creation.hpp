/* Copyright 2025-2025 Filip Optolowicz
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

// PIConGPU Includes
#include "picongpu/defines.hpp"
#include "picongpu/particles/fusion/param.hpp"
#include "picongpu/traits/frame/GetCharge.hpp"
#include "picongpu/traits/frame/GetMass.hpp"
#include "picongpu/unitless/simulation.unitless"

// PMacc Includes
#include <pmacc/lockstep.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/math/functions/Round.hpp>
#include <pmacc/math/operation.hpp>
#include <pmacc/memory/shared/Allocate.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/mpi/reduceMethods/Reduce.hpp>
#include <pmacc/particles/algorithm/ForEach.hpp>
#include <pmacc/random/RNGProvider.hpp>
#include <pmacc/random/distributions/Uniform.hpp>

// Standard Library Includes
#include <cmath>
#include <cstddef>
#include <limits>

#if (BOOST_LANG_CUDA || BOOST_COMP_HIP)
#    include <mallocMC/mallocMC.hpp>
#endif

namespace picongpu::particles::fusion
{
    namespace detail
    {
        using namespace precision;

        /**
         * @brief constexpr implementation used instead of std::round
         * @param x value to round
         * @return rounded value
         */
        template<typename T>
        constexpr T round_cx(T x)
        {
            T truncated = static_cast<long long>(x);
            T fraction = x - truncated;

            if(fraction >= T{0.5})
            {
                return truncated + T{1.0};
            }
            if(fraction <= T{-0.5})
            {
                return truncated - T{1.0};
            }
            return truncated;
        }

        // Cap-aware validator allowing W > 1 up to caps
        constexpr bool areWeightsValidWithCaps(
            float_COLL W1,
            float_COLL W2,
            float_COLL W3,
            float_COLL W4,
            float_COLL c3,
            float_COLL c4)
        {
            // clang-format off
            return (W1 >= 0.0_COLL && W1 <= c3 &&
                    W3 >= 0.0_COLL && W3 <= c3 &&
                    W2 >= 0.0_COLL && W2 <= c4 &&
                    W4 >= 0.0_COLL && W4 <= c4);
            // clang-format on
        }

        /** Result structure for weight calculations */
        struct WeightResult
        {
            float_COLL W1, W2, W3, W4;
            bool isValid;
        };

        // New: stoichiometric caps for products 1 and 2
        struct StoichiometryCaps
        {
            float_COLL c3; // total multiplicity for product 1 (split across sites)
            float_COLL c4; // total multiplicity for product 2 (split across sites)
            bool isValid;
        };

        // New: compute stoichiometric caps (c3, c4) from global charge and mass conservation
        constexpr StoichiometryCaps computeStoichiometryCaps(
            float_COLL q1,
            float_COLL m1,
            float_COLL q2,
            float_COLL m2,
            float_COLL q3,
            float_COLL m3,
            float_COLL q4,
            float_COLL m4)
        {
            constexpr float_COLL tolerance = 10.0_COLL * std::numeric_limits<float_COLL>::epsilon();

            float_COLL const Q = q1 + q2;
            float_COLL const M = m1 + m2;

            // Solve [q3 q4; m3 m4] [c3 c4]^T = [Q M]^T
            float_COLL const det = q3 * m4 - q4 * m3;
            if(std::abs(det) > tolerance)
            {
                float_COLL const c3 = (Q * m4 - q4 * M) / det;
                float_COLL const c4 = (q3 * M - Q * m3) / det;
                bool const ok = (c3 >= 0.0_COLL && c4 >= 0.0_COLL);
                return StoichiometryCaps{c3, c4, ok};
            }

            // Degenerate q/m ratio: use symmetric caps across products
            // Prefer charge-based symmetry when charges are defined; fall back to mass-based symmetry otherwise.
            float_COLL const sQ = q3 + q4;
            if(std::abs(sQ) > tolerance)
            {
                float_COLL const c = Q / sQ; // c3=c4 ensures global charge; mass follows when q/m are identical
                return StoichiometryCaps{c, c, c >= 0.0_COLL};
            }

            // If both products are effectively neutral, rely on mass-only symmetric split
            float_COLL const sM = m3 + m4;
            if(std::abs(sM) > tolerance)
            {
                float_COLL const c = M / sM;
                return StoichiometryCaps{c, c, c >= 0.0_COLL};
            }
            return StoichiometryCaps{0.0_COLL, 0.0_COLL, false};
        }

        // New: local mass+charge solver under caps. Solve site-1 2x2 and derive site-2 via caps.
        constexpr WeightResult calculateMassChargeConservingWeightsWithCaps(
            float_COLL q1,
            float_COLL m1,
            float_COLL q2,
            float_COLL m2,
            float_COLL q3,
            float_COLL m3,
            float_COLL q4,
            float_COLL m4,
            float_COLL c3,
            float_COLL c4)
        {
            constexpr float_COLL tolerance = 10.0_COLL * std::numeric_limits<float_COLL>::epsilon();

            float_COLL const det = q3 * m4 - q4 * m3;
            if(std::abs(det) < tolerance)
                return WeightResult{0.0_COLL, 0.0_COLL, 0.0_COLL, 0.0_COLL, false};

            // Solve for site-1 weights (W1,W2)
            float_COLL const W1 = (q1 * m4 - q4 * m1) / det;
            float_COLL const W2 = (q3 * m1 - q1 * m3) / det;
            float_COLL const W3 = c3 - W1;
            float_COLL const W4 = c4 - W2;

            // Check bounds
            if(!areWeightsValidWithCaps(W1, W2, W3, W4, c3, c4))
                return WeightResult{0.0_COLL, 0.0_COLL, 0.0_COLL, 0.0_COLL, false};

            // Verify site-2 consistency
            float_COLL const q2_chk = W3 * q3 + W4 * q4;
            float_COLL const m2_chk = W3 * m3 + W4 * m4;
            bool const ok2 = (std::abs(q2 - q2_chk) < tolerance) && (std::abs(m2 - m2_chk) < tolerance);
            return WeightResult{W1, W2, W3, W4, ok2};
        }

        // New: charge-only solver under caps (robust, neutral-aware, minimal macros when possible)
        constexpr WeightResult calculateChargeOnlyWithCaps(
            float_COLL q1,
            float_COLL q2,
            float_COLL q3,
            float_COLL q4,
            float_COLL c3,
            float_COLL c4)
        {
            constexpr float_COLL tolerance = 10.0_COLL * std::numeric_limits<float_COLL>::epsilon();

            // Both neutral: minimize macro creation -> put all at site 2
            if(std::abs(q3) < tolerance && std::abs(q4) < tolerance)
            {
                return WeightResult{0.0_COLL, 0.0_COLL, c3, c4, true};
            }

            // Second product neutral
            if(std::abs(q4) < tolerance)
            {
                float_COLL const W1 = q1 / q3;
                float_COLL const W3 = c3 - W1;
                // Place all neutral at site 2
                float_COLL const W2 = 0.0_COLL;
                float_COLL const W4 = c4;
                bool const ok = areWeightsValidWithCaps(W1, W2, W3, W4, c3, c4)
                                && (std::abs(q1 - (W1 * q3 + W2 * q4)) < tolerance);
                return WeightResult{W1, W2, W3, W4, ok};
            }

            // First product neutral
            if(std::abs(q3) < tolerance)
            {
                float_COLL const W2 = q1 / q4;
                float_COLL const W4 = c4 - W2;
                float_COLL const W1 = 0.0_COLL;
                float_COLL const W3 = c3;
                bool const ok = areWeightsValidWithCaps(W1, W2, W3, W4, c3, c4)
                                && (std::abs(q1 - (W1 * q3 + W2 * q4)) < tolerance);
                return WeightResult{W1, W2, W3, W4, ok};
            }

            // Both charged: try single-product split first
            {
                float_COLL const W1a = q1 / q3;
                if(W1a >= 0.0_COLL && W1a <= c3)
                {
                    float_COLL const W1 = W1a;
                    float_COLL const W2 = 0.0_COLL;
                    float_COLL const W3 = c3 - W1;
                    float_COLL const W4 = c4;
                    return WeightResult{W1, W2, W3, W4, true};
                }
                float_COLL const W2a = q1 / q4;
                if(W2a >= 0.0_COLL && W2a <= c4)
                {
                    float_COLL const W1 = 0.0_COLL;
                    float_COLL const W2 = W2a;
                    float_COLL const W3 = c3;
                    float_COLL const W4 = c4 - W2;
                    return WeightResult{W1, W2, W3, W4, true};
                }
            }

            // General case: set one at bound and solve the other
            // If q1 is large relative to q3, fill W1 to c3 and use W2 for the rest
            if(q1 > q3 * c3)
            {
                float_COLL const W1 = c3;
                float_COLL const W2 = (q1 - q3 * W1) / q4;
                float_COLL const W3 = 0.0_COLL;
                float_COLL const W4 = c4 - W2;
                bool const ok = areWeightsValidWithCaps(W1, W2, W3, W4, c3, c4);
                return WeightResult{W1, W2, W3, W4, ok};
            }
            // Otherwise, fill W2 to c4 and use W1 for the rest
            {
                float_COLL const W2 = c4;
                float_COLL const W1 = (q1 - q4 * W2) / q3;
                float_COLL const W3 = c3 - W1;
                float_COLL const W4 = 0.0_COLL;
                bool const ok = areWeightsValidWithCaps(W1, W2, W3, W4, c3, c4);
                return WeightResult{W1, W2, W3, W4, ok};
            }
        }

        template<
            typename T_Reactant1ParBox,
            typename T_Reactant2ParBox,
            typename T_Product1ParBox,
            typename T_Product2ParBox>
        struct CreationFusion
        {
            // Extract mass and charge ratios directly from the frame types at compile-time
            using MassRatio1 = typename pmacc::traits::Resolve<
                typename pmacc::traits::GetFlagType<typename T_Reactant1ParBox::FrameType, massRatio<>>::type>::type;
            using ChargeRatio1 = typename pmacc::traits::Resolve<
                typename pmacc::traits::GetFlagType<typename T_Reactant1ParBox::FrameType, chargeRatio<>>::type>::type;
            using MassRatio2 = typename pmacc::traits::Resolve<
                typename pmacc::traits::GetFlagType<typename T_Reactant2ParBox::FrameType, massRatio<>>::type>::type;
            using ChargeRatio2 = typename pmacc::traits::Resolve<
                typename pmacc::traits::GetFlagType<typename T_Reactant2ParBox::FrameType, chargeRatio<>>::type>::type;
            using MassRatio3 = typename pmacc::traits::Resolve<
                typename pmacc::traits::GetFlagType<typename T_Product1ParBox::FrameType, massRatio<>>::type>::type;
            using ChargeRatio3 = typename pmacc::traits::Resolve<
                typename pmacc::traits::GetFlagType<typename T_Product1ParBox::FrameType, chargeRatio<>>::type>::type;
            using MassRatio4 = typename pmacc::traits::Resolve<
                typename pmacc::traits::GetFlagType<typename T_Product2ParBox::FrameType, massRatio<>>::type>::type;
            using ChargeRatio4 = typename pmacc::traits::Resolve<
                typename pmacc::traits::GetFlagType<typename T_Product2ParBox::FrameType, chargeRatio<>>::type>::type;

            // Conversion factor from ratio of atomic unit to Electron mass to atomic mass units (u)
            static constexpr float_COLL invAmu = sim.fusion.electronMassAMU;

            // Calculate compile-time mass and charge values using ratios
            // We can use ratios here because the constants will cancel out in the conservation equations
            static constexpr float_COLL m1_u = MassRatio1::getValue() * invAmu;
            static constexpr float_COLL m2_u = MassRatio2::getValue() * invAmu;
            static constexpr float_COLL m3_u = MassRatio3::getValue() * invAmu;
            static constexpr float_COLL m4_u = MassRatio4::getValue() * invAmu;

            // Round to the nearest integer to get the mass number (A)
            static constexpr float_COLL m1 = round_cx(m1_u);
            static constexpr float_COLL q1 = ChargeRatio1::getValue();
            static constexpr float_COLL m2 = round_cx(m2_u);
            static constexpr float_COLL q2 = ChargeRatio2::getValue();
            static constexpr float_COLL m3 = round_cx(m3_u);
            static constexpr float_COLL q3 = ChargeRatio3::getValue();
            static constexpr float_COLL m4 = round_cx(m4_u);
            static constexpr float_COLL q4 = ChargeRatio4::getValue();

            template<
                typename T_Worker,
                typename T_ParAccessor0,
                typename T_ParAccessor1,
                typename T_ParAccessor2,
                typename T_ParAccessor3,
                typename T_RngHandle>
            DINLINE void createParticles(
                T_Worker const& worker,
                IdGenerator& idGen,
                T_ParAccessor0 const& r1,
                T_ParAccessor1 const& r2,
                float_X const productWeighting,
                float3_X const& mom1,
                float3_X const& mom2,
                T_ParAccessor2& p1r1, // product 1 at pos 1
                T_ParAccessor2& p1r2, // product 1 at pos 2
                T_ParAccessor3& p2r1, // product 2 at pos 1
                T_ParAccessor3& p2r2, // product 2 at pos 2
                T_RngHandle& rngHandle) const
            {
                namespace partOp = pmacc::particles::operations;

                // Initialize particle clones
                auto targetClone2 = partOp::deselect<pmacc::mp_list<multiMask, momentum, weighting>>(p1r1);
                auto targetClone3 = partOp::deselect<pmacc::mp_list<multiMask, momentum, weighting>>(p1r2);
                auto targetClone4 = partOp::deselect<pmacc::mp_list<multiMask, momentum, weighting>>(p2r1);
                auto targetClone5 = partOp::deselect<pmacc::mp_list<multiMask, momentum, weighting>>(p2r2);

                targetClone2.derive(worker, idGen, r1);
                targetClone3.derive(worker, idGen, r2);
                targetClone4.derive(worker, idGen, r1);
                targetClone5.derive(worker, idGen, r2);


                static constexpr float_COLL tolerance = 10.0_COLL * std::numeric_limits<float_COLL>::epsilon();
                // Charges must be non-negative
                static_assert(
                    q1 >= 0.0_COLL && q2 >= 0.0_COLL && q3 >= 0.0_COLL && q4 >= 0.0_COLL,
                    "All charges must be non-negative");

                // Compute stoichiometric caps from species (can be fractional)
                constexpr auto caps = computeStoichiometryCaps(q1, m1, q2, m2, q3, m3, q4, m4);
                static_assert(caps.isValid, "Stoichiometry caps invalid for given species (check mass/charge)");
                constexpr float_COLL c3 = caps.c3;
                constexpr float_COLL c4 = caps.c4;

                // Try local mass+charge conservation with caps, otherwise fall back to charge-only
                constexpr auto massChargeCaps
                    = calculateMassChargeConservingWeightsWithCaps(q1, m1, q2, m2, q3, m3, q4, m4, c3, c4);
                constexpr auto chargeOnlyCaps = calculateChargeOnlyWithCaps(q1, q2, q3, q4, c3, c4);

                constexpr float_COLL W1 = massChargeCaps.isValid ? massChargeCaps.W1 : chargeOnlyCaps.W1;
                constexpr float_COLL W2 = massChargeCaps.isValid ? massChargeCaps.W2 : chargeOnlyCaps.W2;
                constexpr float_COLL W3 = massChargeCaps.isValid ? massChargeCaps.W3 : chargeOnlyCaps.W3;
                constexpr float_COLL W4 = massChargeCaps.isValid ? massChargeCaps.W4 : chargeOnlyCaps.W4;

                // Weight conservation under caps
                constexpr float_COLL W1_plus_W3 = W1 + W3;
                constexpr float_COLL W2_plus_W4 = W2 + W4;

                // Valid weight ranges under caps
                constexpr bool weightsValid = areWeightsValidWithCaps(W1, W2, W3, W4, c3, c4);

                // Charge conservation per site
                constexpr float_COLL q1_check = W1 * q3 + W2 * q4;
                constexpr float_COLL q2_check = W3 * q3 + W4 * q4;
                constexpr float_COLL q1_diff = q1 - q1_check;
                constexpr float_COLL q2_diff = q2 - q2_check;

                // Compile-time checks
                static_assert(weightsValid, "Calculated weights violate caps (negative or exceed caps)");
                static_assert(std::abs(W1_plus_W3 - c3) < tolerance, "Weight conservation failed for product 1 (cap)");
                static_assert(std::abs(W2_plus_W4 - c4) < tolerance, "Weight conservation failed for product 2 (cap)");
                static_assert(std::abs(q1_diff) < tolerance, "Charge conservation failed for reactant 1 (local)");
                static_assert(std::abs(q2_diff) < tolerance, "Charge conservation failed for reactant 2 (local)");

                // Assign multiMask to indicate these are product particles
                p1r1[multiMask_] = (W1 > tolerance) ? 1u : 0u;
                p1r2[multiMask_] = (W3 > tolerance) ? 1u : 0u;
                p2r1[multiMask_] = (W2 > tolerance) ? 1u : 0u;
                p2r2[multiMask_] = (W4 > tolerance) ? 1u : 0u;

                // Assign momentum (weighted with weights) and weights to product particles
                // At reactant 1 position
                p1r1[weighting_] = W1 * productWeighting;
                p1r1[momentum_] = mom1 * W1 * productWeighting;
                p2r1[weighting_] = W2 * productWeighting;
                p2r1[momentum_] = mom2 * W2 * productWeighting;

                // At reactant 2 position
                p1r2[weighting_] = W3 * productWeighting;
                p1r2[momentum_] = mom1 * W3 * productWeighting;
                p2r2[weighting_] = W4 * productWeighting;
                p2r2[momentum_] = mom2 * W4 * productWeighting;

                // Debug prints: species, caps and weights
                if constexpr(debugFusion)
                {
                    using UniformFloat = pmacc::random::distributions::Uniform<
                        pmacc::random::distributions::uniform::ExcludeOne<precision::float_COLL>::Reduced>;
                    auto rng = rngHandle.template applyDistribution<UniformFloat>();
                    if(worker.workerIdx() == 0 && rng(worker) < 1e-6)
                    {
                        printf("Charges: %f, %f, %f, %f\n", q1, q2, q3, q4);
                        printf("Masses (A): %f, %f, %f, %f\n", m1, m2, m3, m4);
                        printf("Caps (c3,c4): %f, %f\n", c3, c4);
                        printf("Weights: %f, %f, %f, %f\n", W1, W2, W3, W4);

                        printf(
                            "Checks: W1+W3=%f (c3=%f), W2+W4=%f (c4=%f), q1_check=%f, q2_check=%f\n",
                            W1_plus_W3,
                            c3,
                            W2_plus_W4,
                            c4,
                            q1_check,
                            q2_check);
                        printf(
                            "Diffs: W1+W3-c3=%e, W2+W4-c4=%e, q1_diff=%e, q2_diff=%e\n",
                            W1_plus_W3 - c3,
                            W2_plus_W4 - c4,
                            q1_diff,
                            q2_diff);
                    }
                }
            }
        };

    } // namespace detail
} // namespace picongpu::particles::fusion
