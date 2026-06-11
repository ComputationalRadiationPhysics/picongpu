/*
 * SPDX-FileCopyrightText: Heiko Burau
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/algorithms/KinEnergy.hpp"
#include "picongpu/defines.hpp"
#include "picongpu/plugins/particleCalorimeter/ParticleCalorimeterFunctors.hpp"
#include "picongpu/plugins/particleCalorimeter/param.hpp"
#include "picongpu/traits/attribute/GetMass.hpp"

#include <pmacc/algorithms/math.hpp>
#include <pmacc/lockstep.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/shared/Allocate.hpp>

namespace picongpu
{
    using namespace pmacc;

    template<typename T_CalorimeterDataBox>
    struct CalorimeterFunctor
    {
        T_CalorimeterDataBox m_calorimeterBox;

        float_X const maxYaw;
        float_X const maxPitch;
        uint32_t const numBinsYaw;
        uint32_t const numBinsPitch;
        int32_t const numBinsEnergy;
        /* depending on `logScale` the energy range is initialized
         * with the logarithmic or the linear value. */
        float_X const minEnergy;
        float_X const maxEnergy;
        bool const logScale;

        float3_X const calorimeterFrameVecX;
        float3_X const calorimeterFrameVecY;
        float3_X const calorimeterFrameVecZ;

        CalorimeterFunctor(
            float_X const maxYaw,
            float_X const maxPitch,
            uint32_t const numBinsYaw,
            uint32_t const numBinsPitch,
            uint32_t const numBinsEnergy,
            float_X const minEnergy,
            float_X const maxEnergy,
            bool const logScale,
            float3_X const calorimeterFrameVecX,
            float3_X const calorimeterFrameVecY,
            float3_X const calorimeterFrameVecZ)
            : maxYaw(maxYaw)
            , maxPitch(maxPitch)
            , numBinsYaw(numBinsYaw)
            , numBinsPitch(numBinsPitch)
            , numBinsEnergy(numBinsEnergy)
            , minEnergy(minEnergy)
            , maxEnergy(maxEnergy)
            , logScale(logScale)
            , calorimeterFrameVecX(calorimeterFrameVecX)
            , calorimeterFrameVecY(calorimeterFrameVecY)
            , calorimeterFrameVecZ(calorimeterFrameVecZ)
        {
        }

        HDINLINE CalorimeterFunctor(CalorimeterFunctor const&) = default;

        HINLINE void setCalorimeterData(T_CalorimeterDataBox const& calorimeterBox)
        {
            this->m_calorimeterBox = calorimeterBox;
        }

        template<typename T_Particle, typename T_Worker>
        DINLINE void operator()(T_Worker const& worker, T_Particle& particle)
        {
            float3_X const mom = particle[momentum_];
            float_X const mom2 = pmacc::math::dot(mom, mom);
            float3_X dirVec = mom * math::rsqrt(mom2);

            /* rotate dirVec into the calorimeter frame. This coordinate transformation
             * is performed by a matrix vector multiplication. */
            using namespace pmacc::math;
            dirVec = float3_X(
                pmacc::math::dot(this->calorimeterFrameVecX, dirVec),
                pmacc::math::dot(this->calorimeterFrameVecY, dirVec),
                pmacc::math::dot(this->calorimeterFrameVecZ, dirVec));

            /* convert dirVec to yaw and pitch */
            float_X const yaw = math::atan2(dirVec.x(), dirVec.y());
            float_X const pitch = math::asin(dirVec.z());

            if(math::abs(yaw) < this->maxYaw && math::abs(pitch) < this->maxPitch)
            {
                float2_X const calorimeterPos
                    = particleCalorimeter::mapYawPitchToNormedRange(yaw, pitch, this->maxYaw, this->maxPitch);

                // yaw
                int32_t yawBin = calorimeterPos.x() * static_cast<float_X>(numBinsYaw);
                // catch out-of-range values
                yawBin = yawBin >= static_cast<int32_t>(numBinsYaw) ? numBinsYaw - 1 : yawBin;
                yawBin = yawBin < 0 ? 0 : yawBin;

                // pitch
                int32_t pitchBin = calorimeterPos.y() * static_cast<float_X>(numBinsPitch);
                // catch out-of-range values
                pitchBin = pitchBin >= static_cast<int32_t>(numBinsPitch) ? numBinsPitch - 1 : pitchBin;
                pitchBin = pitchBin < 0 ? 0 : pitchBin;

                // energy
                float_X const weighting = particle[weighting_];
                float_X const normedWeighting
                    = weighting / static_cast<float_X>(sim.unit.typicalNumParticlesPerMacroParticle());
                float_X const mass = picongpu::traits::attribute::getMass(weighting, particle);
                float_X const energy = KinEnergy<>()(mom, mass) / weighting;

                int32_t energyBin = 0;
                if(this->numBinsEnergy > 1)
                {
                    int32_t const numBinsOutOfRange = 2;
                    energyBin
                        = pmacc::math::float2int_rd(
                              ((logScale ? pmacc::math::log10(energy) : energy) - minEnergy) / (maxEnergy - minEnergy)
                              * static_cast<float_X>(this->numBinsEnergy - numBinsOutOfRange))
                          + 1;

                    // all entries larger than maxEnergy go into last bin
                    energyBin = energyBin < this->numBinsEnergy ? energyBin : this->numBinsEnergy - 1;

                    // all entries smaller than minEnergy go into bin zero
                    energyBin = energyBin > 0 ? energyBin : 0;
                }

                alpaka::atomicAdd(
                    worker.getAcc(),
                    &this->m_calorimeterBox(DataSpace<DIM3>(yawBin, pitchBin, energyBin)),
                    energy * normedWeighting,
                    ::alpaka::hierarchy::Threads{});
            }
        }
    };

} // namespace picongpu
