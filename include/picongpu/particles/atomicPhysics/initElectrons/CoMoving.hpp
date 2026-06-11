/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! @file implemets init of macro electron as co-moving with ion

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/initElectrons/CloneAdditionalAttributes.hpp"
#include "picongpu/traits/frame/GetMass.hpp"

namespace picongpu::particles::atomicPhysics::initElectrons
{
    struct CoMoving
    {
        template<typename T_Worker, typename T_IonParticle, typename T_ElectronParticle>
        HDINLINE static void initElectron(
            T_Worker const& worker,
            // cannot be const even though we do not write to the ion
            T_IonParticle& ion,
            T_ElectronParticle& electron,
            IdGenerator& idGen)
        {
            CloneAdditionalAttributes::init(worker, ion, electron, idGen);

            float_X const massElectronPerMassIon
                = picongpu::traits::frame::getMass<typename T_ElectronParticle::FrameType>()
                  / picongpu::traits::frame::getMass<typename T_IonParticle::FrameType>();

            // init electron as co-moving with ion
            electron[momentum_] = ion[momentum_] * massElectronPerMassIon;
        }
    };
} // namespace picongpu::particles::atomicPhysics::initElectrons
