/* Copyright 2016-2017 Heiko Burau
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

#include <pmacc/particles/policies/ExchangeParticles.hpp>
#include <pmacc/pluginSystem/IPlugin.hpp>
#include <list>

namespace picongpu
{

/**
 * Guard handler policy calling all registered plugins when particles
 * leave the global simulation volume. This class serves as policy for
 * the `ParticleDescription` template class.
 *
 * For each plugin the method `IPlugin::onParticleLeave()` is called.
 * After that the guard particles are deleted.
 */
struct GuardHandlerCallPlugins
{
    typedef pmacc::particles::policies::ExchangeParticles HandleExchanged;
    typedef GuardHandlerCallPlugins HandleNotExchanged;

    template< class T_Particles >
    void
    handleOutgoing(T_Particles& particles, const int32_t direction) const
    {
        typedef std::list<pmacc::IPlugin*> Plugins;
        Plugins plugins = Environment<>::get().PluginConnector().getAllPlugins();

        for(Plugins::iterator iter = plugins.begin(); iter != plugins.end(); iter++)
        {
            (*iter)->onParticleLeave(T_Particles::FrameType::getName(), direction);
        }

        particles.deleteGuardParticles(direction);
    }

    template< class T_Particles >
    void
    handleIncoming(T_Particles&, const int32_t) const
    {}
};

} // namespace picongpu
