/**
 * Copyright 2015 Alexander Grund
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "particles/ExchangeParticles.hpp"
#include "particles/DeleteParticles.hpp"

namespace PMacc {
namespace particles {

    /**
     * Property struct that exposes policies for handling particles in guard cells
     *
     * All policies are functors with signature void(Particles&, int32_t direction)
     *
     * @tparam T_HandleSentParticles Policy for handling particles that should be sent
     *         to a neighboring rank
     * @tparam T_HandleLeavingParticles Policy for handling particles that are not sent
     *         to any other rank, which means they are leaving the simulation volume
     */
    template<
    class T_HandleSentParticles = ExchangeParticles,
    class T_HandleLeavingParticles = DeleteParticles
    >
    struct HandleGuardParticles
    {
        typedef T_HandleSentParticles HandleSentParticles;
        typedef T_HandleLeavingParticles HandleLeavingParticles;
    };

}  // namespace particles
}  // namespace PMacc
