/* Copyright 2019-2021 Rene Widera, Pawel Ordyna
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

namespace picongpu
{
    namespace particles
    {
        namespace collision
        {
            namespace detail
            {
                template<typename T_Acc, typename T_RngHandle>
                struct CollisionContext
                {
                    T_Acc const* m_acc;
                    mutable T_RngHandle* m_hRng;

                    DINLINE CollisionContext(T_Acc const& acc, T_RngHandle& hRng) : m_acc(&acc), m_hRng(&hRng)
                    {
                    }
                };

                template<typename T_Acc, typename T_RngHandle>
                DINLINE CollisionContext<T_Acc, T_RngHandle> makeCollisionContext(T_Acc const& acc, T_RngHandle& hRng)
                {
                    return CollisionContext<T_Acc, T_RngHandle>(acc, hRng);
                }

            } // namespace detail
        } // namespace collision
    } // namespace particles
} // namespace picongpu
