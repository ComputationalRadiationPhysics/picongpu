/* Copyright 2023 Pawel Ordyna
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
        namespace externalBeam
        {
            namespace detail
            {
                //! Combine worker information with the random generator handle for StartAttributes sub-functors
                template<typename T_Worker, typename T_RngHandle>
                struct StartAttributesContext
                {
                    T_Worker const* m_worker;
                    mutable T_RngHandle* m_hRng;

                    DINLINE StartAttributesContext(T_Worker const& worker, T_RngHandle& hRng)
                        : m_worker(&worker)
                        , m_hRng(&hRng)
                    {
                    }
                };

                //! Get the context without knowing the types
                template<typename T_Worker, typename T_RngHandle>
                DINLINE auto makeContext(T_Worker const& worker, T_RngHandle& hRng)
                {
                    return StartAttributesContext<T_Worker, T_RngHandle>(worker, hRng);
                }

            } // namespace detail
        } // namespace externalBeam
    } // namespace particles
} // namespace picongpu
