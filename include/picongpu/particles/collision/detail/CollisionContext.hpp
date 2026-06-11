/*
 * SPDX-FileCopyrightText: Rene Widera, Pawel Ordyna
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

namespace picongpu
{
    namespace particles
    {
        namespace collision
        {
            namespace detail
            {
                template<typename T_Worker, typename T_RngHandle>
                struct CollisionContext
                {
                    T_Worker const* m_worker;
                    mutable T_RngHandle* m_hRng;

                    DINLINE CollisionContext(T_Worker const& worker, T_RngHandle& hRng)
                        : m_worker(&worker)
                        , m_hRng(&hRng)
                    {
                    }
                };

                template<typename T_Worker, typename T_RngHandle>
                DINLINE auto makeCollisionContext(T_Worker const& worker, T_RngHandle& hRng)
                {
                    return CollisionContext<T_Worker, T_RngHandle>(worker, hRng);
                }

            } // namespace detail
        } // namespace collision
    } // namespace particles
} // namespace picongpu
