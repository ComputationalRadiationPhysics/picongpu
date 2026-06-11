/*
 * SPDX-FileCopyrightText: Sergei Bastrakov, Lennert Sprenger
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

namespace picongpu
{
    namespace particles
    {
        namespace boundary
        {
            //! Supported particle boundary kinds
            enum class Kind
            {
                Periodic,
                Absorbing,
                Reflecting,
                Thermal
            };

        } // namespace boundary
    } // namespace particles
} // namespace picongpu
