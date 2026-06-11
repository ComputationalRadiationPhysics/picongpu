/*
 * SPDX-FileCopyrightText: Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace picongpu
{
    /** 2D Phase Space Selection
     *
     * This structure stores the plot we want to create.
     * e.g. (py | x) from (momentum | spatial-coordinate)
     */
    struct AxisDescription
    {
        /** px, py or pz: @see element_momentum*/
        uint32_t momentum;
        /** x, y or z: @see element_coordinate */
        uint32_t space;

        /** short hand enums */
        enum element_momentum
        {
            px = 0u,
            py = 1u,
            pz = 2u
        };

        enum element_coordinate
        {
            x = 0u,
            y = 1u,
            z = 2u
        };

        std::string momentumAsString() const
        {
            switch(momentum)
            {
            case px:
                return "px";
            case py:
                return "py";
            case pz:
                return "pz";
            default:
                throw std::runtime_error("Unreachable!");
            }
        }

        std::string spaceAsString() const
        {
            switch(space)
            {
            case x:
                return "x";
            case y:
                return "y";
            case z:
                return "z";
            default:
                throw std::runtime_error("Unreachable!");
            }
        }
    };

} /* namespace picongpu */
