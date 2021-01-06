/* Copyright 2020-2021 Pawel Ordyna
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
#include "picongpu/plugins/xrayScattering/beam/AxisSwap.hpp"

namespace picongpu
{
    namespace plugins
    {
        namespace xrayScattering
        {
            namespace beam
            {
                /* This file defines the possible base beam orientations.
                 *
                 *  Example: X Side
                 *      The beam propagates along the x axis ( PIC coordinate system).
                 *      The base position of the beam coordinate system (0,0,0) point it
                 *      the beam system is placed at in the middle of the x_PIC=0 plane.
                 *      That is at (0, 0.5 * Y, 0.5 * Z), where Y and Z are the lengths of
                 *      the simulation box sides along y_PIC and z_PIC axes.
                 *      Therefore beamStartPosition= ( 0.0, 0.5, 0.5 ) for the XSide.
                 *
                 *      AxisSwap defines the base rotation of the
                 *      coordinate system. First three integers set how the 3 directions
                 *      (x, y, z) in the PIC system correspond to the ones in the beam
                 *      system. The last 3 numbers are the relative orientations. For XSide:
                 *      AxisSwap< 2, 1, 0, -1, 1, 1 > says:
                 *          * x_beam = - z_PIC,
                 *          * y_beam = y_PIC,
                 *          * z_beam = x_PIC,
                 */

                //! Probing along the PIC x basis vector.
                struct XSide
                {
                    static constexpr float_X beamStartPosition[3] = {0.0, 0.5, 0.5};
                    using FirstRotation = AxisSwap<2, 1, 0, -1, 1, 1>;
                };


                //! Probing against the PIC x basis vector.
                struct XRSide
                {
                    static constexpr float_X beamStartPosition[3] = {1.0, 0.5, 0.5};
                    using FirstRotation = AxisSwap<2, 1, 0, -1, -1, -1>;
                };


                //! Probing along the PIC y basis vector.
                struct YSide
                {
                    static constexpr float_X beamStartPosition[3] = {0.5, 0.0, 0.5};
                    using FirstRotation = AxisSwap<2, 0, 1, -1, -1, 1>;
                };


                //! Probing against the PIC y basis vector.
                struct YRSide
                {
                    static constexpr float_X beamStartPosition[3] = {0.5, 1.0, 0.5};
                    using FirstRotation = AxisSwap<2, 0, 1, -1, 1, -1>;
                };


                //! Probing along the PIC z basis vector.
                struct ZSide
                {
                    static constexpr float_X beamStartPosition[3] = {0.5, 0.5, 0.0};
                    using FirstRotation = AxisSwap<1, 0, 2, -1, 1, 1>;
                };


                //! Probing against the PIC z basis vector.
                struct ZRSide
                {
                    static constexpr float_X beamStartPosition[3] = {0.5, 0.5, 0.0};
                    using FirstRotation = AxisSwap<1, 0, 2, -1, -1, -1>;
                };

            } // namespace beam
        } // namespace xrayScattering
    } // namespace plugins
} // namespace picongpu
