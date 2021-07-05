/* Copyright 2021 Pawel Ordyna
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

#include "picongpu/plugins/externalBeam/Side.hpp"
#include "picongpu/plugins/photonDetector/DetectorParams.def"

#include <experimental/optional>

namespace picongpu
{
    namespace plugins
    {
        namespace photonDetector
        {
            //! Get the coordinate transform between simulation and detector systems based on a detector placement
            HINLINE externalBeam::AxisSwap getAxisSwap(DetectorPlacement const& detectorPlacement)
            {
                switch(detectorPlacement)
                {
                case DetectorPlacement::XFront:
                    return externalBeam::ProbingSideCfg<externalBeam::XRSide>().getAxisSwap();
                case DetectorPlacement::XRear:
                    return externalBeam::ProbingSideCfg<externalBeam::XSide>().getAxisSwap();
                case DetectorPlacement::YFront:
                    return externalBeam::ProbingSideCfg<externalBeam::YRSide>().getAxisSwap();
                case DetectorPlacement::YRear:
                    return externalBeam::ProbingSideCfg<externalBeam::YSide>().getAxisSwap();
                case DetectorPlacement::ZFront:
                    return externalBeam::ProbingSideCfg<externalBeam::ZRSide>().getAxisSwap();
                // case DetectorPlacement::ZRear:
                default:
                    return externalBeam::ProbingSideCfg<externalBeam::ZSide>().getAxisSwap();
                }
            }

            //! Get the pic cell dimensions along the detector coordinate system axes based on a detector placement
            HDINLINE constexpr auto getCellLengths(DetectorPlacement const& detectorPlacement)
            {
                using ReturnType = std::tuple<float_X, float_X, float_X>;
                switch(detectorPlacement)
                {
                case DetectorPlacement::XFront:
                    // SideXR: x_det is  z_pic , y_det is y_pic
                    return ReturnType{CELL_DEPTH, CELL_HEIGHT, CELL_WIDTH};
                case DetectorPlacement::XRear:
                    // SideX: x_det is z_pic , y_det is y_pic
                    return ReturnType{CELL_DEPTH, CELL_HEIGHT, CELL_WIDTH};
                case DetectorPlacement::YFront:
                    // SideYR: x_det is z_pic , y_det is x_pic
                    return ReturnType{CELL_DEPTH, CELL_WIDTH, CELL_HEIGHT};
                case DetectorPlacement::YRear:
                    // SideY: x_det is y_pic , y_det is x_pic
                    return ReturnType{CELL_DEPTH, CELL_WIDTH, CELL_HEIGHT};
                case DetectorPlacement::ZFront:
                    // SideYR: x_det is y_pic , y_det is x_pic
                    return ReturnType{CELL_HEIGHT, CELL_WIDTH, CELL_DEPTH};
                // case DetectorPlacement::ZRear:
                default:
                    return ReturnType{CELL_HEIGHT, CELL_WIDTH, CELL_DEPTH};
                }
                // TODO it would be probably better to use case DetectorPlacement::ZRear: and handle default extra
            }
        } // namespace photonDetector
    } // namespace plugins
} // namespace picongpu
