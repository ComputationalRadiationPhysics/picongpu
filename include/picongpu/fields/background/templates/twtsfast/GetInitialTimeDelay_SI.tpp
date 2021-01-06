/* Copyright 2014-2021 Alexander Debus
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

#include <pmacc/types.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/dimensions/DataSpace.hpp>

namespace picongpu
{
    namespace templates
    {
        namespace twtsfast
        {
            /* Auxiliary functions for calculating the TWTS field */
            namespace detail
            {
                template<unsigned T_dim>
                class GetInitialTimeDelay
                {
                public:
                    /** Obtain the SI time delay that later enters the Ex(r, t), By(r, t) and Bz(r, t)
                     *  calculations as t.
                     * @tparam T_dim Specializes for the simulation dimension
                     *  @param auto_tdelay calculate the time delay such that the TWTS pulse is not
                     *                     inside the simulation volume at simulation start
                     *                     timestep = 0 [default = true]
                     *  @param tdelay_user_SI manual time delay if auto_tdelay is false
                     *  @param halfSimSize center of simulation volume in number of cells
                     *  @param pulselength_SI sigma of std. gauss for intensity (E^2)
                     *  @param focus_y_SI the distance to the laser focus in y-direction [m]
                     *  @param phi interaction angle between TWTS laser propagation vector and
                     *             the y-axis [rad, default = 90.*(PI / 180.)]
                     *  @param beta_0 propagation speed of overlap normalized
                     *                to the speed of light [c, default = 1.0]
                     *  @return time delay in SI units */
                    HDINLINE float_64 operator()(
                        bool const auto_tdelay,
                        float_64 const tdelay_user_SI,
                        DataSpace<simDim> const& halfSimSize,
                        float_64 const pulselength_SI,
                        float_64 const focus_y_SI,
                        float_X const phi,
                        float_X const beta_0) const;
                };

                template<>
                HDINLINE float_64 GetInitialTimeDelay<DIM3>::operator()(
                    bool const auto_tdelay,
                    float_64 const tdelay_user_SI,
                    DataSpace<simDim> const& halfSimSize,
                    float_64 const pulselength_SI,
                    float_64 const focus_y_SI,
                    float_X const phi,
                    float_X const beta_0) const
                {
                    if(auto_tdelay)
                    {
                        /* angle between the laser pulse front and the y-axis. Good approximation for
                         * beta0\simeq 1. For exact relation look in TWTS core routines for Ex, By or Bz. */
                        float_64 const eta = (PI / 2) - (phi / 2);
                        /* halfSimSize[2] --> Half-depth of simulation volume (in z); By geometric
                         * projection we calculate the y-distance walkoff of the TWTS-pulse.
                         * The abs( )-function is for correct offset for -phi<-90Deg and +phi>+90Deg. */
                        float_64 const y1
                            = float_64(halfSimSize[2] * picongpu::SI::CELL_DEPTH_SI) * math::abs(math::cos(eta));
                        /* Fudge parameter to make sure, that TWTS pulse starts to impact simulation volume
                         * at low intensity values. */
                        float_64 const m = 3.;
                        /* Approximate cross section of laser pulse through y-axis,
                         * scaled with "fudge factor" m. */
                        float_64 const y2 = m * (pulselength_SI * picongpu::SI::SPEED_OF_LIGHT_SI) / math::cos(eta);
                        /* y-position of laser coordinate system origin within simulation. */
                        float_64 const y3 = focus_y_SI;
                        /* Programmatically obtained time-delay */
                        float_64 const tdelay = (y1 + y2 + y3) / (picongpu::SI::SPEED_OF_LIGHT_SI * beta_0);

                        return tdelay;
                    }
                    else
                        return tdelay_user_SI;
                }

                template<>
                HDINLINE float_64 GetInitialTimeDelay<DIM2>::operator()(
                    bool const auto_tdelay,
                    float_64 const tdelay_user_SI,
                    DataSpace<simDim> const& halfSimSize,
                    float_64 const pulselength_SI,
                    float_64 const focus_y_SI,
                    float_X const phi,
                    float_X const beta_0) const
                {
                    if(auto_tdelay)
                    {
                        /* angle between the laser pulse front and the y-axis. Good approximation for
                         * beta0\simeq 1. For exact relation look in TWTS core routines for Ex, By or Bz. */
                        float_64 const eta = (PI / 2) - (phi / 2);
                        /* halfSimSize[0] --> Half-depth of simulation volume (in x); By geometric
                         * projection we calculate the y-distance walkoff of the TWTS-pulse.
                         * The abs( )-function is for correct offset for -phi<-90Deg and +phi>+90Deg. */
                        float_64 const y1
                            = float_64(halfSimSize[0] * picongpu::SI::CELL_WIDTH_SI) * math::abs(math::cos(eta));
                        /* Fudge parameter to make sure, that TWTS pulse starts to impact simulation volume
                         * at low intensity values. */
                        float_64 const m = 3.;
                        /* Approximate cross section of laser pulse through y-axis,
                         * scaled with "fudge factor" m. */
                        float_64 const y2 = m * (pulselength_SI * picongpu::SI::SPEED_OF_LIGHT_SI) / math::cos(eta);
                        /* y-position of laser coordinate system origin within simulation. */
                        float_64 const y3 = focus_y_SI;
                        /* Programmatically obtained time-delay */
                        float_64 const tdelay = (y1 + y2 + y3) / (picongpu::SI::SPEED_OF_LIGHT_SI * beta_0);

                        return tdelay;
                    }
                    else
                        return tdelay_user_SI;
                }

                template<unsigned T_Dim>
                HDINLINE float_64 getInitialTimeDelay_SI(
                    bool const auto_tdelay,
                    float_64 const tdelay_user_SI,
                    DataSpace<T_Dim> const& halfSimSize,
                    float_64 const pulselength_SI,
                    float_64 const focus_y_SI,
                    float_X const phi,
                    float_X const beta_0)
                {
                    return GetInitialTimeDelay<T_Dim>()(
                        auto_tdelay,
                        tdelay_user_SI,
                        halfSimSize,
                        pulselength_SI,
                        focus_y_SI,
                        phi,
                        beta_0);
                }

            } /* namespace detail */
        } /* namespace twtsfast */
    } /* namespace templates */
} /* namespace picongpu */
