/* Copyright 2014-2020 Alexander Debus
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
namespace gaussian
{
/* Auxiliary functions for calculating the Gaussian laser field */
namespace detail
{

    template <unsigned T_dim>
    class GetInitialTimeDelay
    {
        public:
        /** Obtain the SI time delay that later enters the Ex(r, t), By(r, t) and Bz(r, t)
         *  calculations as t.
         * \tparam T_dim Specializes for the simulation dimension
         *  \param auto_tdelay calculate the time delay such that the Gaussian laser pulse is not
         *                     inside the simulation volume at simulation start
         *                     timestep = 0 [default = true]
         *  \param tdelay_user_SI manual time delay if auto_tdelay is false
         *  \param halfSimSize center of simulation volume in number of cells
         *  \param pulselength_SI sigma of std. gauss for intensity (E^2)
         *  \param focus_y_SI the distance to the laser focus in y-direction [m]
         *  \param Gaussian laser spot size
         *  \param phi interaction angle between Gaussian laser propagation vector and
         *             the y-axis [rad, default = 90.*(PI / 180.)]
         *  \return time delay in SI units */
        HDINLINE float_64 operator()( const bool auto_tdelay,
                                      const float_64 tdelay_user_SI,
                                      const DataSpace<simDim>& halfSimSize,
                                      const float_64 pulselength_SI,
                                      const float_64 focus_y_SI,
                                      const float_64 w0_SI,
                                      const float_X phi ) const;
    };

    template<>
    HDINLINE float_64
    GetInitialTimeDelay<DIM3>::operator()( const bool auto_tdelay,
                                           const float_64 tdelay_user_SI,
                                           const DataSpace<simDim>& halfSimSize,
                                           const float_64 pulselength_SI,
                                           const float_64 focus_y_SI,
                                           const float_64 w0_SI,
                                           const float_X phi ) const
    {
        if ( auto_tdelay ) {
            const float_64 m = 3; /* "fudge factor"  */
            const float_64 h2 = float_64(halfSimSize[2] * picongpu::SI::CELL_DEPTH_SI);
            if ( math::sin( phi ) == float_X( 0.0 )  ) {
                const float_64 tdelay = h2 / picongpu::SI::SPEED_OF_LIGHT_SI;
                return tdelay;
            }
            else
            {
                if ( focus_y_SI * math::tan( phi ) - m * w0_SI / math::cos ( phi )  ) {
                   const float_64 tdelay = ( h2 / math::sin( phi ) + m * w0_SI / math::tan( phi ) )
                                              / picongpu::SI::SPEED_OF_LIGHT_SI
                                           + m * pulselength_SI / math::sin( phi );
                   return tdelay;
                }
                else
                {
                   const float_64 tdelay = ( focus_y_SI / math::cos( phi ) + m * w0_SI * math::tan( phi ) )
                                              / picongpu::SI::SPEED_OF_LIGHT_SI
                                           + m * pulselength_SI / math::cos( phi );
                   return tdelay;
                }
            }
        }
        else
        {
            return tdelay_user_SI;
        }
    }

    template <>
    HDINLINE float_64
    GetInitialTimeDelay<DIM2>::operator()( const bool auto_tdelay,
                                           const float_64 tdelay_user_SI,
                                           const DataSpace<simDim>& halfSimSize,
                                           const float_64 pulselength_SI,
                                           const float_64 focus_y_SI,
                                           const float_64 w0_SI,
                                           const float_X phi ) const
    {
        if ( auto_tdelay ) {
            const float_64 m = 3; /* "fudge factor" */
            const float_64 h2 = float_64(halfSimSize[2] * picongpu::SI::CELL_DEPTH_SI);
            if ( math::sin( phi ) == float_X( 0.0 )  ) {
                const float_64 tdelay = h2 / picongpu::SI::SPEED_OF_LIGHT_SI;
                return tdelay;
            }
            else
            {
                if ( focus_y_SI * math::tan( phi ) - m * w0_SI / math::cos ( phi )  ) {
                   const float_64 tdelay = ( h2 / math::sin( phi ) + m * w0_SI / math::tan( phi ) )
                                              / picongpu::SI::SPEED_OF_LIGHT_SI
                                           + m * pulselength_SI / math::sin( phi );
                   return tdelay;
                }
                else
                {
                   const float_64 tdelay = ( focus_y_SI / math::cos( phi ) + m * w0_SI * math::tan( phi ) )
                                              / picongpu::SI::SPEED_OF_LIGHT_SI
                                           + m * pulselength_SI / math::cos( phi );
                   return tdelay;
                }
            }
        }
        else
        {
            return tdelay_user_SI;
        }
    }

    template <unsigned T_Dim>
    HDINLINE float_64
    getInitialTimeDelay_SI( const bool auto_tdelay,
                            const float_64 tdelay_user_SI,
                            const DataSpace<T_Dim>& halfSimSize,
                            const float_64 pulselength_SI,
                            const float_64 focus_y_SI,
                            const float_64 w0_SI,
                            const float_X phi )
    {
        return GetInitialTimeDelay<T_Dim>()(auto_tdelay, tdelay_user_SI,
                                            halfSimSize, pulselength_SI,
                                            focus_y_SI, w0_SI, phi );
    }

} /* namespace detail */
} /* namespace gaussian */
} /* namespace templates */
} /* namespace picongpu */
