/**
 * Copyright 2014 Alexander Debus, Axel Huebl
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

#include "math/Vector.hpp"
#include "dimensions/DataSpace.hpp"
#include "mappings/simulation/SubGrid.hpp"

#include "fields/numericalCellTypes/YeeCell.hpp"
/** \todo not great... if complex is that general, refactor it to libPMacc! */
#include "plugins/radiation/complex.hpp"


namespace picongpu
{
    /** Load external TWTS field
     *
     */
    namespace templates
    {
        namespace detail
        {
            HDINLINE float_64
            getTime(const float_64& time, const float_64& tdelay)
            {
                if (::picongpu::bgrTWTS::auto_tdelay)
                    return time-tdelay;
                else
                    return time-::picongpu::bgrTWTS::SI::tdelay;
            }
        } /* namespace detail */

        class TWTSFieldE
        {
        public:
            /* We use this to calculate your SI input back to our unit system */
            const DataSpace<simDim> halfSimSize;

            HDINLINE
            TWTSFieldE()
            {
#if !defined(__CUDA_ARCH__)
                const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                const DataSpace<simDim> halfSimSize(subGrid.getGlobalDomain().size / 2);
#endif
            }

            /** Specify your background field E(r,t) here
             *
             * \param cellIdx The total cell id counted from the start at t=0
             * \param currentStep The current time step
             * \param halfSimSize Center of simulation volume in number of cells */
            HDINLINE float3_X
            operator()( const DataSpace<simDim>& cellIdx,
                        const uint32_t currentStep ) const
            {
                const float_X focus_y = ::picongpu::bgrTWTS::SI::FOCUS_POS_SI/::picongpu::SI::CELL_HEIGHT_SI;
#if( SIMDIM == DIM3 )
                const float3_X helper = float3_X( halfSimSize.x(), focus_y, halfSimSize.z() );
                
                /* For the Yee-Cell shifted fields, obtain the fractional cell index components and add that to the total cell indices. The physical field coordinate origin is transversally centered with respect to the global simulation volume. */
                ::PMacc::math::Vector<float3_X,DIM3> eFieldPositions = picongpu::yeeCell::YeeCell::getEFieldPosition();
                const float3_X cellDimensions = precisionCast<float3_X>(::picongpu::cellSize) * (float_X)::picongpu::UNIT_LENGTH;
                eFieldPositions[0] = ((float3_X)cellIdx+eFieldPositions[0]-helper) * cellDimensions; // cellIdx(Ex)
                eFieldPositions[1] = ((float3_X)cellIdx+eFieldPositions[1]-helper) * cellDimensions; // cellIdx(Ey)
                eFieldPositions[2] = ((float3_X)cellIdx+eFieldPositions[2]-helper) * cellDimensions; // cellIdx(Ez)
#elif( SIMDIM == DIM2 )
                const float2_X helper = float2_X( halfSimSize.x(), focus_y );
                
                /* For the Yee-Cell shifted fields, obtain the fractional cell index components and add that to the total cell indices. The physical field coordinate origin is transversally centered with respect to the global simulation volume. */
                ::PMacc::math::Vector<float2_X,DIM3> eFieldPositions = picongpu::yeeCell::YeeCell::getEFieldPosition();
                const float2_X cellDimensions = precisionCast<float2_X>(::picongpu::cellSize) * (float_X)::picongpu::UNIT_LENGTH;
                eFieldPositions[0] = ((float2_X)cellIdx+eFieldPositions[0]-helper) * cellDimensions;; // cellIdx(Ex)
                eFieldPositions[1] = ((float2_X)cellIdx+eFieldPositions[1]-helper) * cellDimensions;; // cellIdx(Ey)
                eFieldPositions[2] = ((float2_X)cellIdx+eFieldPositions[2]-helper) * cellDimensions;; // cellIdx(Ez)
#endif
                
                const float_X time=currentStep*::picongpu::SI::DELTA_T_SI;
            
                /* specify your E-Field in V/m and convert to PIConGPU units */
                if ( ! ::picongpu::bgrTWTS::includeCollidingTWTS ) {
                // Single TWTS-Pulse
#if( SIMDIM == DIM3 )
                    return float3_X((::picongpu::bgrTWTS::SI::AMPLITUDE_SI)*calcTWTSEx(eFieldPositions[0],time,halfSimSize,::picongpu::bgrTWTS::SI::PHI_SI), 0., 0.);
#elif( SIMDIM == DIM2 )
                    /** Corresponding position vector for the Ez-components in 2D simulations.
                     *  3D     2D
                     *  x -->  z
                     *  y -->  y
                     *  z --> -x (Since z=0 for 2D, we use the existing TWTS-field-function and set -x=0)
                     *  Ex --> Ez (--> Same function values can be used in 2D, but with Yee-Cell-Positions for Ez.)
                     *  By --> By
                     *  Bz --> -Bx
                     */
                    const float3_X dim2PosEz = float3_X( 0.0, (eFieldPositions[2]).y(), (eFieldPositions[2]).x() );
                    return float3_X( 0.0, 0.0, (::picongpu::bgrTWTS::SI::AMPLITUDE_SI)*calcTWTSEx(dim2PosEz,time,halfSimSize,::picongpu::bgrTWTS::SI::PHI_SI) );
#endif
                }
                else {
                // Colliding TWTS-Pulse
#if( SIMDIM == DIM3 )
                    return float3_X( (::picongpu::bgrTWTS::SI::AMPLITUDE_SI)
                                       *( calcTWTSEx(eFieldPositions[0],time,halfSimSize,+(::picongpu::bgrTWTS::SI::PHI_SI))
                                         +calcTWTSEx(eFieldPositions[0],time,halfSimSize,-(::picongpu::bgrTWTS::SI::PHI_SI)) ),
                                     0.0, 0.0 );
#elif( SIMDIM == DIM2 )
                    const float3_X dim2PosEz = float3_X( 0.0, (eFieldPositions[2]).y(), (eFieldPositions[2]).x() );
                    return float3_X( 0.0, 0.0, (::picongpu::bgrTWTS::SI::AMPLITUDE_SI)
                                       *( calcTWTSEx(dim2PosEz,time,halfSimSize,+(::picongpu::bgrTWTS::SI::PHI_SI))
                                         +calcTWTSEx(dim2PosEz,time,halfSimSize,-(::picongpu::bgrTWTS::SI::PHI_SI)) )
                                   );
#endif
                }
            }

            /** Calculate the Ex(r,t) field here
             *
             * \param pos Spatial position of the target field.
             * \param time Absolute time (SI, including all offsets and transformations) for calculating the field
             * \param halfSimSize Center of simulation volume in number of cells
             * \param phiReal interaction angle between TWTS laser propagation vector and the y-axis */
            HDINLINE float_X
            calcTWTSEx( const float3_X& pos, const float_X& time, const DataSpace<simDim> halfSimSize, const float_X& phiReal ) const
            {
                const float_64 beta0=::picongpu::bgrTWTS::SI::BETA0_SI; // propagation speed of overlap normalized to the speed of light. [Default: beta0=1.0]
                const float_64 alphaTilt=atan2(1-beta0*cos(phiReal),beta0*sin(phiReal));
                const float_64 phi=2*alphaTilt; // Definition of the laser pulse front tilt angle for the laser field below. For beta0=1.0, this is equivalent to our standard definition.
                const float_64 eta = PI/2 - (phiReal - alphaTilt); // angle between the laser pulse front and the y-axis

                const float_64 cspeed=::picongpu::SI::SPEED_OF_LIGHT_SI;
                const float_64 lambda0=::picongpu::bgrTWTS::SI::WAVE_LENGTH_SI;
                const float_64 om0=2*PI*cspeed/lambda0;
                const float_64 tauG=(::picongpu::bgrTWTS::SI::PULSE_LENGTH_SI)*2.0; // factor 2 arises from definition convention in laser formula
                const float_64 w0=::picongpu::bgrTWTS::SI::WX_SI; // w0 is wx here --> w0 could be replaced by wx
                const float_64 rho0=PI*w0*w0/lambda0;
                const float_64 wy=::picongpu::bgrTWTS::SI::WY_SI; // Width of TWTS pulse
                const float_64 k=2*PI/lambda0;
                const float_64 x=pos.x();
                const float_64 y=-sin(phiReal)*pos.y()-cos(phiReal)*pos.z();    // RotationMatrix[PI-phiReal].(y,z)
                const float_64 z=+cos(phiReal)*pos.y()-sin(phiReal)*pos.z();    // TO DO: For 2 counter-propagation TWTS pulses take +phiReal and -phiReal. Where do we implement this?
                const float_64 y1=(float_64)(halfSimSize[2]*::picongpu::SI::CELL_DEPTH_SI)/tan(eta); // halfSimSize[2] --> Half-depth of simulation volume (in z); By geometric projection we calculate the y-distance walkoff of the TWTS-pulse.
                const float_64 m=3.; // Fudge parameter to make sure, that TWTS pulse starts to impact simulation volume at low intensity values.
                const float_64 y2=(tauG/2*cspeed)/sin(eta)*m; // pulse length projected on y-axis, scaled with "fudge factor" m.
                const float_64 y3=::picongpu::bgrTWTS::SI::FOCUS_POS_SI; // Position of maximum intensity in simulation volume along y
                const float_64 tdelay= (y1+y2+y3)/(cspeed*beta0);
                const float_64 t=::picongpu::bgrTWTS::getTime(time,tdelay);

                const Complex_64 helpVar1=Complex_64(0,1)*rho0 - y*cos(phi) - z*sin(phi);
                const Complex_64 helpVar2=Complex_64(0,-1)*cspeed*om0*tauG*tauG - y*cos(phi)/cos(phi/2.)/cos(phi/2.)*tan(phi/2.) - 2*z*tan(phi/2.)*tan(phi/2.);
                const Complex_64 helpVar3=Complex_64(0,1)*rho0 - y*cos(phi) - z*sin(phi);

                const Complex_64 helpVar4=(
                -(cspeed*cspeed*k*om0*tauG*tauG*wy*wy*x*x)
                - 2*cspeed*cspeed*om0*t*t*wy*wy*rho0
                + Complex_64(0,2)*cspeed*cspeed*om0*om0*t*tauG*tauG*wy*wy*rho0
                - 2*cspeed*cspeed*om0*tauG*tauG*y*y*rho0
                + 4*cspeed*om0*t*wy*wy*z*rho0
                - Complex_64(0,2)*cspeed*om0*om0*tauG*tauG*wy*wy*z*rho0
                - 2*om0*wy*wy*z*z*rho0
                - Complex_64(0,8)*om0*wy*wy*y*(cspeed*t - z)*z*sin(phi/2.)*sin(phi/2.)
                + Complex_64(0,8)/sin(phi)*(
                        +2*z*z*(cspeed*om0*t*wy*wy + Complex_64(0,1)*cspeed*y*y - om0*wy*wy*z)
                        + y*(
                            + cspeed*k*wy*wy*x*x
                            - Complex_64(0,2)*cspeed*om0*t*wy*wy*rho0
                            + 2*cspeed*y*y*rho0
                            + Complex_64(0,2)*om0*wy*wy*z*rho0
                        )*tan(PI/2-phi)/sin(phi)
                    )*sin(phi/2.)*sin(phi/2.)*sin(phi/2.)*sin(phi/2.)
                - Complex_64(0,2)*cspeed*cspeed*om0*t*t*wy*wy*z*sin(phi)
                - 2*cspeed*cspeed*om0*om0*t*tauG*tauG*wy*wy*z*sin(phi)
                - Complex_64(0,2)*cspeed*cspeed*om0*tauG*tauG*y*y*z*sin(phi)
                + Complex_64(0,4)*cspeed*om0*t*wy*wy*z*z*sin(phi)
                + 2*cspeed*om0*om0*tauG*tauG*wy*wy*z*z*sin(phi)
                - Complex_64(0,2)*om0*wy*wy*z*z*z*sin(phi)
                - 4*cspeed*om0*t*wy*wy*y*rho0*tan(phi/2.)
                + 4*om0*wy*wy*y*z*rho0*tan(phi/2.)
                + Complex_64(0,2)*y*y*(cspeed*om0*t*wy*wy + Complex_64(0,1)*cspeed*y*y - om0*wy*wy*z)*cos(phi)*cos(phi)/cos(phi/2.)/cos(phi/2.)*tan(phi/2.)
                + Complex_64(0,2)*cspeed*k*wy*wy*x*x*z*tan(phi/2.)*tan(phi/2.)
                - 2*om0*wy*wy*y*y*rho0*tan(phi/2.)*tan(phi/2.)
                + 4*cspeed*om0*t*wy*wy*z*rho0*tan(phi/2.)*tan(phi/2.)
                + Complex_64(0,4)*cspeed*y*y*z*rho0*tan(phi/2.)*tan(phi/2.)
                - 4*om0*wy*wy*z*z*rho0*tan(phi/2.)*tan(phi/2.)
                - Complex_64(0,2)*om0*wy*wy*y*y*z*sin(phi)*tan(phi/2.)*tan(phi/2.)
                - 2*y*cos(phi)*(om0*(cspeed*cspeed*(Complex_64(0,1)*t*t*wy*wy + om0*t*tauG*tauG*wy*wy + Complex_64(0,1)*tauG*tauG*y*y) - cspeed*(Complex_64(0,2)*t + om0*tauG*tauG)*wy*wy*z + Complex_64(0,1)*wy*wy*z*z) + Complex_64(0,2)*om0*wy*wy*y*(cspeed*t - z)*tan(phi/2.) + Complex_64(0,1)*(Complex_64(0,-4)*cspeed*y*y*z + om0*wy*wy*(y*y - 4*(cspeed*t - z)*z))*tan(phi/2.)*tan(phi/2.))
                )/(2.*cspeed*wy*wy*helpVar1*helpVar2);

                const Complex_64 helpVar5=cspeed*om0*tauG*tauG - Complex_64(0,8)*y*tan(PI/2-phi)/sin(phi)/sin(phi)*sin(phi/2.)*sin(phi/2.)*sin(phi/2.)*sin(phi/2.) - Complex_64(0,2)*z*tan(phi/2.)*tan(phi/2.);
                const Complex_64 result=(Complex_64::cexp(helpVar4)*tauG*Complex_64::csqrt((cspeed*om0*rho0)/helpVar3))/Complex_64::csqrt(helpVar5);
                return result.get_real();
            }

        };

        class TWTSFieldB
        {
        public:
            /* We use this to calculate your SI input back to our unit system */
            const DataSpace<simDim> halfSimSize;

            HDINLINE
            TWTSFieldB()
            {
#if !defined(__CUDA_ARCH__)
                const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                const DataSpace<simDim> halfSimSize(subGrid.getGlobalDomain().size / 2);
#endif
            }

            /** Specify your background field B(r,t) here
             *
             * \param cellIdx The total cell id counted from the start at t=0
             * \param currentStep The current time step
             * \param halfSimSize Center of simulation volume in number of cells */
            HDINLINE float3_X
            operator()( const DataSpace<simDim>& cellIdx,
                        const uint32_t currentStep ) const
            {
                const float_X focus_y = ::picongpu::bgrTWTS::SI::FOCUS_POS_SI/::picongpu::SI::CELL_HEIGHT_SI;
#if( SIMDIM == DIM3 )
                const float3_X helper = float3_X( halfSimSize.x(), focus_y, halfSimSize.z() );

                /* For the Yee-Cell shifted fields, obtain the fractional cell index components and add that to the total cell indices. The physical field coordinate origin is in the center of the simulation */
                ::PMacc::math::Vector<float3_X,DIM3> bFieldPositions = picongpu::yeeCell::YeeCell::getBFieldPosition();
                const float3_X cellDimensions = precisionCast<float3_X>(::picongpu::cellSize) * (float_X)::picongpu::UNIT_LENGTH;
                bFieldPositions[0] = ((float3_X)cellIdx+bFieldPositions[0]-helper) * cellDimensions; // cellIdx(Bx)
                bFieldPositions[1] = ((float3_X)cellIdx+bFieldPositions[1]-helper) * cellDimensions; // cellIdx(By)
                bFieldPositions[2] = ((float3_X)cellIdx+bFieldPositions[2]-helper) * cellDimensions; // cellIdx(Bz)
#elif( SIMDIM == DIM2 )
                const float2_X helper = float2_X( halfSimSize.x(), focus_y );

                /* For the Yee-Cell shifted fields, obtain the fractional cell index components and add that to the total cell indices. The physical field coordinate origin is in the center of the simulation */
                ::PMacc::math::Vector<float2_X,DIM3> bFieldPositions = picongpu::yeeCell::YeeCell::getBFieldPosition();
                const float2_X cellDimensions = precisionCast<float2_X>(::picongpu::cellSize) * (float_X)::picongpu::UNIT_LENGTH;
                bFieldPositions[0] = ((float2_X)cellIdx+bFieldPositions[0]-helper) * float2_X(::picongpu::SI::CELL_WIDTH_SI,::picongpu::SI::CELL_HEIGHT_SI); // cellIdx(Bx)
                bFieldPositions[1] = ((float2_X)cellIdx+bFieldPositions[1]-helper) * float2_X(::picongpu::SI::CELL_WIDTH_SI,::picongpu::SI::CELL_HEIGHT_SI); // cellIdx(By)
                bFieldPositions[2] = ((float2_X)cellIdx+bFieldPositions[2]-helper) * float2_X(::picongpu::SI::CELL_WIDTH_SI,::picongpu::SI::CELL_HEIGHT_SI); // cellIdx(Bz)
#endif
                const float_X time=currentStep*::picongpu::SI::DELTA_T_SI;

                /* specify your E-Field in V/m and convert to PIConGPU units */
                if ( ! ::picongpu::bgrTWTS::includeCollidingTWTS ) {
                    // Single TWTS-Pulse
#if( SIMDIM == DIM3 )
                    return float3_X(0.0, (::picongpu::bgrTWTS::SI::AMPLITUDE_SI)*calcTWTSBy(bFieldPositions[1], time, halfSimSize, ::picongpu::bgrTWTS::SI::PHI_SI), (::picongpu::bgrTWTS::SI::AMPLITUDE_SI)*calcTWTSBz(bFieldPositions[2], time, halfSimSize, ::picongpu::bgrTWTS::SI::PHI_SI) );
#elif( SIMDIM == DIM2 )
                    /** Corresponding position vector for the Ez-components in 2D simulations.
                     *  3D     2D
                     *  x -->  z
                     *  y -->  y
                     *  z --> -x (Since z=0 for 2D, we use the existing TWTS-field-function and set -x=0)
                     *  Ex --> Ez (--> Same function values can be used in 2D, but with Yee-Cell-Positions for Ez.)
                     *  By --> By
                     *  Bz --> -Bx
                     */
                    const float3_X dim2PosBx = float3_X( 0.0, (bFieldPositions[0]).y(), (bFieldPositions[0]).x() );
                    const float3_X dim2PosBy = float3_X( 0.0, (bFieldPositions[1]).y(), (bFieldPositions[1]).x() );
                    return float3_X( -1.0*(::picongpu::bgrTWTS::SI::AMPLITUDE_SI)*calcTWTSBz(dim2PosBx, time, halfSimSize, ::picongpu::bgrTWTS::SI::PHI_SI) , (::picongpu::bgrTWTS::SI::AMPLITUDE_SI)*calcTWTSBy(dim2PosBy, time, halfSimSize, ::picongpu::bgrTWTS::SI::PHI_SI), 0.0 );
#endif
                }
                else {
                    // Colliding TWTS-Pulse
#if( SIMDIM == DIM3 )
                    return float3_X(0.0, (::picongpu::bgrTWTS::SI::AMPLITUDE_SI)
                                    *( calcTWTSBy(bFieldPositions[1], time, halfSimSize, +(::picongpu::bgrTWTS::SI::PHI_SI))
                                      +calcTWTSBy(bFieldPositions[1], time, halfSimSize, -(::picongpu::bgrTWTS::SI::PHI_SI))
                                    ),
                                    (::picongpu::bgrTWTS::SI::AMPLITUDE_SI)
                                    * ( calcTWTSBz(bFieldPositions[2], time, halfSimSize, +(::picongpu::bgrTWTS::SI::PHI_SI))
                                        +calcTWTSBz(bFieldPositions[2], time, halfSimSize, -(::picongpu::bgrTWTS::SI::PHI_SI))
                                      )
                                    );
#elif( SIMDIM == DIM2 )
                    const float3_X dim2PosBx = float3_X( 0.0, (bFieldPositions[0]).y(), (bFieldPositions[0]).x() );
                    const float3_X dim2PosBy = float3_X( 0.0, (bFieldPositions[1]).y(), (bFieldPositions[1]).x() );
                    return float3_X( -1.0*(::picongpu::bgrTWTS::SI::AMPLITUDE_SI)
                                    *( calcTWTSBz(dim2PosBx, time, halfSimSize, +(::picongpu::bgrTWTS::SI::PHI_SI))
                                      +calcTWTSBz(dim2PosBx, time, halfSimSize, -(::picongpu::bgrTWTS::SI::PHI_SI))
                                    ),
                            (::picongpu::bgrTWTS::SI::AMPLITUDE_SI)
                                    *( calcTWTSBz(dim2PosBy, time, halfSimSize, +(::picongpu::bgrTWTS::SI::PHI_SI))
                                      +calcTWTSBz(dim2PosBy, time, halfSimSize, -(::picongpu::bgrTWTS::SI::PHI_SI))
                                    ), 0.0 );
#endif
                }
            }

            /** Calculate the By(r,t) field here
             *
             * \param pos Spatial position of the target field.
             * \param time Absolute time (SI, including all offsets and transformations) for calculating the field
             * \param halfSimSize Center of simulation volume in number of cells
             * \param phiReal interaction angle between TWTS laser propagation vector and the y-axis */
            HDINLINE float_X
            calcTWTSBy( const float3_X& pos, const float_X& time, const DataSpace<simDim> halfSimSize, const float_X& phiReal ) const
            {
                const float_64 beta0=::picongpu::bgrTWTS::SI::BETA0_SI; // propagation speed of overlap normalized to the speed of light. [Default: beta0=1.0]
                const float_64 alphaTilt=atan2(1-beta0*cos(phiReal),beta0*sin(phiReal));
                const float_64 phi=2*alphaTilt; // Definition of the laser pulse front tilt angle for the laser field below. For beta0=1.0, this is equivalent to our standard definition.
                const float_64 eta = PI/2 - (phiReal - alphaTilt); // angle between the laser pulse front and the y-axis

                const float_64 cspeed=::picongpu::SI::SPEED_OF_LIGHT_SI;
                const float_64 lambda0=::picongpu::bgrTWTS::SI::WAVE_LENGTH_SI;
                const float_64 om0=2*PI*cspeed/lambda0;
                const float_64 tauG=(::picongpu::bgrTWTS::SI::PULSE_LENGTH_SI)*2.0; // factor 2 arises from definition convention in laser formula
                const float_64 w0=::picongpu::bgrTWTS::SI::WX_SI; // w0 is wx here --> w0 could be replaced by wx
                const float_64 rho0=PI*w0*w0/lambda0;
                const float_64 wy=::picongpu::bgrTWTS::SI::WY_SI; // Width of TWTS pulse
                const float_64 k=2*PI/lambda0;
                const float_64 x=pos.x();
                const float_64 y=-sin(phiReal)*pos.y()-cos(phiReal)*pos.z();    // RotationMatrix[PI-phiReal].(y,z)
                const float_64 z=+cos(phiReal)*pos.y()-sin(phiReal)*pos.z();    // TO DO: For 2 counter-propagation TWTS pulses take +phiReal and -phiReal. Where do we implement this?
                const float_64 y1=(float_64)(halfSimSize[2]*::picongpu::SI::CELL_DEPTH_SI)/tan(eta); // halfSimSize[2] --> Half-depth of simulation volume (in z); By geometric projection we calculate the y-distance walkoff of the TWTS-pulse.
                const float_64 m=3.; // Fudge parameter to make sure, that TWTS pulse starts to impact simulation volume at low intensity values.
                const float_64 y2=(tauG/2*cspeed)/sin(eta)*m; // pulse length projected on y-axis, scaled with "fudge factor" m.
                const float_64 y3=::picongpu::bgrTWTS::SI::FOCUS_POS_SI; // Position of maximum intensity in simulation volume along y
                const float_64 tdelay= (y1+y2+y3)/(cspeed*beta0);
                const float_64 t=::picongpu::bgrTWTS::getTime(time,tdelay);

                const Complex_64 helpVar1=rho0 + Complex_64(0,1)*y*cos(phi) + Complex_64(0,1)*z*sin(phi);
                const Complex_64 helpVar2=cspeed*om0*tauG*tauG + Complex_64(0,2)*(-z - y*tan(PI/2-phi))*tan(phi/2.)*tan(phi/2.);
                const Complex_64 helpVar3=Complex_64(0,1)*rho0 - y*cos(phi) - z*sin(phi);
                const Complex_64 helpVar4=-1.0*(
                cspeed*cspeed*k*om0*tauG*tauG*wy*wy*x*x
                + 2*cspeed*cspeed*om0*t*t*wy*wy*rho0
                - Complex_64(0,2)*cspeed*cspeed*om0*om0*t*tauG*tauG*wy*wy*rho0
                + 2*cspeed*cspeed*om0*tauG*tauG*y*y*rho0
                - 4*cspeed*om0*t*wy*wy*z*rho0
                + Complex_64(0,2)*cspeed*om0*om0*tauG*tauG*wy*wy*z*rho0
                + 2*om0*wy*wy*z*z*rho0
                + 4*cspeed*om0*t*wy*wy*y*rho0*tan(phi/2.)
                - 4*om0*wy*wy*y*z*rho0*tan(phi/2.)
                - Complex_64(0,2)*cspeed*k*wy*wy*x*x*z*tan(phi/2.)*tan(phi/2.)
                + 2*om0*wy*wy*y*y*rho0*tan(phi/2.)*tan(phi/2.)
                - 4*cspeed*om0*t*wy*wy*z*rho0*tan(phi/2.)*tan(phi/2.)
                - Complex_64(0,4)*cspeed*y*y*z*rho0*tan(phi/2.)*tan(phi/2.)
                + 4*om0*wy*wy*z*z*rho0*tan(phi/2.)*tan(phi/2.)
                - Complex_64(0,2)*cspeed*k*wy*wy*x*x*y*tan(PI/2-phi)*tan(phi/2.)*tan(phi/2.)
                - 4*cspeed*om0*t*wy*wy*y*rho0*tan(PI/2-phi)*tan(phi/2.)*tan(phi/2.)
                - Complex_64(0,4)*cspeed*y*y*y*rho0*tan(PI/2-phi)*tan(phi/2.)*tan(phi/2.)
                + 4*om0*wy*wy*y*z*rho0*tan(PI/2-phi)*tan(phi/2.)*tan(phi/2.)
                + 2*z*sin(phi)*(
                    om0*(cspeed*cspeed*(Complex_64(0,1)*t*t*wy*wy + om0*t*tauG*tauG*wy*wy + Complex_64(0,1)*tauG*tauG*y*y) - cspeed*(Complex_64(0,2)*t + om0*tauG*tauG)*wy*wy*z + Complex_64(0,1)*wy*wy*z*z)
                    + Complex_64(0,2)*om0*wy*wy*y*(cspeed*t - z)*tan(phi/2.) + Complex_64(0,1)*(Complex_64(0,-2)*cspeed*y*y*z + om0*wy*wy*(y*y - 2*(cspeed*t - z)*z))*tan(phi/2.)*tan(phi/2.)
                    )
                + 2*y*cos(phi)*(
                    om0*(cspeed*cspeed*(Complex_64(0,1)*t*t*wy*wy + om0*t*tauG*tauG*wy*wy + Complex_64(0,1)*tauG*tauG*y*y) - cspeed*(Complex_64(0,2)*t + om0*tauG*tauG)*wy*wy*z + Complex_64(0,1)*wy*wy*z*z)
                    + Complex_64(0,2)*om0*wy*wy*y*(cspeed*t - z)*tan(phi/2.)
                    + Complex_64(0,1)*(Complex_64(0,-4)*cspeed*y*y*z + om0*wy*wy*(y*y - 4*(cspeed*t - z)*z) - 2*y*(cspeed*om0*t*wy*wy + Complex_64(0,1)*cspeed*y*y - om0*wy*wy*z)*tan(PI/2-phi))*tan(phi/2.)*tan(phi/2.)
                    )
                )/(2.*cspeed*wy*wy*helpVar1*helpVar2);

                const Complex_64 helpVar5=Complex_64(0,-1)*cspeed*om0*tauG*tauG + (-z - y*tan(PI/2-phi))*tan(phi/2.)*tan(phi/2.)*2;
                const Complex_64 helpVar6=(cspeed*(cspeed*om0*tauG*tauG + Complex_64(0,2)*(-z - y*tan(PI/2-phi))*tan(phi/2.)*tan(phi/2.)))/(om0*rho0);
                const Complex_64 result=(Complex_64::cexp(helpVar4)*tauG/cos(phi/2.)/cos(phi/2.)*(rho0 + Complex_64(0,1)*y*cos(phi) + Complex_64(0,1)*z*sin(phi))*(Complex_64(0,2)*cspeed*t + cspeed*om0*tauG*tauG - Complex_64(0,4)*z + cspeed*(Complex_64(0,2)*t + om0*tauG*tauG)*cos(phi) + Complex_64(0,2)*y*tan(phi/2.))*Complex_64::cpow(helpVar3,-1.5))/(2.*helpVar5*Complex_64::csqrt(helpVar6));

                return result.get_real();
            }

            /** Calculate the Bz(r,t) field here
             *
             * \param pos Spatial position of the target field.
             * \param time Absolute time (SI, including all offsets and transformations) for calculating the field
             * \param halfSimSize Center of simulation volume in number of cells
             * \param phiReal interaction angle between TWTS laser propagation vector and the y-axis */
            HDINLINE float_X
            calcTWTSBz( const float3_X& pos, const float_X& time, const DataSpace<simDim> halfSimSize, const float_X& phiReal ) const
            {
                const float_64 beta0=::picongpu::bgrTWTS::SI::BETA0_SI; // propagation speed of overlap normalized to the speed of light. [Default: beta0=1.0]
                const float_64 alphaTilt=atan2(1-beta0*cos(phiReal),beta0*sin(phiReal));
                const float_64 phi=2*alphaTilt; // Definition of the laser pulse front tilt angle for the laser field below. For beta0=1.0, this is equivalent to our standard definition.
                const float_64 eta = PI/2 - (phiReal - alphaTilt); // angle between the laser pulse front and the y-axis

                const float_64 cspeed=::picongpu::SI::SPEED_OF_LIGHT_SI;
                const float_64 lambda0=::picongpu::bgrTWTS::SI::WAVE_LENGTH_SI;
                const float_64 om0=2*PI*cspeed/lambda0;
                const float_64 tauG=(::picongpu::bgrTWTS::SI::PULSE_LENGTH_SI)*2.0; // factor 2 arises from definition convention in laser formula
                const float_64 w0=::picongpu::bgrTWTS::SI::WX_SI; // w0 is wx here --> w0 could be replaced by wx
                const float_64 rho0=PI*w0*w0/lambda0;
                const float_64 wy=::picongpu::bgrTWTS::SI::WY_SI; // Width of TWTS pulse
                const float_64 k=2*PI/lambda0;
                const float_64 x=pos.x();
                const float_64 y=-sin(phiReal)*pos.y()-cos(phiReal)*pos.z();    // RotationMatrix[PI-phiReal].(y,z)
                const float_64 z=+cos(phiReal)*pos.y()-sin(phiReal)*pos.z();    // TO DO: For 2 counter-propagation TWTS pulses take +phiReal and -phiReal. Where do we implement this?
                const float_64 y1=(float_64)(halfSimSize[2]*::picongpu::SI::CELL_DEPTH_SI)/tan(eta); // halfSimSize[2] --> Half-depth of simulation volume (in z); By geometric projection we calculate the y-distance walkoff of the TWTS-pulse.
                const float_64 m=3.; // Fudge parameter to make sure, that TWTS pulse starts to impact simulation volume at low intensity values.
                const float_64 y2=(tauG/2*cspeed)/sin(eta)*m; // pulse length projected on y-axis, scaled with "fudge factor" m.
                const float_64 y3=::picongpu::bgrTWTS::SI::FOCUS_POS_SI; // Position of maximum intensity in simulation volume along y
                const float_64 tdelay= (y1+y2+y3)/(cspeed*beta0);
                const float_64 t=::picongpu::bgrTWTS::getTime(time,tdelay);

                const Complex_64 helpVar1=-(cspeed*z) - cspeed*y*tan(PI/2-phi) + Complex_64(0,1)*cspeed*rho0/sin(phi);
                const Complex_64 helpVar2=Complex_64(0,1)*rho0 - y*cos(phi) - z*sin(phi);
                const Complex_64 helpVar3=helpVar2*cspeed;
                const Complex_64 helpVar4=cspeed*om0*tauG*tauG - Complex_64(0,1)*y*cos(phi)/cos(phi/2.)/cos(phi/2.)*tan(phi/2.) - Complex_64(0,2)*z*tan(phi/2.)*tan(phi/2.);
                const Complex_64 helpVar5=2*cspeed*t - Complex_64(0,1)*cspeed*om0*tauG*tauG - 2*z + 8*y/sin(phi)/sin(phi)/sin(phi)*sin(phi/2.)*sin(phi/2.)*sin(phi/2.)*sin(phi/2.) - 2*z*tan(phi/2.)*tan(phi/2.);

                const Complex_64 helpVar6=(
                (om0*y*rho0/cos(phi/2.)/cos(phi/2.)/cos(phi/2.)/cos(phi/2.))/helpVar1
                - (Complex_64(0,2)*k*x*x)/helpVar2
                - (Complex_64(0,1)*om0*om0*tauG*tauG*rho0)/helpVar2
                - (Complex_64(0,4)*y*y*rho0)/(wy*wy*helpVar2)
                + (om0*om0*tauG*tauG*y*cos(phi))/helpVar2
                + (4*y*y*y*cos(phi))/(wy*wy*helpVar2)
                + (om0*om0*tauG*tauG*z*sin(phi))/helpVar2
                + (4*y*y*z*sin(phi))/(wy*wy*helpVar2)
                + (Complex_64(0,2)*om0*y*y*cos(phi)/cos(phi/2.)/cos(phi/2.)*tan(phi/2.))/helpVar3
                + (om0*y*rho0*cos(phi)/cos(phi/2.)/cos(phi/2.)*tan(phi/2.))/helpVar3
                + (Complex_64(0,1)*om0*y*y*cos(phi)*cos(phi)/cos(phi/2.)/cos(phi/2.)*tan(phi/2.))/helpVar3
                + (Complex_64(0,4)*om0*y*z*tan(phi/2.)*tan(phi/2.))/helpVar3
                - (2*om0*z*rho0*tan(phi/2.)*tan(phi/2.))/helpVar3
                - (Complex_64(0,2)*om0*z*z*sin(phi)*tan(phi/2.)*tan(phi/2.))/helpVar3
                - (om0*helpVar5*helpVar5)/(cspeed*helpVar4)
                )/4.;

                const Complex_64 helpVar7=cspeed*om0*tauG*tauG - Complex_64(0,1)*y*cos(phi)/cos(phi/2.)/cos(phi/2.)*tan(phi/2.) - Complex_64(0,2)*z*tan(phi/2.)*tan(phi/2.);
                const Complex_64 result=(Complex_64(0,2)*Complex_64::cexp(helpVar6)*tauG*tan(phi/2.)*(cspeed*t - z + y*tan(phi/2.))*Complex_64::csqrt((om0*rho0)/helpVar3))/Complex_64::cpow(helpVar7,1.5);

                return result.get_real();
            }

        };

    } /* namespace templates */
} /* namespace picongpu */
