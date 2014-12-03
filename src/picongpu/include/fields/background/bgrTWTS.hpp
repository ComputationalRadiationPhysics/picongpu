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
#include "fields/numericalCellTypes/YeeCell.hpp"
#include "plugins/radiation/complex.hpp"
#include "math/Vector.hpp"
#include <iostream>

/** Load external TWTS field
 *
 */
namespace picongpu
{
    namespace bgrTWTS
    {
		const bool includeTWTSlaser = true;
		
		class fieldBackgroundE
		{
		public:
			/* Add this additional field for pushing particles */
			static const bool InfluenceParticlePusher = ::picongpu::bgrTWTS::includeTWTSlaser;

			/* We use this to calculate your SI input back to our unit system */
			const float3_64 unitField;
			HDINLINE fieldBackgroundE( const float3_64 unitField ) : unitField(unitField)
			{}

			/** Specify your background field E(r,t) here
			 *
			 * \param cellIdx The total cell id counted from the start at t=0
			 * \param currentStep The current time step
			 * \param halfSimSize Center of simulation volume in number of cells */
			HDINLINE float3_X
			operator()( const DataSpace<simDim>& cellIdx,
						const uint32_t currentStep,
						const DataSpace<simDim> halfSimSize	) const
			{
				const float_X focus_y = ::picongpu::bgrTWTS::SI::FOCUS_POS_SI/::picongpu::SI::CELL_HEIGHT_SI;
				const float3_X helper = float3_X( halfSimSize.x(), focus_y, halfSimSize.z() );
				
				/* For the Yee-Cell shifted fields, obtain the fractional cell index components and add that to the total cell indices. The physical field coordinate origin is transversally centered with respect to the global simulation volume. */
				::PMacc::math::Vector<floatD_X,simDim> eFieldPositions = picongpu::yeeCell::YeeCell::getEFieldPosition();
				eFieldPositions[0] = ((float3_X)cellIdx+eFieldPositions[0]-helper) * float3_X(::picongpu::SI::CELL_WIDTH_SI,::picongpu::SI::CELL_HEIGHT_SI,::picongpu::SI::CELL_DEPTH_SI); // cellIdx(Ex)
				eFieldPositions[1] = ((float3_X)cellIdx+eFieldPositions[1]-helper) * float3_X(::picongpu::SI::CELL_WIDTH_SI,::picongpu::SI::CELL_HEIGHT_SI,::picongpu::SI::CELL_DEPTH_SI); // cellIdx(Ey)
				eFieldPositions[2] = ((float3_X)cellIdx+eFieldPositions[2]-helper) * float3_X(::picongpu::SI::CELL_WIDTH_SI,::picongpu::SI::CELL_HEIGHT_SI,::picongpu::SI::CELL_DEPTH_SI); // cellIdx(Ez)
				
				const float_X time=currentStep*::picongpu::SI::DELTA_T_SI;
			
				/* specify your E-Field in V/m and convert to PIConGPU units */
				if ( ! ::picongpu::bgrTWTS::includeCollidingTWTS ) {
				// Single TWTS-Pulse
					return float3_X((::picongpu::bgrTWTS::SI::AMPLITUDE_SI)*calcTWTSEx(eFieldPositions[0],time,halfSimSize,::picongpu::bgrTWTS::SI::PHI_SI) / unitField[1],0.0, 0.0);
					//return float3_X((eFieldPositions[0]).x()/ unitField[1],(eFieldPositions[0]).y()/ unitField[1], (eFieldPositions[0]).z() / unitField[1]);
				}
				else {
				// Colliding TWTS-Pulse
					return float3_X( (::picongpu::bgrTWTS::SI::AMPLITUDE_SI)
									   *( calcTWTSEx(eFieldPositions[0],time,halfSimSize,+(::picongpu::bgrTWTS::SI::PHI_SI))
										 +calcTWTSEx(eFieldPositions[0],time,halfSimSize,-(::picongpu::bgrTWTS::SI::PHI_SI)) )
										/ unitField[1],0.0, 0.0);
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
				const float_X beta0=::picongpu::bgrTWTS::SI::BETA0_SI; // propagation speed of overlap normalized to the speed of light. [Default: beta0=1.0]
				const float_X alphaTilt=atan2(1-beta0*cos(phiReal),beta0*sin(phiReal));
				const float_X phi=2*alphaTilt; // Definition of the laser pulse front tilt angle for the laser field below. For beta0=1.0, this is equivalent to our standard definition.
				const float_X eta = PI - phiReal + alphaTilt; // angle between the laser pulse front and the y-axis
				
				const float_X cspeed=::picongpu::SI::SPEED_OF_LIGHT_SI;
				const float_X lambda0=::picongpu::bgrTWTS::SI::WAVE_LENGTH_SI;
				const float_X om0=2*PI*cspeed/lambda0;
				const float_X tauG=(::picongpu::bgrTWTS::SI::PULSE_LENGTH_SI)*2.0; // factor 2 arises from definition convention in laser formula
				const float_X w0=::picongpu::bgrTWTS::SI::WX_SI; // w0 is wx here --> w0 could be replaced by wx
				const float_X rho0=PI*w0*w0/lambda0;
				const float_X wy=::picongpu::bgrTWTS::SI::WY_SI; // Width of TWTS pulse
				const float_X k=2*PI/lambda0;
				const float_X x=pos.x();
				const float_X y=-sin(phiReal)*pos.y()-cos(phiReal)*pos.z();	// RotationMatrix[PI-phiReal].(y,z)
				const float_X z=+cos(phiReal)*pos.y()-sin(phiReal)*pos.z();	// TO DO: For 2 counter-propagation TWTS pulses take +phiReal and -phiReal. Where do we implement this?
				const float_X y1=(float_X)(halfSimSize[2]*::picongpu::SI::CELL_DEPTH_SI)/tan(eta); // halfSimSize[2] --> Half-depth of simulation volume (in z); By geometric projection we calculate the y-distance walkoff of the TWTS-pulse.
				const float_X m=3.; // Fudge parameter to make sure, that TWTS pulse starts to impact simulation volume at low intensity values.
				const float_X y2=(tauG/2*cspeed)/sin(eta)*m; // pulse length projected on y-axis, scaled with "fudge factor" m.
				const float_X y3=::picongpu::bgrTWTS::SI::FOCUS_POS_SI; // Position of maximum intensity in simulation volume along y
				const float_X tdelay= (y1+y2+y3)/(cspeed*beta0);
				const float_X t=time;//-tdelay;
				
				const float_X exprDivInt_3_1=cspeed;
				const float_X exprDivInt_3_2=wy;
				const Complex_float_X exprDivInt_3_3=Complex_float_X(0,1)*rho0 - y*cos(phi) - z*sin(phi);
				const Complex_float_X exprDivInt_3_4=Complex_float_X(0,-1)*cspeed*om0*tauG*tauG - y*cos(phi)/cos(phi/2.)/cos(phi/2.)*tan(phi/2.) - 2*z*tan(phi/2.)*tan(phi/2.);
				const Complex_float_X exprDivInt_3_5=Complex_float_X(0,1)*rho0 - y*cos(phi) - z*sin(phi);

				const Complex_float_X exprE_1_1=(
				-(cspeed*cspeed*k*om0*tauG*tauG*wy*wy*x*x)
				- 2*cspeed*cspeed*om0*t*t*wy*wy*rho0 
				+ Complex_float_X(0,2)*cspeed*cspeed*om0*om0*t*tauG*tauG*wy*wy*rho0
				- 2*cspeed*cspeed*om0*tauG*tauG*y*y*rho0
				+ 4*cspeed*om0*t*wy*wy*z*rho0
				- Complex_float_X(0,2)*cspeed*om0*om0*tauG*tauG*wy*wy*z*rho0
				- 2*om0*wy*wy*z*z*rho0
				- Complex_float_X(0,8)*om0*wy*wy*y*(cspeed*t - z)*z*sin(phi/2.)*sin(phi/2.)
				+ Complex_float_X(0,8)/sin(phi)*(
						+2*z*z*(cspeed*om0*t*wy*wy + Complex_float_X(0,1)*cspeed*y*y - om0*wy*wy*z)
						+ y*(
							+ cspeed*k*wy*wy*x*x
							- Complex_float_X(0,2)*cspeed*om0*t*wy*wy*rho0
							+ 2*cspeed*y*y*rho0
							+ Complex_float_X(0,2)*om0*wy*wy*z*rho0
						)*tan(PI/2-phi)/sin(phi)
					)*sin(phi/2.)*sin(phi/2.)*sin(phi/2.)*sin(phi/2.)
				- Complex_float_X(0,2)*cspeed*cspeed*om0*t*t*wy*wy*z*sin(phi)
				- 2*cspeed*cspeed*om0*om0*t*tauG*tauG*wy*wy*z*sin(phi)
				- Complex_float_X(0,2)*cspeed*cspeed*om0*tauG*tauG*y*y*z*sin(phi)
				+ Complex_float_X(0,4)*cspeed*om0*t*wy*wy*z*z*sin(phi)
				+ 2*cspeed*om0*om0*tauG*tauG*wy*wy*z*z*sin(phi)
				- Complex_float_X(0,2)*om0*wy*wy*z*z*z*sin(phi)
				- 4*cspeed*om0*t*wy*wy*y*rho0*tan(phi/2.)
				+ 4*om0*wy*wy*y*z*rho0*tan(phi/2.)
				+ Complex_float_X(0,2)*y*y*(cspeed*om0*t*wy*wy + Complex_float_X(0,1)*cspeed*y*y - om0*wy*wy*z)*cos(phi)*cos(phi)/cos(phi/2.)/cos(phi/2.)*tan(phi/2.)
				+ Complex_float_X(0,2)*cspeed*k*wy*wy*x*x*z*tan(phi/2.)*tan(phi/2.)
				- 2*om0*wy*wy*y*y*rho0*tan(phi/2.)*tan(phi/2.)
				+ 4*cspeed*om0*t*wy*wy*z*rho0*tan(phi/2.)*tan(phi/2.)
				+ Complex_float_X(0,4)*cspeed*y*y*z*rho0*tan(phi/2.)*tan(phi/2.)
				- 4*om0*wy*wy*z*z*rho0*tan(phi/2.)*tan(phi/2.)
				- Complex_float_X(0,2)*om0*wy*wy*y*y*z*sin(phi)*tan(phi/2.)*tan(phi/2.)
				- 2*y*cos(phi)*(om0*(cspeed*cspeed*(Complex_float_X(0,1)*t*t*wy*wy + om0*t*tauG*tauG*wy*wy + Complex_float_X(0,1)*tauG*tauG*y*y) - cspeed*(Complex_float_X(0,2)*t + om0*tauG*tauG)*wy*wy*z + Complex_float_X(0,1)*wy*wy*z*z) + Complex_float_X(0,2)*om0*wy*wy*y*(cspeed*t - z)*tan(phi/2.) + Complex_float_X(0,1)*(Complex_float_X(0,-4)*cspeed*y*y*z + om0*wy*wy*(y*y - 4*(cspeed*t - z)*z))*tan(phi/2.)*tan(phi/2.))
				)/(2.*exprDivInt_3_1*exprDivInt_3_2*exprDivInt_3_2*exprDivInt_3_3*exprDivInt_3_4);

				const Complex_float_X exprDivRat_1_1=cspeed*om0*tauG*tauG - Complex_float_X(0,8)*y*tan(PI/2-phi)/sin(phi)/sin(phi)*sin(phi/2.)*sin(phi/2.)*sin(phi/2.)*sin(phi/2.) - Complex_float_X(0,2)*z*tan(phi/2.)*tan(phi/2.);
				const Complex_float_X result=(Complex_float_X::cexp(exprE_1_1)*tauG*Complex_float_X::csqrt((cspeed*om0*rho0)/exprDivInt_3_5))/Complex_float_X::csqrt(exprDivRat_1_1);			
				return result.get_real();
			}

		};
		
		class fieldBackgroundB
		{
		public:
			/* Add this additional field for pushing particles */
			static const bool InfluenceParticlePusher = ::picongpu::bgrTWTS::includeTWTSlaser;

			/* We use this to calculate your SI input back to our unit system */
			const float3_64 unitField;
			HDINLINE fieldBackgroundB( const float3_64 unitField ) : unitField(unitField)
			{}

			/** Specify your background field B(r,t) here
			 *
			 * \param cellIdx The total cell id counted from the start at t=0
			 * \param currentStep The current time step
			 * \param halfSimSize Center of simulation volume in number of cells */
			HDINLINE float3_X
			operator()( const DataSpace<simDim>& cellIdx,
						const uint32_t currentStep,
						const DataSpace<simDim> halfSimSize	) const
			{
				const float_X focus_y = ::picongpu::bgrTWTS::SI::FOCUS_POS_SI/::picongpu::SI::CELL_HEIGHT_SI;
				const float3_X helper = float3_X( halfSimSize.x(), focus_y, halfSimSize.z() );
				
				/* For the Yee-Cell shifted fields, obtain the fractional cell index components and add that to the total cell indices. The physical field coordinate origin is in the center of the simulation */
				::PMacc::math::Vector<floatD_X,simDim> bFieldPositions = picongpu::yeeCell::YeeCell::getBFieldPosition();
				bFieldPositions[0] = ((float3_X)cellIdx+bFieldPositions[0]-helper) * float3_X(::picongpu::SI::CELL_WIDTH_SI,::picongpu::SI::CELL_HEIGHT_SI,::picongpu::SI::CELL_DEPTH_SI); // cellIdx(Bx)
				bFieldPositions[1] = ((float3_X)cellIdx+bFieldPositions[1]-helper) * float3_X(::picongpu::SI::CELL_WIDTH_SI,::picongpu::SI::CELL_HEIGHT_SI,::picongpu::SI::CELL_DEPTH_SI); // cellIdx(By)
				bFieldPositions[2] = ((float3_X)cellIdx+bFieldPositions[2]-helper) * float3_X(::picongpu::SI::CELL_WIDTH_SI,::picongpu::SI::CELL_HEIGHT_SI,::picongpu::SI::CELL_DEPTH_SI); // cellIdx(Bz)
				
				const float_X time=currentStep*::picongpu::SI::DELTA_T_SI;
				
				/* specify your E-Field in V/m and convert to PIConGPU units */
				if ( ! ::picongpu::bgrTWTS::includeCollidingTWTS ) {
					// Single TWTS-Pulse
					return float3_X(0.0, (::picongpu::bgrTWTS::SI::AMPLITUDE_SI)*calcTWTSBy(bFieldPositions[1], time, halfSimSize, ::picongpu::bgrTWTS::SI::PHI_SI) / unitField[1], (::picongpu::bgrTWTS::SI::AMPLITUDE_SI)*calcTWTSBz(bFieldPositions[2], time, halfSimSize, ::picongpu::bgrTWTS::SI::PHI_SI) / unitField[1]);
					//return float3_X ( ((float3_X)cellIdx)[0] / unitField[1], ((float3_X)cellIdx)[1] / unitField[1], ((float3_X)cellIdx)[2] / unitField[1] );
				}
				else {
					// Colliding TWTS-Pulse
					return float3_X(0.0, (::picongpu::bgrTWTS::SI::AMPLITUDE_SI)
									*( calcTWTSBy(bFieldPositions[1], time, halfSimSize, +(::picongpu::bgrTWTS::SI::PHI_SI))
									  +calcTWTSBy(bFieldPositions[1], time, halfSimSize, -(::picongpu::bgrTWTS::SI::PHI_SI))
									)/ unitField[1]
							, (::picongpu::bgrTWTS::SI::AMPLITUDE_SI)
									*( calcTWTSBz(bFieldPositions[2], time, halfSimSize, +(::picongpu::bgrTWTS::SI::PHI_SI))
									  +calcTWTSBz(bFieldPositions[2], time, halfSimSize, -(::picongpu::bgrTWTS::SI::PHI_SI))
									)/ unitField[1]);
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
				const float_X beta0=::picongpu::bgrTWTS::SI::BETA0_SI; // propagation speed of overlap normalized to the speed of light. [Default: beta0=1.0]
				const float_X alphaTilt=atan2(1-beta0*cos(phiReal),beta0*sin(phiReal));
				const float_X phi=2*alphaTilt; // Definition of the laser pulse front tilt angle for the laser field below. For beta0=1.0, this is equivalent to our standard definition.
				const float_X eta = PI - phiReal + alphaTilt; // angle between the laser pulse front and the y-axis
				
				const float_X cspeed=::picongpu::SI::SPEED_OF_LIGHT_SI;
				const float_X lambda0=::picongpu::bgrTWTS::SI::WAVE_LENGTH_SI;
				const float_X om0=2*PI*cspeed/lambda0;
				const float_X tauG=(::picongpu::bgrTWTS::SI::PULSE_LENGTH_SI)*2.0; // factor 2 arises from definition convention in laser formula
				const float_X w0=::picongpu::bgrTWTS::SI::WX_SI; // w0 is wx here --> w0 could be replaced by wx
				const float_X rho0=PI*w0*w0/lambda0;
				const float_X wy=::picongpu::bgrTWTS::SI::WY_SI; // Width of TWTS pulse
				const float_X k=2*PI/lambda0;
				const float_X x=pos.x();
				const float_X y=-sin(phiReal)*pos.y()-cos(phiReal)*pos.z();	// RotationMatrix[PI-phiReal].(y,z)
				const float_X z=+cos(phiReal)*pos.y()-sin(phiReal)*pos.z();	// TO DO: For 2 counter-propagation TWTS pulses take +phiReal and -phiReal. Where do we implement this?
				const float_X y1=(float_X)(halfSimSize[2]*::picongpu::SI::CELL_DEPTH_SI)/tan(eta); // halfSimSize[2] --> Half-depth of simulation volume (in z); By geometric projection we calculate the y-distance walkoff of the TWTS-pulse.
				const float_X m=3.; // Fudge parameter to make sure, that TWTS pulse starts to impact simulation volume at low intensity values.
				const float_X y2=(tauG/2*cspeed)/sin(eta)*m; // pulse length projected on y-axis, scaled with "fudge factor" m.
				const float_X y3=::picongpu::bgrTWTS::SI::FOCUS_POS_SI; // Position of maximum intensity in simulation volume along y
				const float_X tdelay= (y1+y2+y3)/(cspeed*beta0);
				const float_X t=time;//-tdelay;
				
				const float_X exprDivInt_3_1=cspeed;
				const float_X exprDivInt_3_2=wy;
				const Complex_float_X exprDivInt_3_3=rho0 + Complex_float_X(0,1)*y*cos(phi) + Complex_float_X(0,1)*z*sin(phi);
				const Complex_float_X exprDivInt_3_4=cspeed*om0*tauG*tauG + Complex_float_X(0,2)*(-z - y*tan(PI/2-phi))*tan(phi/2.)*tan(phi/2.);
				const float_X exprDivInt_3_5=2;
				const float_X exprDivInt_3_6=rho0;
				const Complex_float_X exprDivInt_2_1=Complex_float_X(0,1)*rho0 - y*cos(phi) - z*sin(phi);
				const Complex_float_X exprE_1_1=-1.0*(
				cspeed*cspeed*k*om0*tauG*tauG*wy*wy*x*x
				+ 2*cspeed*cspeed*om0*t*t*wy*wy*rho0
				- Complex_float_X(0,2)*cspeed*cspeed*om0*om0*t*tauG*tauG*wy*wy*rho0
				+ 2*cspeed*cspeed*om0*tauG*tauG*y*y*rho0
				- 4*cspeed*om0*t*wy*wy*z*rho0
				+ Complex_float_X(0,2)*cspeed*om0*om0*tauG*tauG*wy*wy*z*rho0
				+ 2*om0*wy*wy*z*z*rho0
				+ 4*cspeed*om0*t*wy*wy*y*rho0*tan(phi/2.)
				- 4*om0*wy*wy*y*z*rho0*tan(phi/2.)
				- Complex_float_X(0,2)*cspeed*k*wy*wy*x*x*z*tan(phi/2.)*tan(phi/2.)
				+ 2*om0*wy*wy*y*y*rho0*tan(phi/2.)*tan(phi/2.)
				- 4*cspeed*om0*t*wy*wy*z*rho0*tan(phi/2.)*tan(phi/2.)
				- Complex_float_X(0,4)*cspeed*y*y*z*rho0*tan(phi/2.)*tan(phi/2.)
				+ 4*om0*wy*wy*z*z*rho0*tan(phi/2.)*tan(phi/2.)
				- Complex_float_X(0,2)*cspeed*k*wy*wy*x*x*y*tan(PI/2-phi)*tan(phi/2.)*tan(phi/2.)
				- 4*cspeed*om0*t*wy*wy*y*rho0*tan(PI/2-phi)*tan(phi/2.)*tan(phi/2.)
				- Complex_float_X(0,4)*cspeed*y*y*y*rho0*tan(PI/2-phi)*tan(phi/2.)*tan(phi/2.)
				+ 4*om0*wy*wy*y*z*rho0*tan(PI/2-phi)*tan(phi/2.)*tan(phi/2.)
				+ 2*z*sin(phi)*(
					om0*(cspeed*cspeed*(Complex_float_X(0,1)*t*t*wy*wy + om0*t*tauG*tauG*wy*wy + Complex_float_X(0,1)*tauG*tauG*y*y) - cspeed*(Complex_float_X(0,2)*t + om0*tauG*tauG)*wy*wy*z + Complex_float_X(0,1)*wy*wy*z*z)
					+ Complex_float_X(0,2)*om0*wy*wy*y*(cspeed*t - z)*tan(phi/2.) + Complex_float_X(0,1)*(Complex_float_X(0,-2)*cspeed*y*y*z + om0*wy*wy*(y*y - 2*(cspeed*t - z)*z))*tan(phi/2.)*tan(phi/2.)
					)
				+ 2*y*cos(phi)*(
					om0*(cspeed*cspeed*(Complex_float_X(0,1)*t*t*wy*wy + om0*t*tauG*tauG*wy*wy + Complex_float_X(0,1)*tauG*tauG*y*y) - cspeed*(Complex_float_X(0,2)*t + om0*tauG*tauG)*wy*wy*z + Complex_float_X(0,1)*wy*wy*z*z)
					+ Complex_float_X(0,2)*om0*wy*wy*y*(cspeed*t - z)*tan(phi/2.)
					+ Complex_float_X(0,1)*(Complex_float_X(0,-4)*cspeed*y*y*z + om0*wy*wy*(y*y - 4*(cspeed*t - z)*z) - 2*y*(cspeed*om0*t*wy*wy + Complex_float_X(0,1)*cspeed*y*y - om0*wy*wy*z)*tan(PI/2-phi))*tan(phi/2.)*tan(phi/2.)
					)
				)/(2.*exprDivInt_3_1*exprDivInt_3_2*exprDivInt_3_2*exprDivInt_3_3*exprDivInt_3_4);

				const Complex_float_X exprDivInt_1_1=Complex_float_X(0,-1)*cspeed*om0*tauG*tauG + (-z - y*tan(PI/2-phi))*tan(phi/2.)*tan(phi/2.)*exprDivInt_3_5;
				const Complex_float_X exprDivRat_1_1=(cspeed*(cspeed*om0*tauG*tauG + Complex_float_X(0,2)*(-z - y*tan(PI/2-phi))*tan(phi/2.)*tan(phi/2.)))/(om0*exprDivInt_3_6);
				const Complex_float_X result=(Complex_float_X::cexp(exprE_1_1)*tauG/cos(phi/2.)/cos(phi/2.)*(rho0 + Complex_float_X(0,1)*y*cos(phi) + Complex_float_X(0,1)*z*sin(phi))*(Complex_float_X(0,2)*cspeed*t + cspeed*om0*tauG*tauG - Complex_float_X(0,4)*z + cspeed*(Complex_float_X(0,2)*t + om0*tauG*tauG)*cos(phi) + Complex_float_X(0,2)*y*tan(phi/2.))*Complex_float_X::cpow(exprDivInt_2_1,-1.5))/(2.*exprDivInt_1_1*Complex_float_X::csqrt(exprDivRat_1_1));

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
				const float_X beta0=::picongpu::bgrTWTS::SI::BETA0_SI; // propagation speed of overlap normalized to the speed of light. [Default: beta0=1.0]
				const float_X alphaTilt=atan2(1-beta0*cos(phiReal),beta0*sin(phiReal));
				const float_X phi=2*alphaTilt; // Definition of the laser pulse front tilt angle for the laser field below. For beta0=1.0, this is equivalent to our standard definition.
				const float_X eta = PI - phiReal + alphaTilt; // angle between the laser pulse front and the y-axis
				
				const float_X cspeed=::picongpu::SI::SPEED_OF_LIGHT_SI;
				const float_X lambda0=::picongpu::bgrTWTS::SI::WAVE_LENGTH_SI;
				const float_X om0=2*PI*cspeed/lambda0;
				const float_X tauG=(::picongpu::bgrTWTS::SI::PULSE_LENGTH_SI)*2.0; // factor 2 arises from definition convention in laser formula
				const float_X w0=::picongpu::bgrTWTS::SI::WX_SI; // w0 is wx here --> w0 could be replaced by wx
				const float_X rho0=PI*w0*w0/lambda0;
				const float_X wy=::picongpu::bgrTWTS::SI::WY_SI; // Width of TWTS pulse
				const float_X k=2*PI/lambda0;
				const float_X x=pos.x();
				const float_X y=-sin(phiReal)*pos.y()-cos(phiReal)*pos.z();	// RotationMatrix[PI-phiReal].(y,z)
				const float_X z=+cos(phiReal)*pos.y()-sin(phiReal)*pos.z();	// TO DO: For 2 counter-propagation TWTS pulses take +phiReal and -phiReal. Where do we implement this?
				const float_X y1=(float_X)(halfSimSize[2]*::picongpu::SI::CELL_DEPTH_SI)/tan(eta); // halfSimSize[2] --> Half-depth of simulation volume (in z); By geometric projection we calculate the y-distance walkoff of the TWTS-pulse.
				const float_X m=3.; // Fudge parameter to make sure, that TWTS pulse starts to impact simulation volume at low intensity values.
				const float_X y2=(tauG/2*cspeed)/sin(eta)*m; // pulse length projected on y-axis, scaled with "fudge factor" m.
				const float_X y3=::picongpu::bgrTWTS::SI::FOCUS_POS_SI; // Position of maximum intensity in simulation volume along y
				const float_X tdelay= (y1+y2+y3)/(cspeed*beta0);
				const float_X t=time;//-tdelay;
				
				const Complex_float_X exprDivInt_5_1=-(cspeed*z) - cspeed*y*tan(PI/2-phi) + Complex_float_X(0,1)*cspeed*rho0/sin(phi);
				const Complex_float_X exprDivInt_5_2=Complex_float_X(0,1)*rho0 - y*cos(phi) - z*sin(phi);
				const float_X exprDivInt_5_4=wy;
				const Complex_float_X exprDivInt_5_12=exprDivInt_5_2*cspeed;
				const float_X exprDivInt_5_18=cspeed;
				const Complex_float_X exprDivInt_5_19=cspeed*om0*tauG*tauG - Complex_float_X(0,1)*y*cos(phi)/cos(phi/2.)/cos(phi/2.)*tan(phi/2.) - Complex_float_X(0,2)*z*tan(phi/2.)*tan(phi/2.);
				const Complex_float_X exprPower=2*cspeed*t - Complex_float_X(0,1)*cspeed*om0*tauG*tauG - 2*z + 8*y/sin(phi)/sin(phi)/sin(phi)*sin(phi/2.)*sin(phi/2.)*sin(phi/2.)*sin(phi/2.) - 2*z*tan(phi/2.)*tan(phi/2.);

				const Complex_float_X exprE_1_1=(
				(om0*y*rho0/cos(phi/2.)/cos(phi/2.)/cos(phi/2.)/cos(phi/2.))/exprDivInt_5_1 
				- (Complex_float_X(0,2)*k*x*x)/exprDivInt_5_2 
				- (Complex_float_X(0,1)*om0*om0*tauG*tauG*rho0)/exprDivInt_5_2
				- (Complex_float_X(0,4)*y*y*rho0)/(exprDivInt_5_4*exprDivInt_5_4*exprDivInt_5_2)
				+ (om0*om0*tauG*tauG*y*cos(phi))/exprDivInt_5_2
				+ (4*y*y*y*cos(phi))/(exprDivInt_5_4*exprDivInt_5_4*exprDivInt_5_2)
				+ (om0*om0*tauG*tauG*z*sin(phi))/exprDivInt_5_2
				+ (4*y*y*z*sin(phi))/(exprDivInt_5_4*exprDivInt_5_4*exprDivInt_5_2)
				+ (Complex_float_X(0,2)*om0*y*y*cos(phi)/cos(phi/2.)/cos(phi/2.)*tan(phi/2.))/exprDivInt_5_12
				+ (om0*y*rho0*cos(phi)/cos(phi/2.)/cos(phi/2.)*tan(phi/2.))/exprDivInt_5_12
				+ (Complex_float_X(0,1)*om0*y*y*cos(phi)*cos(phi)/cos(phi/2.)/cos(phi/2.)*tan(phi/2.))/exprDivInt_5_12
				+ (Complex_float_X(0,4)*om0*y*z*tan(phi/2.)*tan(phi/2.))/exprDivInt_5_12
				- (2*om0*z*rho0*tan(phi/2.)*tan(phi/2.))/exprDivInt_5_12
				- (Complex_float_X(0,2)*om0*z*z*sin(phi)*tan(phi/2.)*tan(phi/2.))/exprDivInt_5_12
				- (om0*exprPower*exprPower)/(exprDivInt_5_18*exprDivInt_5_19)
				)/4.;
						
				const Complex_float_X exprDivRat_1_1=cspeed*om0*tauG*tauG - Complex_float_X(0,1)*y*cos(phi)/cos(phi/2.)/cos(phi/2.)*tan(phi/2.) - Complex_float_X(0,2)*z*tan(phi/2.)*tan(phi/2.);
				const Complex_float_X result=(Complex_float_X(0,2)*Complex_float_X::cexp(exprE_1_1)*tauG*tan(phi/2.)*(cspeed*t - z + y*tan(phi/2.))*Complex_float_X::csqrt((om0*rho0)/exprDivInt_5_12))/Complex_float_X::cpow(exprDivRat_1_1,1.5);

				return result.get_real();
			}
			
		};
		
		class fieldBackgroundJ
		{
		public:
			/* Add this additional field? */
			static const bool activated = false;

			/* We use this to calculate your SI input back to our unit system */
			const float3_64 unitField;
			HDINLINE fieldBackgroundJ( const float3_64 unitField ) : unitField(unitField)
			{}

			/** Specify your background field J(r,t) here
			 *
			 * \param cellIdx The total cell id counted from the start at t=0
			 * \param currentStep The current time step
			 * \param halfSimSize Center of simulation volume in number of cells */
			HDINLINE float3_X
			operator()( const DataSpace<simDim>& cellIdx,
						const uint32_t currentStep,
						const DataSpace<simDim> halfSimSize	) const
			{
				/* example: periodicity of 20 microns (=2.0e-5 m) */
				const float_64 period_SI(20.0e-6);
				/* calculate cells -> SI -> m to microns*/
				const float_64 y_SI = cellIdx.y() * ::picongpu::SI::CELL_HEIGHT_SI * 1.0e6;
				/* note: you can also transform the time step to seconds by
				 *       multiplying with DELTA_T_SI */

				/* specify your J-Field in A/m^2 and convert to PIConGPU units */
				const float_X sinArg = precisionCast<float_X>( y_SI / period_SI * 2.0 * PI );
				return float3_X(0.0, math::cos( sinArg ) / unitField[1], 0.0);
			}
		};
	} // namespace bgrTWTS
} // namespace picongpu
