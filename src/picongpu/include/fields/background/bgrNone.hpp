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

/** Do not load external fields
 *
 */
namespace picongpu
{
    namespace bgrNone
    {
		class fieldBackgroundE
		{
		public:
			/* Add this additional field for pushing particles */
			static const bool InfluenceParticlePusher = false;

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
				return float3_X( 0.0, 0.0, 0.0 );				
			}
			
		};
		
		class fieldBackgroundB
		{
		public:
			/* Add this additional field for pushing particles */
			static const bool InfluenceParticlePusher = false;

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
				return float3_X(0.0, 0.0, 0.0 );
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
				return float3_X( 0.0, 0.0, 0.0 );
			}
		};
	} // namespace bgrNone
} // namespace picongpu
