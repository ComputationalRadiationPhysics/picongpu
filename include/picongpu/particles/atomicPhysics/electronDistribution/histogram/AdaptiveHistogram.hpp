/* Copyright 2019-2020 Brian Marre
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

/** @file
 * This file defines the adaptive binning algorithm used for the electron
 * spectrum representation in rate calculations
 */

#pragma once

#include <list>


namespace picongpu
{
namespace particles
{
namespace atomicPhysics
{
namespace electronDistribution
{
namespace histogram
{

template< typename T_Energy, typename T_Weight, uint8_t order >
class AdaptiveHistogram
{

    /** this class defines the histogram of the electron spectrum
    *
    * Members():
    *
    * Member Functions(public):
    *   void binParticle( T_Energy E, T_Weight w )
    *   void mergeHistograms( adaptiveHistogram< T_Weight, T_Energy> h2 )
    *   T_weight getWeight( T_Energy E )
    */

    private:
        std::list< T_HistogramBin< T_Energy, T_Weight > > bins;

        static float vTermDerivation( T_Energy E, uint8_t order )
        {
            
        crossSectionTermDerivation()
        static scaledChebyshevNodes()
        calculateRelativeError()


    public:
        // make template parameters available for later use
        static constexpr using dataTypeEnergy = T_Energy;
        static constexpr using dataTypeWeightElement = T_Weight;

        void binParticle( T_Energy E, T_Weight w )
        {
            
        }

        void mergeHistograms( adaptiveHistogram< T_Weight, T_Energy> h2 )
        {
            
        }

        T_weight getWeight( T_Energy E )
        {
            
        }
}


} // namespace histogram
} // namespace electronDistribution
} // namespace atomic Physics
} // namespace particles
} // namespace picongpu
