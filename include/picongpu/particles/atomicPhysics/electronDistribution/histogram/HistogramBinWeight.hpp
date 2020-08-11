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
 * This file defines a struct that stores information of a single bin of the
 * electron energy histogram, f(E), of electron density f over electron energy E
 * 
 * Template Parameters:
 *  T_Energy ... data type used for energy in storage
 *  T_Weight ... data type used for weight in storage
 *
 * Members(private):
 *  centralEnergy  ... central energy E_i of the energy interval
 *  deltaEnergy    ... width of energy interval
 *
 * actuall data stored in histogram:
 *  weight         ... sum of weight values of all electrons inside this energy
 *                      bin, Kahan algorithm used to reduce floating point
 *                      summation error,
                   see https://en.wikipedia.org/wiki/Kahan_summation_algorithm
 *  compensation   ... compensation variable required for Kahan summation
 *
 * Member functions(public):
 *  HistogramBinWeight(             constructor of HistogramBinWeight
 *    T_Energy centralE, T_Energy deltaE )
 *  void addWeight( T_Weight w )    add w to weight using Kahan summation
 *  void reset()                    reset HistogramBinWeight to initial condition, 
 *  void removeWeight( T_Weight w ) remove w from weight
 *  bool checkEmpty()               returns true if bin is empty
 *
 *  Promoted getWeight()            return weight in bin, Promoted depends on
 *                                  specialisation,
 *                                  Promoted \in {T_Weight, 'long' T_Weight}
 */

#pragma once

#include <list>
#include <limits>
#include <stdexcept>


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

// standard implementation:     reusing T_Weight for weight
template < typename T_Energy, typename T_Weight>
class HistogramBinWeight
{
    /** general definition of HistogramBinWeight
    *
    * This implementation uses T_Weight as data type of accumulated weights,
    *
    * This would result in an overflow, if individual weights are large or a
    * large number of particles are in this bin, instead a this class will
    * throw a runtime overflow exception to alert the user.
    * It is the user responsibility to choose a large enough data type T_Weight
    * to hold the expected weight, OR use one of the predefined/(define a new)
    * specializations.
    */

    private:

        // central energy of bin and energy width
        T_Energy centralEnergy;
        T_Energy deltaEnergy;

        // summed weights of all electrons inside this energy bin
        T_Weight weight;
        // Kahan summation algorithm compensation varaible
        T_Weight compensation;

    public:

        HistogramBinWeight( T_Energy centralE, T_Energy deltaE)
        {
            /** constructor of HistogramBinWeight
            *
            * initialise central enrgy with E, and both compensation and weight
            * with 0.
            */
            this->centralEnergy = centralE;
            this->deltaEnergy = deltaE;
            this-> weight = 0;
            this->compensation = 0;
        }

        bool checkEmpty()
        {
            if (this-> weight == 0) { return true; }
            return false;
        }

        T_Weight getWeight()
        {
            /** get current weight
            */
            return this->weight;
        }

        void addWeight(T_Weight w)
        {
            /** add weight w using Karhan algorithm to bin
            *
            * overflow resulsts in a runtime error
            */

            T_Weight y, t;

            // check assumptions for w
            PMACC_ASSERT_MSG(
                w < 0,
                "weights should not be < 0, "
                );

            // check for overflow on runtime
            if ( w > std::numeric_limits< T_Weight >::max - this->weight )
            {
                throw std::runtime_error
                (
                    "overflow in weight of histogramBinWeight, see documentation"
                    "of HistogramBinWeight class of atomicPhysics for further"
                    "information on what happened and how to avoid this"
                );
            }

            // Karhan summation
            y = w - this->compensation;     // apply prevoius compensation
            t = this->weight + y;           // add to total
            this->compensation = ( t - this->weight ) - y;  // calculate new
                                                            // compensation
            this->weight = t;               // set new total

        }

        void reset()
        {
            /** resets HistrogragmBin
            *
            *the central energy is not changed
            */

            this->weight = 0;
            this->compensation = 0;
        }

        void removeWeight(T_Weight w)
        {
            /** removes the weight w from the bin
            */

            PMACC_ASSERT_MSG(
                this->weight < w,
                "tried to remove more weight than available"
                )

            this->weight -= w
        }
}

/* specialisation for float_x as particle data type weight, uses float to store
 * cumulated weight of bin
 */
template < typename T_Energy >
class HistogramBinWeight < T_Energy, float_X >
{
    /** HistogramBinWeight specialization for T_Weight = float_X
    */

    private:

        // central energy of bin, width stored seperately
        T_Energy centralEnergy;

        // summed weights of all electrons inside this energy bin
         float weight;

        // necessary for Kahan Summation Alogorithm, see
        float compensation;

    public:
        HistogramBinWeight( T_Energy centralE, T_Energy deltaE )
        {
            /** constructor of HistogramBinWeight
            */
            this->centralEnergy = centralE;
            this->deltaEnergy = deltaE;
            this->weight = 0.0;
            this->compensation = 0.0;
        }

        bool checkEmpty()
        {
            if (this-> weight == 0) { return true; }
            return false;
        }

        float getWeight()
        {
            /** get current weight
            */
            return this->weight;
        }

        void addWeight(float_X w)
        {
            /** add weight w using Karhan algorithmus
            */

            float y, t;

            // check assumptions for w
            PMACC_ASSERT_MSG(
                w < 0,
                "weights should not be < 0, "
                );

            // check for overflow on runtime
            if ( w > std::numeric_limits< float >::max - this->weight )
            {
                throw std::runtime_error
                (
                    "overflow in weight of histogramBinWeight, see documentation"
                    "of HistogramBin class of atomicPhysics for further"
                    "information on what happened and how to avoid this"
                );
            }

            // Karhan summation

            // apply prevoius compensation
            y = static_cast<float>(w) - this->compensation;
            // add to total
            t = this->weight + y;
            // calculate new compensation
            this->compensation = ( t - this->weight ) - y;
            // set new total
            this->weight = t;

        }

        void reset()
        {
            /** resets HistrogragmBin
            */

            this->weight = 0.0;
            this->compensation = 0.0;
        }

        void removeWeight(float_X w)
        {
            /** removes the weight w from the bin
            */

            PMACC_ASSERT_MSG(
                this->weight < w,
                "tried to remove more weight than available"
                )

            this->weight -= static_cast< float >( w )
        }
}

} // namespace histogram
} // namespace electronDistribution
} // namespace atomic Physics
} // namespace particles
} // namespace picongpu