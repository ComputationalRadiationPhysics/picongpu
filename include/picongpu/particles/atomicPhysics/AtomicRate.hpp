/* Copyright 2017-2020 Axel Huebl, Brian Marre, Sudhir Sharma
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

#inlcude "picongpu/param/physicalConstants.param"
#include <pmacc/algorithms/math.hpp>


#pragma once

/** rate calculation from given atomic data, extracted from flylite, based on FLYchk
 *
 *
 * References:
 * - Axel Huebl
 *  flylite, not yet published
 *
 *  - R. Mewe.
 *  "Interpolation formulae for the electron impact excitation of ions in
 *  the H-, He-, Li-, and Ne-sequences."
 *  Astronomy and Astrophysics 20, 215 (1972)
 *
 *  - H.-K. Chung, R.W. Lee, M.H. Chen.
 *  "A fast method to generate collisional excitation cross-sections of
 *  highly charged ions in a hot dense matter"
 *  High Energy Dennsity Physics 3, 342-352 (2007)
 */

#include <pmaccc/algrotihms/math.hpp>
#include "picongpu/param/physicalConstants"
#inlcude "picongpu/particles/atomicPhysics/stateRepresentation/ConfigNumber"

#pragma once

namespace picongpu
{
namespace particles
{
namespace atomicPhysics
{
    /** functor class containing calculation formulas of rates and crossections
     *
     * @tparam T_TypeIndex ... data type of atomic state index used in configNumber, unitless
     * @tparam T_numLevels ... number of atomic levels modelled in configNumber, unitless
     * @tparam T_AtomicDataBox ... type of atomic data box used, stores actual basic
     *      atomic input data
     * BEWARE: atomic data box input data is assumed to be in eV
     */
    template<
        typename T_AtomicDataBox
        typename T_TypeIndex,
        uint8_t T_numLevels,
    >
    class AtomicRate
    {
     public:
        // shorthands
        static constexpr using Idx = T_TypeIndex;
        static constexpr using AtomicDataBox = T_AtomicDataBox;

        // datatype of occupation number vector
        static constexpr using LevelVector = pmacc::math::Vector<
            uint8_t,
            T_numLevels
        >; // unitless

        // type of storage object of atomic state, access to conversion methods
        static constexpr using ConfigNumber =
            picongpu::particles::atomicPhysics::stateRepresentation::ConfigNumber<
                Idx,
                T_numlevels
            >;

    private:

        /** binomial coefficient calculated using partial pascal triangle
         *
         * BEWARE: return type not large enough for complete range of values
         *      should be no problem in flychk data since largest value ~10^10
         *      will become problem if all possible states are used
         *
         * TODO: add description of iteration,
         * - algorithm tested against scipy.specia.binomial
         *
         * Source: https://www.tutorialspoint.com/binomial-coefficient-in-cplusplus;
         *  22.11.2019
         */
        DINLINE static uint64_t binomialCoefficients(
            uint8_t const n,
            uint8_t const k
            ) const
        {
            uint64_t result[ k + 1u ];

            // init with zero, BEWARE: algorithm depends on zero init
            for ( uint8_t i = 1; i <= k; i++ )
            {
                result[ i ] = 0;
            }

            // init with ( binomial(0,0) )
            result[ 0u ] = 1u;

            for ( uint8_t i = 1u; i <= n; i++ )
            {
                for ( uint8_t j = pmacc::algorithms::math::min(i, k); j > 0u; j-- )
                    result[ j ] = result[ j ] + result[ j - 1 ];
            }

            return result[ k ];
        }

        // number of different atomic configurations in an atomic state
        // @param idx ... index of atomic state, unitless
        // return unit: unitless
        DINLINE static uint64_t Multiplicity( Idx idx ) const
        {
            LevelVector levelVector = ConfigNumber::LevelVector( idx ); // unitless

            uint64_t result = 1u;

            for ( uint8_t i = 0u; i < T_numLevels; i++ )
            {
                result *= binomialCoefficients(
                    static_cast< uint8_t >( 2u * pmacc::math::algorithms::pow( i ,2 ) ),
                    *levelVector[ i ]
                    ); // unitless
            }

            return result; // unitless
        }

        /** gaunt factor like suppression of crosssection
         *
         * @param energyDifference ... difference of energy between atomic states, unit: ATOMIC_UNIT_ENERGY
         * @param energyElectron ... energy of electron, unit ATOMIC_UNIT_ENERGY
         * @param indexTransition ... internal index of transition in atomicDataBox
         *      use findIndexTransition method of atomicDataBox and screen for not found value
         *      BEWARE: method assumes that indexTransition is valid, undefined behaviour otherwise
         *
         * return unit: unitless
         */
        DINLINE static float_X gauntFactor(
            float_X energyDifference,   // unit: E
            float_X energyElectron,     // unit: E
            uint32_t indexTransition,   // unitless
            AtomicDataBox atomicDataBox
            ) const
        {
            // get gaunt coeficients, unit: unitless
            float_X const A = atomicDataBox.getCxin1( indexTransition );
            float_x const B = atomicDataBox.getCxin2( indexTransition );
            float_X const C = atomicDataBox.getCxin3( indexTransition );
            float_X const D = atomicDataBox.getCxin4( indexTransition );
            float_X const a = atomicDataBox.getCxin5( indexTransition );

            // calculate gaunt Factor
            float_X const U = energyElectron / energyDifference; // unit: unitless
            float_X const g = A * math::log(U) + B + C / ( U + a ) + D /
                pmacc::algorithms::math::pow( U + a, 2 ); // unitless

            return g * ( U > 1.0 ); // unitless
        }

    public:

        // return unit: J, SI
        DINLINE static float_X energyDifference(
            Idx const oldIdx,   // unitless
            Idx const newIdx,   // unitless
            AtomicDataBox atomicDataBox
            )
        {
            return ( atomicDataBox( newIdx ) - atomicDataBox( oldIdx )
                ) * picongpu::UNITCONV_eV_to_Joule;
        }

        /** @param energyElectron ... kinetic energy only, unit: ATOMIC_UNIT_ENERGY
         * return unit: m^2
         */ 
        DINLINE static float_X collisionalExcitationCrosssection(
            Idx const oldIdx,   // unitless
            Idx const newIdx,   // unitless
            float_X energyElectron,     // unit: ATOMIC_UNIT_ENERGY
            AtomicDataBox atomicDataBox
            ) const
        {
            // unit conversion to SI
            float_X energyElectron_SI = energyElectron * picongpu::SI::ATOMIC_UNIT_ENERGY;
            // unit: J, SI

            // energy difference between atomic states
            // J <- (eV - eV) * eV_to_J
            float_X energyDifference_SI = energyDifference(
                oldIdx,
                newIdx,
                atomicDataBox
                ); // unit: J, SI

            uint32_t indexTransition; // unitless
            float_X statisticalRatio; // unitless

            if ( energyDifference_SI < 0._X )
            {
                // deexcitation
                energyDifference_SI = - energyDifference_SI; // unit: J, SI

                //collisional absorption obscillator strength of transition [unitless]
                indexTransition = atomicDataBox.findTransition(
                    newIdx, // lower atomic state Idx
                    oldIdx  // upper atomic state Idx
                    );

                // unitless/unitless * (J + J) / J = unitless
                Ratio = static_cast< float_X >(
                    static_cast< float_64 >( Multiplicity( newIdx ) ) /
                    static_cast< float_64 >( Multiplicity( oldIdx ) )
                    ) * ( energyElectron_SI + energyDifference_SI ) / energyElectron_SI; // unitless

                energyElectron_SI = energyElectron_SI + energyDifference_SI; // unit; J, SI
            }
            else
            {
                // excitation
                //collisional absorption obscillator strength of transition [unitless]
                indexTransition = atomicDataBox.findTransition(
                    oldIdx, // lower atomic state Idx, unitless
                    newIdx  // upper atomic state Idx, unitless
                    );

                Ratio = 1._X; // unitless
            }

            // BEWARE: input data may be incomplete
            // TODO: implement better fallback calculation

            // check whether transition index exists
            if ( indexTransition == atomicDataBox.getNumTransitions() )
                // fallback
                return 0._X; // 0 crossection for non existing transition, unit: m^2, SI

            // unitless * unitless = unitless
            float_X const collisionalOscillatorStrength = Ratio *
                atomicDataBox.getCollisionalOscillatorStrength( indexTransition ); // unitless

            // physical constants
            // (unitless * m)^2 / unitless = m^2
            float_X c0_SI = float_X( 8._X *
                pmacc::algorithms::math::pow( picongpu::pi * picongpu::BOHR_RADIUS, 2 ) /
                pmacc::algorithms::math::sqrt( 3._X) ); // uint: m^2, SI

            // m^2 * (J/J)^2 * unitless * J/J * unitless<-[ J, J, unitless, unitless ] = m^2
            return c0_SI *
                pmacc::algorithms::math::pow(
                    ( picongpu::SI::ATOMIC_UNIT_ENERGY / 2._X )
                    / energyDifference_SI,
                    2
                    ) *
                collisionalOscillatorStrength *
                ( energyDifference_SI / energyElectron_SI ) *
                gauntFactor(
                    energyDifference_SI,
                    energyElectron_SI,
                    indexTransition,
                    atomicDataBox
                    ); // unit: m^2, SI
        }

        // @param energyElectron ... kinetic energy only, unit: ATOMIC_UNIT_ENERGY
        //return unit: m^2, SI
        template< typename Idx >
        DINLINE static float_X totalCrossection(
            float_X energyElectron,     // unit: ATOMIC_UNIT_ENERGY
            AtomicDataBox atomicDataBox
            ) const
        {
            float_X result = 0._X; // unit: m^2, SI

            Idx lowerIdx;
            Idx upperIdx;

            for ( uint32_t i = 0u; i < atomicDataBox.getNumTransitions(); i++ )
            {
                upperIdx = atomicDataBox.getUpperIdx( i );
                lowerIdx = atomicDataBox.getLowerIdx( i );

                // excitation crossection
                result += collisionalExcitationCrosssection(
                    lowerIdx,   // unitless
                    upperIdx,   // unitless
                    energyElectron, // unit: ATOMIC_UNIT_ENERGY
                    atomicDataBox
                    ); // unit: m^2, SI

                // deexcitation crosssection
                result += collisionalExcitationCrosssection(
                    upperIdx,
                    lowerIdx,
                    energyElectron,
                    atomicDataBox
                    ); // unit: m^2, SI
            }

            return result; // unit: m^2, SI
        }

        /** rate function
         * uses 1th order integration <-> a = 0, => T_minOrderApprox = 1
         * TODO: implement higher order integration
         * TODO: change density to internal units
         * TODO: change return unit to internal units
         *
         * @param energyElectron ... kinetic energy only, unit: ATOMIC_UNIT_ENERGY
         * @param energyElectronBinWidth ... unit: ATOMIC_UNIT_ENERGY
         * @param densityElectron ... unit: 1/(m^3 * J)
         * @param atomicDataBox ... acess to input atomic data
         *
         * return unit: 1/s ... SI
         */
        DINLINE static float_X Rate( )(
            Idx const oldIdx,   // old atomic state
            Idx const newIdx,   // new atomic state
            float_X const energyElectron,   // unit: ATOMIC_UNIT_ENERGY
            float_X const energyElectronBinWidth, // unit: ATOMIC_UNIT_ENERGY
            float_X const densityElectrons, // unit: 1/(m^3*J), SI
            AtomicData const atomicDataBox
            ) const // return unit: 1/s, SI
        {
            using mathFunc = pmacc::algorithms::math;

            //constants in SI
            constexpr float_64 c_SI = picongpu::SI::SPEED_OF_LIGHT_SI; // unit: m/s, SI
            constexpr float_64 m_e_SI = picongpu::SI::ELECTRON_MASS_SI; // unit: kg, SI

            const float_64 E_e_SI = energyElectron * picongpu::UNITCONV_AU_eV * UNITCONV_eV_Joule;
                // unit: J, SI
            const float_64 dE_SI = energyElectronBinWidth * picongpu::UNITCONV_AU_eV * UNITCONV_eV_Joule;
                // unit: J, SI

            float_X sigma_SI = collisionalExcitationCrosssection(
                    oldState,   // unitless
                    newState,   // unitless
                    energyElectron, // unit: ATOMIC_UNIT_ENERGY
                    atomicDataBox
                ); // unit: (m^2), SI

            // J * m^2 * 1/(m^3*J) * m/s * sqrt( unitless - [ ( (kg*m^2/s^2)/J )^2 = Nm/J = J/J = unitless ] ) = J/J m^3/m^3 * 1/s
            return dE_SI * sigma_SI * densityElectron * c_SI *
                mathFunc::sqrt( 1 - mathFunc::pow(
                    1._X / (1._X + E_e_SI / ( m_e_SI * mathFunc::pow( c_SI, 2 ) ) ),
                    2
                    )
                    );
                // unit: 1/s; SI
        }
    }

} // namespace atomicPhysics
} // namespace particles
} // namespace picongpu
