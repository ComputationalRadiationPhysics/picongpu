/* Copyright 2017-2020 Axel Huebl, Brian Marre
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


#pragma once

namespace picongpu
{
namespace particles
{
namespace atomicPhysics
{
    template<
        typename T_TypeIndex
        typename T_AtomicDataBox
    >
    class AtomicRate
    {
     public:
        using constexpr Idx = T_TypeIndex;
        using constexpr AtomicDataBox = T_AtomicDataBox;

    private:
        /** gaunt factor like suppression of crosssection
         *
         * @param energyDifference ... difference of energy between atomic states
         *      unit: 2*Ry ... double the Rydberg energy == picongpu::ATOMIC_UNIT_ENERGY
         * @param energy Electron ... energy of electron
         *      unit: 2*Ry ... see energyDifference
         * @param indexTransition ... internal index of transition in atomicDataBox
         *      use findIndexTransition method of atomicDataBox and screen for not found value
         *      BEWARE: method assumes that indexTransition is valid, undefined behaviour otherwise
         *
         * return uint: unitless
         */
        float_X gauntFactor(
            float_X energyDifference,
            float_X energyElectron,
            uint32_t indexTransition,
            AtomicDataBox atomicDataBox
            )
        {
            // get gaunt coeficients
            float_X const A = atomicDataBox.getCxin1( indexTransition );
            float_x const B = atomicDataBox.getCxin2( indexTransition );
            float_X const C = atomicDataBox.getCxin3( indexTransition );
            float_X const D = atomicDataBox.getCxin4( indexTransition );
            float_X const a = atomicDataBox.getCxin5( indexTransition );

            // calculate gaunt Factor
            float_X const U = energyElectron / energyDifference;
            float_X const g = A * math::log(U) + B + C / ( U + a ) + D / pmacc::algorithms::math::pow( U + a, 2 );

            // make sure 
            return g * (U > 1.0);
        }

        // @param energyElectron ... uint: picongpu::ATOMIC_UNIT_ENERGY
        // returns unit: m^2
        float_X collisionalExcitationCrosssection(
            Idx const lowerIdx,
            Idx const upperIdx,
            float_X const energyElectron,
            AtomicDataBox atomicDataBox
            )
        {
            // energy difference between atomic states
            float_X const energyDifference = atomicDataBox( upperIdx ) - atomicDataBox( lowerIdx );

            //collisional absorption obscillator strength of transition [unitless]
            uint32_t const indexTransition = atomicDataBox.findTransition( idxLower, idxUpper );

            // check whether Transition exists
            if ( indexTransition == atomicDataBox.getNumTransitions() )
                return 0._X; // 0 crossection for non existing transition,
            // BEWARE: input data may be incomplete
            // TODO: implement fallback

            float_X const collisionalOscillatorStrength = atomicDataBox.getCollisionalOscillatorStrength( indexTransition );

            // physical constants, ask Axel Huebl if you want to know more
            float_X c0_SI = float_X( 8._X *
                pmacc::algorithms::math::pow( picongpu::pi * picongpu::BOHR_RADIUS, 2 ) /
                pmacc::algorithms::math::sqrt( 3._X) ); // uint: m^2
            float_X c0 = c0_SI /

            // uint: m^2
            return c0_SI *
                pmacc::algorithms::math::pow(
                    ( picongpu::ATOMIC_UNIT_ENERGY / 2._X )
                    / energyDifference, 2
                    )
                * collisionalOscillatorStrength
                * ( energyDifference / energyElectron )
                * gauntFactor( lowerIdx, upperIdx, energyElectron );
        }

        float_x collisionalDeExcitationCrosssection()
        {
            
        }

        DINLINE float_X operator()(
            Idx oldState,
            Idx newState,
            float_X energy,
            AtomicData atomicDataBox
            )
        {
            
        }
    }

/*
    def dx_times_Ene(self, iz, i, j):
        """
        see `ex_times_Ene`
        """
        # detailed balance: "gaa = glev" ratio
        statistical_ratio = self._get_state_ratio_weight(iz, i, j)
        
        return statistical_ratio * self.ex_times_Ene(iz, i, j)

    def dx(self, iz, i, j, Ene):
        """
        Collisional De-Excitation Cross-Section

        see `ex`
        """
        # ATTENTION: should gaunt be: 0.15 + 0.28 * log( e/eji + 1.0 ) ? TODO
        #                                                      ^^^^^
        return self.dx_times_Ene(iz, i, j) * self.gaunt(iz, i, j, Ene) / Ene

    def _get_state_dEne(self, iz, i, j):
        """
        """
        lvl_iz = self.atomic_levels[np.where( self.atomic_levels['charge_state'] == iz )]
        lvl_i = lvl_iz[np.where( lvl_iz['state_idx'] == i )]
        lvl_j = lvl_iz[np.where( lvl_iz['state_idx'] == j )]
        
        # TODO: each MINUS 'ionization_potential' of level? (scdrv.f)
        return lvl_j['energy'] - lvl_i['energy']

    def _get_state_ratio_weight(self, iz, i, j):
        """
        """
        lvl_iz = self.atomic_levels[np.where( self.atomic_levels['charge_state'] == iz )]
        gaa_i = lvl_iz[np.where( lvl_iz['state_idx'] == i )]
        gaa_j = lvl_iz[np.where( lvl_iz['state_idx'] == j )]
        
        return gaa_i['statistical_weight'] / gaa_j['statistical_weight']

    def _get_state_f(self, iz, i, j):
        """
        Get absorption oscillator strength f [unitless]
        """
        #print("get faax for {} {}".format(i, j))
        trans_iz = self.b_transitions[np.where( self.b_transitions['charge_state_lower'] == iz )]
        trans_ij = trans_iz[np.where(np.logical_and(
            trans_iz['state_idx_lower'] == i,
            trans_iz['state_idx_upper'] == j
        ))]
        
        return trans_ij['faax']
*/
} // namespace atomicPhysics
} // namespace particles
} // namespace picongpu
