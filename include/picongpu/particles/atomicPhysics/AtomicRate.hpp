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

#inlcude <pmacc/algorithms/math.hpp

#pragma once

/** rate calculation from given atomic data, extracted from flylite based on FLYchk
 *
 * References:
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

namespace picongpu
{
namespace particles
{
namespace atomicPhysics
{
    /*
    def __init__(self, atomic_levels, b_transitions):
        """
        Parameters
        ----------
        atomic_levels: flylite.utils.data.AtomicLevels
            atomic structure of element
        b_transitions: flylite.utils.data.AtomicTransitions
            bound-bound transition data
        """
        self.atomic_levels = atomic_levels
        self.b_transitions = b_transitions
    */

    template<
        typename T_TypeIndex
        typename T_AtomicDataBox
    >
    class AtomicRate
    {
        using constexpr Idx = T_TypeIndex;
        using constexpr AtomicDataBox = T_AtomicDataBox;

        float_X gauntFactor(
            Idx const idxLower,
            Idx const idxUpper,
            float_X energyElectron, // energy in eV
            AtomicDataBox atomicDataBox
            )
        {
            // energy difference of atomic states
            float_X energyDifference = atomicDataBox( idxUpper ) - atomicDataBox( idxLower );

            // get gaunt coeficients
            uint32_t const i = atomicDataBox.findTransition( idxLower, idxUpper );
            float_X const A = atomicDataBox.getCxin1( i );
            float_x const B = atomicDataBox.getCxin2( i );
            float_X const C = atomicDataBox.getCxin3( i );
            float_X const D = atomicDataBox.getCxin4( i );
            float_X const a = atomicDataBox.getCxin5( i );

            // calculate Gaunt Factor
            float_X const U = energyElectron / energyDifference;
            float_X const  g = A * math::log(U) + B + C / ( U + a ) + D / ( U + a )**2;

            // make sure 
            return g * (U > 1.0);
        }

        DINLINE float_X operator()(
            Idx oldState,
            Idx newState,
            float_X energy,
            AtomicData atomicDataBox
            )
        {
            
        }

        float_X collosionalExcitationCrosssection( Idx const lowerIdx, Idx const upper Idx)
        {
            return 0._X;
        }
    }

/*
*/

    def ex_times_Ene(self, iz, i, j):
        """
        same as `ex` but times energy and without gaunt factor

        this function does not depend on the energy of the incoming particle

        Returns
        -------
        cross-section [cm^2] * incoming e- energy [eV] / Gaunt factor
        """
        # atomic data
        #   energy difference (transition energy) between level
        #   j (upper) and i (lower) level
        dEne = self._get_state_dEne(iz, i, j)
        #   absorption obscillator strength from i to j [unitless]
        f = self._get_state_f(iz, i, j)
        
        # constants
        a0 = 5.292e-9 # Bohr radius [cm]
        c0 = 8.0 * ( np.pi * a0 )**2 / 3.**0.5 # [cm^2]
        EneH = 13.605693 # Rydberg unit of energy: Hydrogen ioniz. energy [eV]

        return c0 * ( EneH / dEne )**2 * f * dEne

    def ex(self, iz, i, j, Ene):
        """
        Collisional Excitation Cross-Section

        Parameters
        ----------
        iz: (unsigned) int
            charge state [unitless]
        i: (unsigned) int
            lower level [unitless]
        j: (unsigned) int
            upper level [unitless]
        E: float
            energy of incoming electron [eV]

        Returns
        -------
        cross-section [cm^2]
        """
        return self.ex_times_Ene(iz, i, j) * self.gaunt(iz, i, j, Ene) / Ene

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

} // namespace atomicPhysics
} // namespace particles
} // namespace picongpu
