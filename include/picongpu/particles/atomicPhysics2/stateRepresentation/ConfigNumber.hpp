/* Copyright 2019-2023 Brian Marre
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

/** @file implements conversion from/to atomic state ConfigNumbers
 *
 * A ConfigNumber is an index number of a flyCHK super-configuration of an hydrogen like atomic state
 *
 * The Hydrogen-like super configuration is specified by the occupation vector N-vector,
 * a listing of every level n's occupation number N_n, with n: 1 < n < n_max, same as FlyCHK.
 *
 * These different super configurations are organized in a combinatorial table
 * and numbered starting with the completely ionized state.
 *
 * # |N_1, |N_2, |N_3, ...
 * 0 |0    |0    |0
 * 1 |1    |0    |0
 * 2 |2    |0    |0
 * 3 |0    |1    |0
 * 4 |1    |1    |0
 * 5 |2    |1    |0
 * 6 |0    |2    |0
 * 7 |1    |2    |0
 * 8 |2    |2    |0
 * 9 |0    |3    |0
 * ...
 *
 * analytical conversion formula:
 * # ... configNumber assigned
 * # = N_1 *(g(0) + 1)+ N_2 * (g(1)+1) + N_3 * (g(2)+1) * (g(1) + 1)) + ...
 * # = Sum_{n=1}^{n_max}[ N_n * Product_{i=1}^{n} (g(i-1) + 1) ]
 *
 * g(n) ... maximum number of electrons in a given level n
 * g(n) = 2 * n^2
 * quick reference:
 * https://en.wikipedia.org/wiki/Electron_shell#Number_of_electrons_in_each_shell
 *
 * Note: a super configuration only stores occupation numbers, NOT spin or
 *  angular momentum. This was done due to memory constraints.
 *
 * further information see:
 *  https://en.wikipedia.org/wiki/Hydrogen-like_atom#Schr%C3%B6dinger_solution
 *
 * @todo 2020-07-01 BrianMarre: implement usage of current charge to account for
 * one more level than actually saved, => n_max effective = n_max + 1
 */

#pragma once

#include "picongpu/particles/atomicPhysics2/ConvertEnum.hpp"

#include <pmacc/algorithms/math.hpp>
#include <pmacc/math/Vector.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics2::stateRepresentation
{
    /** Implements the actual storage of the super configurations by index
     *
     * @tparam T_DataType ... unsigned integer data type to represent the config number
     * @tparam T_numberLevels ... max principle quantum number, n_max, represented in
     *                              this configNumber
     * @tparam T_atomicNumber ... atomic number of the ion, in units of elementary charge
     */
    template<typename T_DataType, uint8_t T_numberLevels, uint8_t T_atomicNumber>
    class ConfigNumber
    {
    public:
        static constexpr uint8_t numberLevels = T_numberLevels;
        static constexpr uint8_t atomicNumber = T_atomicNumber;
        using DataType = T_DataType;

    private:
        /** g(n) = 2 * n^2, maximum possible occupation number of the n-th shell
         *
         * returns the maximum possible occupation number for the n-th shell,
         *
         * or in other words how many different electron states there are for a given
         * principal quantum number n.
         *
         * @param n principal quantum number
         */
        HDINLINE static constexpr uint16_t g(uint8_t const n)
        {
            // cast necessary to prevent overflow in n^2 calculation
            return static_cast<uint16_t>(2u) * pmacc::math::cPow(static_cast<uint16_t>(n), static_cast<uint8_t>(2u));
        }

        /** number of different occupation number values possible for the n-th shell
         *
         * @attention n larger than 254 cause an overflow.
         *
         * @param n principal quantum number
         */
        HDINLINE static constexpr uint16_t numberOfOccupationNumberValuesInShell(uint8_t const n)
        {
            uint16_t g_n = g(n);

            if(g_n > static_cast<uint16_t>(T_atomicNumber))
                g_n = static_cast<uint16_t>(T_atomicNumber);

            // min(g(n), T_atomicNumber) + 1u
            return g_n + static_cast<uint16_t>(1u);
        }

        /** returns the step length of the n-th shell
         *
         * ie. number of table entries per occupation number VALUE of the
         * current principal quantum number n.
         *
         * @attention n larger than 254 cause an overflow.
         *
         *@param n principal quantum number
         */
        HDINLINE static constexpr T_DataType stepLength(uint8_t const n)
        {
            T_DataType result = static_cast<T_DataType>(1u);

            for(uint8_t i = 1u; i < n; ++i)
            {
                result *= static_cast<T_DataType>(ConfigNumber::numberOfOccupationNumberValuesInShell(i));
            }

            return result;
        }

        /** returns stepLength(current_n+1) based on stepLength(current_n) and current_n
         *
         * equivalent to numberOfOccupationNumberValuesInShell(current_n+1), but faster
         *
         * @attention n larger than 254 cause an overflow.
         *
         * @param current_n principal quantum number of current shell
         * @param currentStepLength current step length
         *
         * @return changes stepLength to stepLength(n+1)
         */
        HDINLINE static constexpr T_DataType nextStepLength(
            T_DataType const currentStepLength,
            uint8_t const current_n)
        {
            return currentStepLength
                * static_cast<T_DataType>(ConfigNumber::numberOfOccupationNumberValuesInShell(current_n));
        }

        /** returns stepLength(current_n-1) based on stepLength(current_n) and current_n
         *
         * equivalent to numberOfOccupationNumberValuesInShell(current_n-1), but faster
         *
         * @attention n larger than 254 cause an overflow
         * @attention does not check for ranges
         *
         * @param current_n principal quantum number of current shell
         * @param currentStepLength current step length
         *
         * @return changes stepLength to stepLength(n-1)
         */
        HDINLINE static constexpr T_DataType previousStepLength(
            T_DataType const currentStepLength,
            uint8_t const current_n)
        {
            return currentStepLength
                / static_cast<T_DataType>(ConfigNumber::numberOfOccupationNumberValuesInShell(current_n - 1u));
        }

    public:
        //! returns number of different states(Configs) that are representable
        HDINLINE static constexpr T_DataType numberStates()
        {
            return stepLength(numberLevels + static_cast<uint8_t>(1u));
        }

        // conversion methods
        //@{
        /** convert an occupation number vector to a configNumber
         *
         * Uses the formula in file description.
         * @param levelVector occupation number vector (N_1, N_2, N_3, ... , N_(n_max)
         */
        HDINLINE static constexpr DataType getAtomicConfigNumber(
            pmacc::math::Vector<uint8_t, numberLevels> const levelVector)
        {
            /* stepLength ... number of table entries per occupation number VALUE of
             * the current principal quantum number n.
             */
            DataType stepLength = static_cast<DataType>(1);

            DataType configNumber = static_cast<DataType>(0);

            for(uint8_t n = 0u; n < numberLevels; ++n)
            {
                /* BEWARE: n here equals n-1 in formula in file documentation,
                 *
                 * since for loop starts with n=0 instead of n=1,
                 */
                stepLength = nextStepLength(stepLength, n);
                configNumber += levelVector[n] * stepLength;
            }

            return configNumber;
        }

        /** converts configNumber into an occupation number vector
         *
         * Index of result vector corresponds to principal quantum number n-1.
         *
         * Exploits that for largest whole number k for a given configNumber N,
         * such that k * stepLength <= N, k is equal to the occupation number of
         * n_max.
         *
         * This is used recursively to determine all occupation numbers.
         * further information: see master thesis of Brian Marre
         *
         * @param N atomic state configNumber, uint like
         * @return (N_1, N_2, N_3, ..., N_(n_max))
         */
        HDINLINE static pmacc::math::Vector<uint8_t, numberLevels> getLevelVector(DataType configNumber)
        {
            pmacc::math::Vector<uint8_t, numberLevels> result = pmacc::math::Vector<uint8_t, numberLevels>::create(0);

            T_DataType stepLength = ConfigNumber::stepLength(numberLevels);

            // BEWARE: for-loop counts down, starting with n_max
            for(uint8_t n = numberLevels; n >= 1; --n)
            {
                // get occupation number N_n by getting largest whole number factor
                result[n - 1] = static_cast<uint8_t>(configNumber / stepLength);

                // remove contribution of current N_n
                configNumber -= stepLength * (result[n - 1]);

                stepLength = ConfigNumber::previousStepLength(stepLength, n);
            }

            return result;
        }

        /** get number bound electrons from configNumber
         *
         * @param configNumber configNumber, uint like, not an object
         *
         * @returns number of bound electrons
         */
        HDINLINE static constexpr uint8_t getBoundElectrons(DataType configNumber)
        {
            uint8_t numberElectrons = 0u;

            T_DataType stepLength = ConfigNumber::stepLength(T_numberLevels);

            // BEWARE: for-loop counts down, starting with n_max
            for(uint8_t n = T_numberLevels; n >= 1; --n)
            {
                // get occupation number N_n by getting largest whole number factor
                uint8_t shellNumberElectrons = static_cast<uint8_t>(configNumber / stepLength);

                // remove contribution of current N_n
                configNumber -= stepLength * shellNumberElectrons;

                stepLength = ConfigNumber::previousStepLength(stepLength, n);

                numberElectrons += shellNumberElectrons;
            }

            return numberElectrons;
        }

        /** get charge state from configNumber
         *
         * @param configNumber configNumber, uint like, not an object
         *
         * @returns charge of ion
         */
        HDINLINE static constexpr uint8_t getChargeState(DataType configNumber)
        {
            return atomicNumber - getBoundElectrons(configNumber);
        }

        /** get the direct pressure ionization state's atomicConfigNumber
         *
         * @returns for the atomicConfigNumber corresponding to the level vector n = (n_1, ..., n_k, 0, ...),
         *      n_i being the occupation number of the i-th shell and k being the highest occupied shell
         * the atomicConfigNumber corresponding to n_PI = (n_1, ..., n_k - 1, 0, ...)
         */
        HDINLINE static DataType getDirectPressureIonizationState(DataType const configNumber)
        {
            pmacc::math::Vector<uint8_t, numberLevels> levelVector = getLevelVector(configNumber);

            // find highest occupied shell
            uint8_t highestOccupiedShell = numberLevels - 1u;
            for(uint8_t k = numberLevels; k >= 1u; --k)
            {
                highestOccupiedShell = k - 1u;
                if(levelVector[k - 1u] > 0u)
                    break;
            }

            // find direct pressure ionization state level vector
            if(levelVector[highestOccupiedShell] != 0u)
                // not completely ionized state
                levelVector[highestOccupiedShell] -= 1u;
            return getAtomicConfigNumber(levelVector);
        }
    };
} // namespace picongpu::particles::atomicPhysics2::stateRepresentation
