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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

/** @file
 *
 * This file implements the actual storage of atomic configurations.
 *
 * This implementation uses config numbers for this. A configNumber is an index number
 * of an hydrogen like atomic state, with several conversion methods and static methods.
 * Instances of this class represent an atomic state of an ion by numbering
 * Hydrogen-like super configurations.
 *
 * The Hydrogen-like super configuration is specified by the occupation
 * vector N-vector, a listing of every level n's occupation number N_n, with n:
 * 1 < n < n_max.
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
 * analytical formula:
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

#include <pmacc/assert.hpp>
#include <pmacc/math/Vector.hpp>

#include <cstdint>

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics
        {
            namespace stateRepresentation
            {
                /** Implements the actual storage of the super configurations by index
                 *
                 * @tparam T_DataType unsigned integer data type to represent the config number
                 * @tparam T_numberLevels max principle quantum number, n_max, represented in
                 *                              this configNumber
                 * @tparam T_atomicNumber ... atomic number of the ion, in units of elementary charge
                 *
                 * public methods:
                 *   - T_DataType numberStates()
                 *       return number of states representable in this configNumber Object
                 *   - ConfigNumber( uint x )
                 *       constructor directly setting private member configNumber,
                 *       BEWARE: no conversions and no runtime range checks, only use if you know
                 *           what you are doing
                 *   - ConfigNumber( pmacc::math::Vector<uint_8, T_numberLevels > N )
                 *       constructor using occupation number vector N = ( N_1, N_2, ..., N_(n_max) )
                 *   - T_DataType getAtomicStateConfigNumber( pmacc::math::Vector<uint8_t, T_numberLevels> N )
                 *       converts occupation number vector N to configNumber index
                 *   - pmacc::math::Vector<uint8_t, T_numberLevels> getLevelVector( T_DataType x )
                 *       converts configNumber ID x to occupation number vector
                 */
                template<typename T_DataType, uint8_t T_numberLevels, uint8_t T_atomicNumber>
                class ConfigNumber
                {
                private:
                    //! actual storage configNumber ID
                    T_DataType configNumber;

                    /** g(n) = n^2, maximum possible occupation number of the n-th shell
                     *
                     * returns the maximum possible occupation number for the n-th shell,
                     *
                     * or in other words how many different electron states there are for a given
                     * principal quantum number n.
                     *
                     * @param n principal quantum number
                     */
                    HDINLINE static constexpr uint16_t g(uint8_t n)
                    {
                        // cast necessary to prevent overflow in n^2 calculation
                        return (static_cast<uint16_t>(n) * static_cast<uint16_t>(n) * 2);
                    }

                    /** number of different occupation number values possible for the n-th shell
                     *
                     * BEWARE: n larger than 254 cause an overflow.
                     *
                     * @param n principal quantum number
                     */
                    HDINLINE static uint16_t numberOfOccupationNumberValuesInShell(uint8_t n)
                    {
                        PMACC_DEVICE_ASSERT_MSG(n < 255, "n too large, must be < 255");
                        return math::min(g(n), static_cast<uint16_t>(T_atomicNumber)) + 1;
                    }

                    /** returns the step length of the n-th shell
                     *
                     * ie. number of table entries per occupation number VALUE of the
                     * current principal quantum number n.
                     *
                     *@param n principal quantum number
                     */
                    HDINLINE static T_DataType stepLength(uint8_t n)
                    {
                        T_DataType result = 1;

                        for(uint8_t i = 1u; i < n; i++)
                        {
                            result *= static_cast<T_DataType>(ConfigNumber::numberOfOccupationNumberValuesInShell(i));
                        }
                        return result;
                    }

                    /** returns the next step length of the next shell, given the
                     * current step length and current_n
                     *
                     * ie.  number of table entries per occupation number VALUE of the
                     * principal quantum number (n+1).
                     *
                     * @param n principal quantum number of current shell
                     * @param currentStepLength current step length
                     */
                    HDINLINE static void nextStepLength(T_DataType* currentStepLength, uint8_t current_n)
                    {
                        *currentStepLength = *currentStepLength
                            * static_cast<T_DataType>(ConfigNumber::numberOfOccupationNumberValuesInShell(current_n));
                    }

                public:
                    // make template parameter available for later use
                    //{
                    //! access to T_DataType, for later reference
                    using DataType = T_DataType;

                    // atomic number, available for later use
                    static constexpr uint8_t Z = T_atomicNumber;

                    //! number of levels, n_max, used for configNumber, same as T_numberLevels
                    static constexpr uint8_t numberLevels = T_numberLevels;
                    //}

                    /** returns number of different states(Configs) that are represented
                     *
                     * for use in Markov chain solver as range of integer random number
                     */
                    HDINLINE static constexpr T_DataType numberStates()
                    {
                        return static_cast<T_DataType>(stepLength(numberLevels + 1));
                    }

                    //! constructor using scalar configNumber, simple assign constructor with range checks
                    HDINLINE ConfigNumber(T_DataType N = static_cast<T_DataType>(0u))
                    {
                        PMACC_ASSERT_MSG(N >= 0, "negative configurationNumbers are not defined");
                        PMACC_ASSERT_MSG(
                            N < this->numberStates() - 1,
                            "configurationNumber N larger than largest possible ConfigNumber"
                            " for T_NumberLevels");

                        this->configNumber = N;
                    }

                    //! constructor using occupation number vector
                    HDINLINE ConfigNumber(pmacc::math::Vector<uint8_t, numberLevels> levelVector)
                    {
                        this->configNumber = getAtomicStateConfigNumber(levelVector);
                    }


                    /** convert occupation number vector to configNumber
                     *
                     * Uses the formula in file description.
                     * Assumes index of vector is principal quantum number shifted by -1.
                     *
                     * @return Returns number, not object, use constructor for getting an
                     * instance of the class.
                     */
                    HDINLINE static T_DataType getAtomicConfigNumber(
                        pmacc::math::Vector<uint8_t, numberLevels> levelVector)
                    {
                        /* stepLength ... number of table entries per occupation number VALUE of
                         * the current principal quantum number n.
                         */
                        T_DataType stepLength = 1;

                        T_DataType configNumber = 0;

                        for(uint8_t n = 0u; n < numberLevels; n++)
                        {
                            /* BEWARE: n here equals n-1 in formula in file documentation,
                             *
                             * since for loop starts with n=0 instead of n=1,
                             */

                            // must not test for < 0 since levelVector is vector of unsigned int
                            PMACC_ASSERT_MSG(
                                g(n + 1) >= levelVector[n],
                                "occupation numbers too large, must be <=2*n^2");

                            nextStepLength(&stepLength, n);
                            configNumber += levelVector[n] * stepLength;
                        }

                        return configNumber;
                    }

                    /** converts configNumber N into an occupation number vector
                     *
                     * Index of result vector corresponds to principal quantum number n -1.
                     *
                     * Exploits that for largest whole number k for a given configNumber N,
                     * such that k * stepLength <= N, k is equal to the occupation number of
                     * n_max.
                     *
                     * This is used recursively to determine all occupation numbers.
                     * further information: see master thesis of Brian Marre
                     *
                     * @param N configNumber, uint like
                     */
                    HDINLINE static pmacc::math::Vector<uint8_t, numberLevels> getLevelVector(T_DataType N)
                    {
                        pmacc::math::Vector<uint8_t, numberLevels> result
                            = pmacc::math::Vector<uint8_t, numberLevels>::create(0);

                        T_DataType stepLength;

                        // BEWARE: for loop counts down, starting with n_max
                        for(uint8_t n = numberLevels; n >= 1; n--)
                        {
                            // calculate current stepLength
                            stepLength = ConfigNumber::stepLength(n);

                            // get occupation number N_n by getting largest whole number factor
                            result[n - 1] = static_cast<uint8_t>(N / stepLength);

                            // remove contribution of current N_n
                            N -= stepLength * (result[n - 1]);
                        }

                        return result;
                    }

                    HDINLINE void operator=(T_DataType newConfigNumber)
                    {
                        this->configNumber = newConfigNumber;
                    }

                    HDINLINE T_DataType getConfigNumber()
                    {
                        return this->configNumber;
                    }
                };

            } // namespace stateRepresentation
        } // namespace atomicPhysics
    } // namespace particles
} // namespace picongpu

/** this specifies how an object of the ConfigNumber class can be written to an
 * external file for storage.
 */

namespace pmacc
{
    namespace traits
    {
        // defines what datatype is to be used to save the data in this object
        template<typename T_DataType, uint8_t T_numberLevels, uint8_t T_atomicNumber>
        struct GetComponentsType<
            picongpu::particles::atomicPhysics::stateRepresentation::
                ConfigNumber<T_DataType, T_numberLevels, T_atomicNumber>,
            false>
        {
            using type = T_DataType;
        };

        // defines how many independent components are saved in the object
        template<typename T_DataType, uint8_t T_numberLevels, uint8_t T_atomicNumber>
        struct GetNComponents<
            picongpu::particles::atomicPhysics::stateRepresentation::
                ConfigNumber<T_DataType, T_numberLevels, T_atomicNumber>,
            false>
        {
            static constexpr uint32_t value = 1u;
        };

    } // namespace traits
} // namespace pmacc
