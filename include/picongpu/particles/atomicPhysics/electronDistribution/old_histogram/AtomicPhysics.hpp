/* Copyright 2020 Brian Marre
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
 */

#pragma once


namespace namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics
        {
            namespace electronDistribution
            {
                namespace histogram2
                {
                    template<
                        /*?type?*/
                        typename T_IonSpecies,
                        typename ConfigNumberDataType,
                        typename RandomIntGen,
                        typename RandomFloatGen>
                    struct AtomicRateSolver
                    {
                        template<
                            typename T_Acc,
                            typename T_IonSpecies,
                            typename ConfigNumberDataType,
                            typename RandomIntGen,
                            typename RandomFloatGen>
                        void operator()(
                            T_Acc acc,
                            T_IonSpecies ions,
                            RandomIntGen& randomIntGen,
                            RandomFloatGen& randomFloatGen,
                            RateMatrix& rateMatrix)
                        {
                            float_X timeRemaining;
                            float_X rate;
                            float_X probability;

                            ConfigNumberDataType newState;
                            ConfigNumberDataType randomNumber;

                            timeRemaining = static_cast<double>(picongpu::SI::DELTA_T_SI);

                            while(timeRemaining > 0)
                            {
                                newState = randomIntGen();

                                rate = rateMatrix(newState, ion[atomicConfigNumber_].configNumber, histogram);
                                probability = rate * timeRemaining;
                                if(probability >= 1)
                                {
                                    currentState.configNumber = newState;
                                    timeRemaining -= 1 / rate;
                                }
                                else
                                {
                                    if(randomFloatGen() <= probability)
                                    {
                                        currentState.configNumber = newState;
                                        timeRemaining = 0;
                                    }
                                }
                            }
                        }
                    }


                    template<typename Acc, typename T_histogram, typename T_IonSpecies>
                    struct AtomicPhysicsKernel
                    {
                    public:
                        template<typename T_Acc, typename T_histogram, typename T_IonSpecies>
                        void operator()(T_Acc const& acc)
                        {
                            {
                                // Define ion species and frame type datatype for later access
                                using IonSpecies
                                    = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

                                // get specialisation of ConfigNumber class used in this species
                                using IonSpeciesAtomicConfigNumber = pmacc::particles::traits::ResolveAliasFromSpecies<
                                    IonSpecies,
                                    atomicConfigNumber<>
                                    /* atomicConfigNumber is alias(interface name) for specific
                                    specialisation of ConfigNumber of this specific species*/
                                    >;

                                /* get T_DataType used as parameter in ConfigNumber.hpp via public
                                typedef in class */
                                using ConfigNumberDataType = typename IonSpeciesAtomicConfigNumber::DataType;

                                std::uniform_int_distribution<ConfigNumberDataType> randomIntGen;
                                std::uniform_real_distribution<float> randomFloatGen;

                                // initializing the random number generators
                                randomIntGen = std::uniform_int_distribution<ConfigNumberDataType>(
                                    0,
                                    IonSpeciesAtomicConfigNumber.numberStates());
                                randomFloatGen = std::uniform_real_distribution<float>(0, 1)
                            }


                            T_histogram histogram = new T_histogram(); // shared mem.

                            fillHistogram(histogram, electrons);

                            // process ions
                            AtomicRateSolver(histogram, ions);

                            // process electrons
                            ElectronEnergyFeedback(histogram, electrons);
                        }
                    }

                } // namespace histogram2
            } // namespace electronDistribution
        } // namespace atomicPhysics
    } // namespace particles
} // namespace picongpu
