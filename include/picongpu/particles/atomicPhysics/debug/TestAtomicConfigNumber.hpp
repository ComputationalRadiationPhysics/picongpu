/* Copyright 2022-2024 Brian Marre
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

#pragma once

#include "picongpu/particles/atomicPhysics/ConvertEnum.hpp"
#include "picongpu/particles/atomicPhysics/stateRepresentation/ConfigNumber.hpp"

#include <pmacc/math/Vector.hpp>

#include <array>
#include <cstdint>
#include <iostream>
#include <list>
#include <string>
#include <utility>

namespace picongpu::particles::atomicPhysics::debug
{
    /** Tests of configNumber conversion methods
     *
     * @tparam T_ConsoleOutput true =^= write results and correct value to console
     * @attention must be called on cpu only, if T_ConsoleOutput==true
     *
     * @return true =^= all tests passed
     */
    template<bool T_ConsoleOutput = true>
    struct TestAtomicConfigNumber
    {
        using DataType = uint64_t;
        static constexpr uint8_t numberLevels = 10u;
        static constexpr uint8_t atomicNumber = 18u;

        using Config = stateRepresentation::ConfigNumber<DataType, numberLevels, atomicNumber>;
        using LevelVector = std::array<uint8_t, numberLevels>;
        using pmaccVector = pmacc::math::Vector<uint8_t, numberLevels>;

        using TestExample = std::pair<DataType, LevelVector>;

        template<uint8_t T_numberLevels>
        void convertToLevelVector(LevelVector& levelVector, pmaccVector vector)
        {
            uint8_t temp;

            for(uint8_t i = 0u; i < T_numberLevels; i++)
            {
                temp = u8(vector[i]);
                levelVector[i] = temp;
            }
        }

        static std::string toString(LevelVector levelVector)
        {
            std::string string = "(" + std::to_string(levelVector[0]);

            for(uint8_t i = 1u; i < levelVector.size(); ++i)
            {
                string += ", " + std::to_string(levelVector[i]);
            }
            return string + ")";
        }

        bool testAll()
        {
            //! testCases, see piconpguAtomicPhysicsTools repo, ConfigNumberConversionReference for new examples
            std::list<TestExample> testExamples
                = {// configNumber, levelVector
                   // standard examples
                   TestExample{9779u, LevelVector{2, 1, 1, 0, 1, 0, 0, 0, 0, 0}},
                   TestExample{66854705u, LevelVector{2, 1, 1, 0, 0, 0, 0, 1, 0, 0}},
                   // high value test examples
                   TestExample{24134536956u, LevelVector{0, 1, 0, 0, 0, 0, 0, 0, 0, 1}},
                   TestExample{24134537168u, LevelVector{2, 8, 7, 0, 0, 0, 0, 0, 0, 1}},
                   TestExample{24134537139u, LevelVector{0, 8, 6, 0, 0, 0, 0, 0, 0, 1}},
                   TestExample{434421665154u, LevelVector{0, 0, 0, 0, 0, 0, 0, 0, 0, 18}},
                   // all levels test example
                   TestExample{25475344564u, LevelVector{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}};

            bool pass = true;

            DataType knownConfigNumber;
            LevelVector knownLevelVector;
            pmaccVector levelVectorTemp;
            LevelVector levelVector;

            for(TestExample const& testCase : testExamples)
            {
                // good Results
                knownConfigNumber = std::get<0>(testCase);
                knownLevelVector = std::get<1>(testCase);

                if constexpr(T_ConsoleOutput)
                {
                    std::cout << "Test: (" << knownConfigNumber << ", " << toString(knownLevelVector) << " )"
                              << std::endl;
                }

                // getLevelVector()
                levelVectorTemp = Config::getLevelVector(knownConfigNumber);
                convertToLevelVector<numberLevels>(levelVector, levelVectorTemp);
                pass = (pass && (levelVector == knownLevelVector));
                if constexpr(T_ConsoleOutput)
                {
                    std::cout << "\t getLevelVector(): " << toString(levelVector) << std::endl;
                }

                // getChargeState()
                uint8_t numberElectrons = 0u;
                for(uint8_t i = 0u; i < numberLevels; i++)
                {
                    numberElectrons += levelVector[i];
                }
                pass = (pass && ((atomicNumber - numberElectrons) == Config::getChargeState(knownConfigNumber)));

                if constexpr(T_ConsoleOutput)
                {
                    std::cout << "\t getChargeState(): " << std::to_string(atomicNumber - numberElectrons);
                    std::cout << " =?= (returnValue:) ";
                    std::cout << std::to_string(static_cast<uint16_t>(Config::getChargeState(knownConfigNumber)));
                    std::cout << std::endl;
                }

                // getAtomicConfigNumber()
                for(uint8_t i = 0u; i < numberLevels; i++)
                {
                    levelVectorTemp[i] = u8(knownLevelVector[i]); // reuse
                }
                pass = (pass && (knownConfigNumber == Config::getAtomicConfigNumber(levelVectorTemp)));

                if constexpr(T_ConsoleOutput)
                {
                    std::cout << "\t getAtomicConfigNumber(): " << knownConfigNumber;
                    std::cout << " =?= " << Config::getAtomicConfigNumber(levelVectorTemp) << std::endl;
                }
            }

            if constexpr(T_ConsoleOutput)
            {
                if(pass)
                    std::cout << "Success" << std::endl;
                else
                    std::cout << "Fail" << std::endl;
            }

            return pass;
        }
    };
} // namespace picongpu::particles::atomicPhysics::debug
