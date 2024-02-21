/* Copyright 2013-2023 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Richard Pausch, Alexander Debus, Marco Garten,
 *                     Benjamin Worpitz, Alexander Grund, Sergei Bastrakov
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

#include <pmacc/meta/ForEach.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>

#include <cstdint>

namespace picongpu::simulation::stage
{
    /** Functor for the stage of the PIC loop performing particle ionization
     *
     * Only affects particle species with the Synchrotron attribute.
     * 
     * @tparam T_numberTableEntries number of synchrotron function values to precompute, stored in table
     * @tparam T_numberSamplePoints number of samples to use in integration in firstSynchrotronFunction
     * @tparam T_CutoffParam::logEnd cutoff for 2nd kind cyclic bessel function function close enough to zero
     */
    template<typename T_CutoffParam = picongpu::particles::synchrotron::params::FirstSynchrotronFunctionParam, 
            typename T_IndexingParam = picongpu::particles::synchrotron::params::InterpolationIndexingParam,
            uint64_t T_numberTableEntries = picongpu::particles::synchrotron::params::numberTableEntries,
            uint32_t T_numberSamplePoints = picongpu::particles::synchrotron::params::numberSamplePoints>
    class SynchrotronRadiation
    {
        
        /// debug only, @todo remove
        static constexpr bool T_Debug = true; 

    private:
        /** exponential integration
         * 
         * @brief approximate function with exponential and integrate on log scale, by using the middle point to fit the exponent
         * 
         * @note overestimates function below xMiddle but underestimates fucntion above xMiddle leading to better fit overal
         *
         * @param xLeft left point
         * @param xMiddle middle point
         * @param xRight right point
         * @param yLeft function value at xLeft
         * @param yMiddle function value at xMiddle
         */
        template<typename T_Number>
        static T_Number integrateAsExponential(T_Number const xLeft, T_Number const xMiddle, T_Number const xRight, T_Number const yLeft, T_Number const yMiddle) {
            if constexpr(T_Debug)
            {
                if (xLeft == xMiddle) {
                    throw "xLeft and xMiddle cannot be the same.";
                }
                if (yLeft <= 0 || yMiddle <= 0) {
                    throw "yLeft and yMiddle must be positive.";
                }
                if (yLeft == yMiddle) {
                    throw "yLeft and yMiddle must not be equal.";
                }
            }

            // fitting function: y = a * e^(b * x)
            T_Number const b = (math::log(yMiddle) - math::log(yLeft)) / (xMiddle - xLeft);
            T_Number const a = yLeft / math::exp(b * xLeft);

            T_Number const integral = (a / b) * (math::exp(b * xRight) - math::exp(b * xLeft));
            return integral;
        }

        /// @todo insure intervals consistent
        /** compute first synchrotorn function
         *
         * @param zq
         * 
         * @returns zq * (integral of 2nd kind bessel function from zq to ~infinity)
         */
        static float_64 firstSynchrotronFunction(float_64 const zq)
        {
            float_64 const log_start = std::log10(zq); // from zq to T_CutoffParam::logEnd
            // std::cout << "log_start = " << log_start << std::endl;
            float_64 const log_step = (T_CutoffParam::logEnd - log_start) / (T_numberSamplePoints - 1);
            // std::cout << "log_step = " << log_step << std::endl;

            float_64 integral = 0;

            float_64 xLeft;
            float_64 xRight = zq;

            for (int i = 0; i < T_numberSamplePoints - 1; ++i) {
                xLeft = xRight;
                xRight = math::pow(10., log_start + log_step * (i + 1));
                float_64 xMiddle = (xLeft + xRight) / 2.0;

                // try and catch errors in the bessel function
                try {
                    float_64 yLeft = std::cyl_bessel_k(5.0 / 3.0, xLeft);
                    float_64 yMiddle = std::cyl_bessel_k(5.0 / 3.0, xMiddle);
                    integral += integrateAsExponential(xLeft, xMiddle, xRight, yLeft, yMiddle);
                } catch (std::exception& e) {
                    std::cout << "Caught exception: " << e.what() << std::endl;
                }
            }
            return zq * integral;
        }

        //! @note see paper "Synchrotron Radiation and its[sic] Applications" by A. A. Sokolov and I. M. Ternov (credit goes to gpt)
        static float_64 secondSynchrotronFunction(float_64 const x)
        {
            return  x * std::cyl_bessel_k(2./3, x);
        }

    public:
        /** Create a particle ionization functor
         *
         * Having this in constructor is a temporary solution.
         *
         * @param cellDescription mapping for kernels
         */
        SynchrotronRadiation(MappingDesc const cellDescription) : cellDescription(cellDescription)
        {
            using namespace picongpu::particles::synchrotron; // for "params" namespace
            auto data_space = DataSpace<2>{T_numberTableEntries, 2};
            auto grid_layout = GridLayout<2>{data_space};

            tableValuesF1F2 = std::make_shared<GridBuffer<float_64, 2>>(grid_layout);
            
            constexpr float_64 minZqExp = T_IndexingParam::minZqExponent;
            constexpr float_64 maxZqExp = T_IndexingParam::maxZqExponent;

            // first and last value set to 0
            tableValuesF1F2->getHostBuffer().getDataBox()(DataSpace<2>{0, params::u32(params::Accessor::f1)}) = 0;
            tableValuesF1F2->getHostBuffer().getDataBox()(DataSpace<2>{0, params::u32(params::Accessor::f2)}) = 0;
            tableValuesF1F2->getHostBuffer().getDataBox()(DataSpace<2>{T_numberTableEntries - 1, params::u32(params::Accessor::f1)}) = 0;
            tableValuesF1F2->getHostBuffer().getDataBox()(DataSpace<2>{T_numberTableEntries - 1, params::u32(params::Accessor::f2)}) = 0;

            // precompute remaining F1 and F2 on log scale
            for (uint32_t iZq = 1; iZq < T_numberTableEntries - 1; iZq++) 
            {
                std::cout << "iZq = " << iZq << std::endl;
                float_64 zq = std::pow(10, minZqExp + (maxZqExp - minZqExp) * iZq / static_cast<float_64>(T_numberTableEntries));
                std::cout << "zq = " << zq << std::endl;
                // inverse function for index retrieval:
                // index = (log10(zq) - minZqExp) / (maxZqExp - minZqExp) * T_numberTableEntries;
                
                float_64 const F1 = firstSynchrotronFunction(zq); 
                std::cout << "F1 = " << F1;
                float_64 const F2 = secondSynchrotronFunction(zq);
                std::cout << " F2 = " << F2 << std::endl;

                tableValuesF1F2->getHostBuffer().getDataBox()(DataSpace<2>{iZq, params::u32(params::Accessor::f1)}) = F1;
                tableValuesF1F2->getHostBuffer().getDataBox()(DataSpace<2>{iZq, params::u32(params::Accessor::f2)}) = F2;
            }

            tableValuesF1F2->hostToDevice();

            //! debug only, print F1 and F2, @todo remove
            int timesMore = 1;
            for (uint32_t zq = 0; zq < T_numberTableEntries*timesMore; zq++) {
                float_64 zq_ = std::pow(10,minZqExp + (maxZqExp - minZqExp) * zq / float_64(T_numberTableEntries*timesMore));
                uint32_t index = (log10(zq_) - minZqExp) * (T_numberTableEntries) / (maxZqExp - minZqExp) ;
                

                std::cout << "zq = " << zq/float(timesMore)  << " index = " << index << std::endl;
                std::cout << "zq = " << zq_ << " F1 = " << tableValuesF1F2->getHostBuffer().getDataBox()(DataSpace<2>{index,0}) << " F2 = " << tableValuesF1F2->getHostBuffer().getDataBox()(DataSpace<2>{index,1}) << std::endl;

                // interpolation:
                // F1 = tableValuesF1F2->getHostBuffer().getDataBox()(DataSpace<2>{index,0});
            }
        }

        /** Ionize particles
         *
         * @param step index of time iteration
         */
        void operator()(uint32_t const step) const
        {
            using pmacc::particles::traits::FilterByFlag;
            using SpeciesWithSynchrotron = typename FilterByFlag<VectorAllSpecies, Synchrotron<>>::type;
            pmacc::meta::ForEach<SpeciesWithSynchrotron, particles::CallSynchrotron<boost::mpl::_1>>
                synchrotronRadiation;
            synchrotronRadiation(cellDescription, step, tableValuesF1F2->getDeviceBuffer().getDataBox());
        }

    private:
        //! Mapping for kernels
        MappingDesc cellDescription;
        // precomputed first and second synchrotron functions -> 2d grid of floats_64 -> tableValuesF1F2[zq][0/1] -> 0/1 = F1/F2
        std::shared_ptr<GridBuffer<float_64,2>> tableValuesF1F2;
    };
} // namespace picongpu::simulation::stage
