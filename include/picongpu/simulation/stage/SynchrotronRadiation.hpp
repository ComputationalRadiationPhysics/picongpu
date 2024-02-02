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


namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            /** Functor for the stage of the PIC loop performing particle ionization
             *
             * Only affects particle species with the Synchrotron attribute.
             */
            class SynchrotronRadiation
            {
                //classified functions
            private:
                //Helper functions
                // quad integration
                float_64 IntegrateQuadrature(std::function<float_64(float_64)> f, float_64 a, float_64 b, float_64 N) {
                    float_64 h = (b - a) / (N-1);
                    float_64 sum = 0;
                    for (int i=1; i < N; i++) {
                        sum += h * ( f(a+(i-1)*h)+4*f(a+(i-1)*h+h/2)+f(a+(i)*h)  )/6;
                    }
                    return sum;
                }

                // Better exponential integration -> approximation function with exponential and also integrating on log scale so we can use the middle point to fit the exponent wchich overestimates the first part but underestimates the second part
                float_64 compute_params_and_integral_middle(float_64 x1, float_64 x2, float_64 x3, float_64 y1, float_64 y2) {
                    if (x1 == x2) {
                        throw "x1 and x2 cannot be the same.";
                    }
                    if (y1 <= 0 || y2 <= 0) {
                        throw "y1 and y2 must be positive.";
                    }

                    float_64 b = (std::log(y2) - std::log(y1)) / (x2 - x1);
                    float_64 a = y1 / std::exp(b * x1);

                    if (b == 0) {
                        throw "Computed 'b' cannot be zero for integral calculation.";
                    }

                    float_64 area = (a / b) * (std::exp(b * x3) - std::exp(b * x1));
                    return area;
                }

                
                float_64 F1_approximation_middle(float_64 zq) {
                    const int intervals = 1000;
                    std::vector<float_64> zq_intervals(intervals);
                    float_64 log_start = std::log10(zq); // from zq
                    float_64 log_end = std::log10(100.0);// to 100
                    float_64 log_step = (log_end - log_start) / (intervals - 1);

                    // Generate logarithmically spaced intervals
                    for (int i = 0; i < intervals; ++i) {
                        zq_intervals[i] = std::pow(10, log_start + i * log_step);
                    }

                    float_64 integral = 0;

                    for (int i = 0; i < intervals - 1; ++i) {
                        float_64 z1 = zq_intervals[i];
                        float_64 z2 = zq_intervals[i + 1];
                        float_64 z3 = z1 + (z2 - z1) / 2;
                        float_64 y1 = std::cyl_bessel_k(5.0 / 3.0, z1);
                        float_64 y2 = std::cyl_bessel_k(5.0 / 3.0, z3);

                        integral += compute_params_and_integral_middle(z1, z3, z2, y1, y2);
                        
                    }

                    
                    return zq * integral;
                }


                // see paper "Synchrotron Radiation of Ultra-Relativistic Electrons" by A. A. Sokolov and I. M. Ternov -> don't see. gpt created this line

                float_64 F1_firstSynchrotronFunction(float_64 x){
                    if( x < 2.9e-6 ) return 2.15*std::pow(x,1./3);
                    else             return x * IntegrateQuadrature([](float_64 t) {return std::cyl_bessel_k(5. / 3, t); }, x, x + 20, 200);
                }
                float_64 F2_secondSynchrotronFunction(float_64 x) {
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
                    const float_64 interpolationPoints = 1000;

                    auto data_space = DataSpace<2>{interpolationPoints,2};
                    auto grid_layout = GridLayout<2>{data_space};
                    F1F2 = std::make_shared<GridBuffer<float_64,2>>(grid_layout);
                    

                    float_64 minZqExp = -20;
                    float_64 maxZqExp = 1;
                    std::cout << "Starting precomputation"<< std::endl;
                    // precompute F1 and F2 on log scale
                    for (uint32_t zq = 0; zq < interpolationPoints; zq++) {

                        float_64 zq_ = std::pow(10, minZqExp + (maxZqExp - minZqExp) * zq / float_64(interpolationPoints));
                        // inverse function for index retrieval:
                        // index = (log10(zq) - minZqExp) / (maxZqExp - minZqExp) * interpolationPoints;

                        float_64 F1 = F1_approximation_middle(zq_); 
                        float_64 F2 = F2_secondSynchrotronFunction(zq_);
                        

                        for (uint32_t i = 0; i < 2; i++)
                            F1F2->getHostBuffer().getDataBox()(DataSpace<2>{zq,i}) = (i == 0) ? F1 : F2;
                    }
                    std::cout << "Done precomputing F1 and F2."<<std::endl;
                    // print F1 and F2
                    for (uint32_t zq = 0; zq < interpolationPoints*100; zq++) {
                        float_64 zq_ = std::pow(10,minZqExp + (maxZqExp - minZqExp) * zq / float_64(interpolationPoints*100));
                        uint32_t index = (log10(zq_) - minZqExp) * (interpolationPoints) / (maxZqExp - minZqExp) ;
                        std::cout << "zq = " << zq/100.  << " index = " << index << std::endl;
                        // std::cout << "zq = " << zq_ << " F1 = " << F1F2->getHostBuffer().getDataBox()(DataSpace<2>{index,0}) << " F2 = " << F1F2->getHostBuffer().getDataBox()(DataSpace<2>{index,1}) << std::endl;
                    }

                    F1F2->hostToDevice();
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
                    synchrotronRadiation(cellDescription, step, F1F2->getDeviceBuffer().getDataBox());
                }

            private:
                //! Mapping for kernels
                MappingDesc cellDescription;
                // precomputed F1 and F2 -> 2d grid of floats_64 -> F1F2[zq][0/1] -> 0/1 = F1/F2
                std::shared_ptr<GridBuffer<float_64,2>> F1F2;
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
