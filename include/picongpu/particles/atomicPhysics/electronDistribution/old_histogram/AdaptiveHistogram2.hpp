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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

/** @file This file defines an adaptive one-dimensional histogram with variable
 *  bin widths, gaps in between stored bins and a variable number of bins.
 *
 * This histogram tries to reduce memory required, compared to fixed bin width,
 * by varying the bin width over the argument space X and allowing gaps where
 * bins are empty.
 * Basis of this histogram are argument points x_i, each of them the central point
 * of an interval of length dx_i. Therefore:
 *
 * x_i - x_{i-1} = (dx_i + dx_{i-1})/2 \and x_{i+1} - x_i = (dx_{i+1} + dx_i)/2
 *
 * This grid is defined by an application specific Relative Error Function(REF),
 * in the following manner
 * Starting from an inital argument point x_0, dx_i is linearly increased until
 * the REF first exceeds a given maximum acceptable error c. The dx_i value
 * before the last is an approximation of the best value of dx_i. The value of
 * dx_i is further refined by iteratively solving the equation:
 *
 * REF(dx_i, x_{i-1}+dx_{i-1} + dx_i/2)- c = 0
 *
 * for a given number number of
 * iterations N_iterations
 and a maximum acceptable relative
 * error.
 * Starting from the initial argument point x_0, the maximum bin size is
 * iteratively
 * Each
 * Initially the Histogram does not have a stored
 * bin, as data points are binned their position
 * a relative error function.
 * with application width varying width.
 * The in central argument x_i of a bin and its width dx_i are given by
 This is done by  bin boundaries independent of the histogram entry f(x_i)
 * , while keeping the histogram locally defined.
 * The fixed width of a bin is replaced by a width  * This histogram bins positions and widths of this histogram are
 determined such that an
 * application specific evaluation function g() is still below a user given
 consists of a varible number of bins, central argument x_i,
 * of differing width dx_i, potentially with gaps.
 * The bin width and central argument is maximised under the constraint that an application specific
 * evaluation function F() is still below a user given
 * threshold. This
 *
 * a double linked list of Energy bins of variable widths, with (maybe) gaps
 * - each bin has central Energy and width, width choosen to maximize width
 *    while staying below a given relative binning error
 * - gaps, missing bins, where empty bins are located
 * - can include arbitary high energies, up to implementation limit
 *
 * Template Parameters:
 *   T_Energy                ... data type used for energies
 *   T_Weight                ... data type used for weights
 *   T_HistogramBin         ... histogram bin type used, must be a specialisation
 *                              of the HistogramBinWeight
 *   physicalParticleMass    ... physical mass of the particle used
 *   orderIntegrationPolynom ... order of the polynom used in approximation of
 *                               the rate integral.
 * members (private):
 *  bins    ... double linked list of occupied bins
 *
 * members(public):
 *  acess to template parameters:
 *   dataTypeEnergy          ... =^= T_Energy
 *   dataTypeWeightElement   ... =^= T_Weight
 *   dataTypeHistogramBin    ... =^= T_HistogramBin
 *
 * member functions(public):
 *   void binParticle( T_Energy E, T_Weight w ) ... bin a particle of weight w
 *                                                   and energy E
 *   void mergeHistograms(                      ... merges two adaptive histo-
 *      AdaptiveHistogram<                           grams h1 and h2, h1.merge(h2)
 *          T_Weight,                                h2 will not be changed while
 *          T_Energy> h2                             the content of h2 will be
 *      )                                            added to h1
 *   T_weight getWeight( T_Energy E )           ... get Weight for a given Energy
 *                                                   E, the weight of the bin
 *                                                   containing E will be
 *                                                   returned
 */

#pragma once

#include <list>
#define _USE_MATH_DEFINES
#include <cmath>


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
                    template<typename T_Energy, typename T_Weight, typename T_Alg, typename T_HistogramBin, float_X physicalParticleMass, uint8_t orderIntegrationPolynom, >
                    class AdaptiveHistogram
                    {
                        /** this class defines the adaptive histogram as a class.
                         *
                         * see file for detailed description
                         */

                    private:
                        // double linked list of all non empty bins, ordered by central energy
                        std::list<T_HistogramBin<T_Energy, T_Weight>> bins;

                        // helper functions
                        static T_Alg scaledChebyshevNodes(uint8_t k, uint8_t N, T_Energy centralE, T_Energy deltaE)
                        {
                            /** returns the k-th of N scaled chebyshev nodes
                             *
                             * BEWARE: k = 1, ..., N, k=0 is not allowed
                             * BEWARE: the highest value node corresponds to the lowest k value
                             *
                             * see https://en.wikipedia.org/wiki/Chebyshev_nodes for more
                             * information
                             */

                            // check for bounds on k
                            PMACC_ASSERT_MSG(k >= 1, "chebyshev nodes are defined only for 1 <= k <= N");
                            PMACC_ASSERT_MSG(k <= N, "chebyshev nodes are defined only for 1 <= k <= N");

                            return static_cast<T_Alg>(std::cos((2 * k - 1) / static_cast<float>(2 * N) * M_PI))
                                * static_cast<T_Alg>(deltaE) / 2.0
                                + static_cast<T_Alg>(centralE);
                        }

                        /*static uint16_t factorial(uint8_t n)
                        {
                            uint16_t result = 1u;
                            for ( uint8_t i = 1u, i <= n, i++ ){
                                result *= i;
                            }
                        }*/

                        T_Alg specificRate(
                            T_Energy centralE,
                            T_Energy deltaE,
                            uint8_t orderExpansion,
                            uint8_t numberSigmaPoints)
                        {
                            /* this function returns the specific Rate R_s over the energy
                             *  interval I = (cE - dE/2, cE + dE/2] under the assumption of ions
                             *  being much slower than electrons.
                             *
                             * cE    ... centralEnergy
                             * dE    ... deltaEnergy
                             *
                             * The specific rate R_s(cE, dE) is defined as R_s(E)= R / f(E),
                             * to aproximate the R_S, the rate R is approximated with a central
                             * Taylor expansion, this allows the analytic solution of the integral
                             *
                             * R = \int { sigma * v_rel * f(E) }     ; v_rel ~ v_e
                             *
                             * The result is a sum over different orders of derivation of f(E),
                             * amongst other things. We disregard order one and higher, allowing
                             * us to seperate the electron density.
                             *
                             * => R_s (cE, dE ) = \int_{ cE - dE/2 }^{ cE + dE/2 }( sigma * v_e )
                             *
                             * R_s = ?
                             *
                             * R     ... collisional rate
                             * sigma ... collisional cross section
                             * v_e   ... electron veolcity
                             * v_rel ... relative velocity of collision partners
                             *
                             * see my master thesis for exact derivation
                             * usage explained in ?
                             */

                            // actual result
                            T_Alg result = 0;

                            /* 1/divisor(d) required as factor in each summand in summation
                             *
                             * d = m! * k! * (2*a+1-m-k)! ;k = 0, since k is order of f derivative
                             * d(m , a) = m! * (2*a+1-m)!
                             *
                             * d(0, a+1) = (2*a+2+1)! = (2*a+3) * (2*a+2) * (2*a+1)!
                             * d(0, a+1) = (2*a+3)*(2*a+2) * d(0, a) =°=factorial
                             *
                             * d(m+1, a) = (m+1)! * (2*a+1-(m+1))! = (m+1) * m! * (2*a+1-m-1)!
                             * d(m+1, a) = (m+1) * m! * (2*a+1-m)!/(2*a+1-m)
                             * d(m+1, a) = (m+1)/(2*a+1-m) * d(m, a)
                             */

                            int16_t divisor;
                            int16_t factorial = 1;

                            for(uint8_t a = orderIntegrationPolynom + 1, a <= errorOrder, a++)
                            {
                                constexpr uint8_t k = 0; // only 0th order in f(E)

                                divisor = factorial;

                                for(uint8_t m = 0u, m <= 2 * a + 1 - k, l++)
                                {
                                    result += static_cast<T_Alg>(1) / (divisor * (2 * a + 2))
                                        * this->derivativeSigma(m, m, centralE, deltaE)
                                        * this->derivativeVTerm(2 * a + 1 - k - m, centralE)
                                        * std::pow(deltaE / 2, 2 * a + 2) * 2;

                                    divisor = (divisor / static_cast<uint16_t>(2 * a + 1 - m));
                                    divisor *= m + 1;
                                }

                                factorial *= (2 * a + 2) * (2 * a + 3);
                            }

                            return result;
                        }

                        static T_Alg derivativeVTerm(uint8_t orderDerivation, T_Energy cE, T_Energy dE)
                        {
                            T_Alg result;

                            for(uint8_t m = 0, m <= orderDeriviation - 1, m++)
                            {
                                result *= (0.5 - m);
                            }

                            result *= static_cast<T_Alg>(
                                std::sqrt(static_cast<float>(2 / physicalParticleMass)) * std::pow(E, 0.5 - order));
                            return result;
                        }


                        static T_Alg w(T_Energy x, T_Energy centralE, T_energy deltaE, uint8_t n, uint8_t N)
                        {
                            /** this function return the value of w_n(x) =
                             *  \prod_{nu=0}^{n}( x - \alpha_nu )
                             *
                             *   w_n(x) = \prod_{nu=0}^{n}( x - \alpha_nu )
                             * with \alpha_nu being reordered scaled chebyshev nodes
                             *
                             *
                             */
                            T_Alg result = 0;

                            for(uint8_t i = 0, i <= order, i++)
                            {
                                result *= x - this->scaledChebyshevNodes(i, order)
                            }

                            float_X derivativeSigma(
                                uint8_t orderDerivation,
                                uint8_t orderNPlus,
                                T_Energy cE,
                                T_Energy dE)
                            {
                            }

                            crossSectionTermDerivation() calculateRelativeError()


                                public
                                :
                                // make template parameters available for later use
                                static constexpr using dataTypeEnergy
                                = T_Energy;
                            static constexpr using dataTypeWeight = T_Weight;
                            static constexpr using dataTypeHistogramBin = T_HistogramBin;

                            void binParticle(T_Energy E, T_Weight w)
                            {
                            }

                            void mergeHistograms(adaptiveHistogram<T_Weight, T_Energy> h2)
                            {
                            }

                            T_weight getWeight(T_Energy E)
                            {
                            }
                        }


                    } // namespace histogram
                } // namespace electronDistribution
            } // namespace atomic Physics
        } // namespace particles
    } // namespace picongpu
