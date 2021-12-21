/* Copyright 2015-2021 Axel Huebl, Benjamin Worpitz, Klaus Steiniger, Sergei Bastrakov
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

#include "picongpu/simulation_defines.hpp"

#include <pmacc/dimensions/DataSpace.hpp>


namespace picongpu
{
    namespace fields
    {
        namespace currentInterpolation
        {
            namespace detail
            {
                template<uint32_t T_dim>
                struct Binomial;

                //! Specialization for 3D
                template<>
                struct Binomial<DIM3>
                {
                    static constexpr uint32_t dim = DIM3;

                    using LowerMargin = typename pmacc::math::CT::make_Int<dim, 1>::type;
                    using UpperMargin = LowerMargin;

                    template<typename T_DataBoxE, typename T_DataBoxB, typename T_DataBoxJ>
                    HDINLINE void operator()(T_DataBoxE fieldE, T_DataBoxB const, T_DataBoxJ const fieldJ)
                    {
                        using TypeJ = typename T_DataBoxJ::ValueType;
                        using DS = DataSpace<dim>;

                        // weighting for original value, i.e. center element of a cell
                        constexpr float_X M = 8.0;
                        // weighting for nearest neighbours, i.e. cells sharing a face with the center cell
                        constexpr float_X S = 4.0;
                        // weighting for next to nearest neighbours, i.e. cells sharing an edge with the center cell
                        constexpr float_X D = 2.0;
                        // weighting for farthest neighbours, i.e. cells sharing a corner with the center cell
                        constexpr float_X T = 1.0;

                        TypeJ averagedJ =
                            // sum far neighbours, i.e. corner elements, weighting T
                            T
                                * (fieldJ(DS(-1, -1, -1)) + fieldJ(DS(+1, -1, -1)) + fieldJ(DS(-1, +1, -1))
                                   + fieldJ(DS(+1, +1, -1)) + fieldJ(DS(-1, -1, +1)) + fieldJ(DS(+1, -1, +1))
                                   + fieldJ(DS(-1, +1, +1)) + fieldJ(DS(+1, +1, +1)))
                            +
                            // sum next to nearest neighbours, i.e. edge elements, weighting D
                            D
                                * (fieldJ(DS(-1, -1, 0)) + fieldJ(DS(+1, -1, 0)) + fieldJ(DS(-1, +1, 0))
                                   + fieldJ(DS(+1, +1, 0)) + fieldJ(DS(-1, 0, -1)) + fieldJ(DS(+1, 0, -1))
                                   + fieldJ(DS(-1, 0, +1)) + fieldJ(DS(+1, 0, +1)) + fieldJ(DS(0, -1, -1))
                                   + fieldJ(DS(0, +1, -1)) + fieldJ(DS(0, -1, +1)) + fieldJ(DS(0, +1, +1)))
                            +
                            // sum next neighbours, i.e. face elements, weighting S
                            S
                                * (fieldJ(DS(-1, 0, 0)) + fieldJ(DS(+1, 0, 0)) + fieldJ(DS(0, -1, 0))
                                   + fieldJ(DS(0, +1, 0)) + fieldJ(DS(0, 0, -1)) + fieldJ(DS(0, 0, +1)))
                            +
                            // add original value, i.e. center element, weighting M
                            M * (fieldJ(DS(0, 0, 0)));

                        /* calc average by normalizing weighted sum In 3D there are:
                         *   - original value with weighting M
                         *   - 6 nearest neighbours with weighting S
                         *   - 12 next to nearest neighbours with weighting D
                         *   - 8 farthest neighbours with weighting T
                         */
                        constexpr float_X inverseDivisor = 1._X / (M + 6._X * S + 12._X * D + 8._X * T);
                        averagedJ *= inverseDivisor;

                        constexpr float_X deltaT = DELTA_T;
                        *fieldE -= averagedJ * (1._X / EPS0) * deltaT;
                    }
                };


                //! Specialization for 2D
                template<>
                struct Binomial<DIM2>
                {
                    static constexpr uint32_t dim = DIM2;

                    using LowerMargin = typename pmacc::math::CT::make_Int<dim, 1>::type;
                    using UpperMargin = LowerMargin;

                    template<typename T_DataBoxE, typename T_DataBoxB, typename T_DataBoxJ>
                    HDINLINE void operator()(T_DataBoxE fieldE, T_DataBoxB const, T_DataBoxJ const fieldJ)
                    {
                        using TypeJ = typename T_DataBoxJ::ValueType;
                        using DS = DataSpace<dim>;

                        // weighting for original value, i.e. center element of a cell
                        constexpr float_X M = 4.0;
                        // weighting for nearest neighbours, i.e. cells sharing an edge with the center cell
                        constexpr float_X S = 2.0;
                        // weighting for next to nearest neighbours, i.e. cells sharing a corner with the center cell
                        constexpr float_X D = 1.0;

                        TypeJ averagedJ =
                            // sum next to nearest neighbours, i.e. corner neighbors, weighting D
                            D * (fieldJ(DS(-1, -1)) + fieldJ(DS(+1, -1)) + fieldJ(DS(-1, +1)) + fieldJ(DS(+1, +1))) +
                            // sum next neighbours, i.e. edge neighbors, weighting S
                            S * (fieldJ(DS(-1, 0)) + fieldJ(DS(+1, 0)) + fieldJ(DS(0, -1)) + fieldJ(DS(0, +1))) +
                            // add original value, i.e. center cell, weighting M
                            M * (fieldJ(DS(0, 0)));

                        /* calc average by normalizing weighted sum
                         * In 2D there are:
                         *    - original value with weighting M
                         *    - 4 nearest neighbours with weighting S
                         *    - 4 next to nearest neighbours with weighting D
                         */
                        constexpr float_X inverseDivisor = 1._X / (M + 4._X * S + 4._X * D);
                        averagedJ *= inverseDivisor;

                        constexpr float_X deltaT = DELTA_T;
                        *fieldE -= averagedJ * (1._X / EPS0) * deltaT;
                    }
                };

            } // namespace detail


            /** Smoothing the current density before passing it to the field solver
             *
             * This technique mitigates numerical Cherenkov effects and short wavelength
             * instabilities as it effectively implements a low pass filter which
             * damps high frequency noise (near the Nyquist frequency) in the
             * current distribution.
             *
             * A description and a two-dimensional implementation of this filter
             * is given in
             * CK Birdsall, AB Langdon. Plasma Physics via Computer Simulation. Appendix C. Taylor & Francis, 2004.
             * It is a 2D version of the commonly used one-dimensional three points filter with binomial coefficients
             *
             * The three-dimensional extension of the above two-dimensional smoothing scheme
             * uses all 26 neighbors of a cell.
             *
             * Smooths the current before assignment in staggered grid.
             * Updates E & breaks local charge conservation slightly.
             */
            struct Binomial : public detail::Binomial<simDim>
            {
                static pmacc::traits::StringProperty getStringProperties()
                {
                    pmacc::traits::StringProperty propList("name", "Binomial");
                    propList["param"] = "period=1;numPasses=1;compensator=false";
                    return propList;
                }
            };

        } // namespace currentInterpolation
    } // namespace fields

    namespace traits
    {
        /* Get margin of the Binomial current interpolation
         *
         * This class defines a LowerMargin and an UpperMargin.
         */
        template<>
        struct GetMargin<fields::currentInterpolation::Binomial>
        {
        private:
            using MyInterpolation = fields::currentInterpolation::Binomial;

        public:
            using LowerMargin = typename MyInterpolation::LowerMargin;
            using UpperMargin = typename MyInterpolation::UpperMargin;
        };

    } // namespace traits
} // namespace picongpu
