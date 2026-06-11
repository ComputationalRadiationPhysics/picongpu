/*
 * SPDX-FileCopyrightText: Axel Huebl, Benjamin Worpitz, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/traits/GetMargin.hpp"

#include <pmacc/dimensions/DataSpace.hpp>

namespace picongpu
{
    namespace fields
    {
        namespace currentInterpolation
        {
            /* None interpolated current assignment functor
             *
             * Default for staggered grids/FDTD field solver.
             * Updates field E only.
             */
            struct None
            {
                static constexpr uint32_t dim = simDim;

                using LowerMargin = typename pmacc::math::CT::make_Int<dim, 0>::type;
                using UpperMargin = LowerMargin;

                /** Perform pointwise E(idx) += coeff * J(idx)
                 *
                 * @tparam T_DataBoxE electric field data box type
                 * @tparam T_DataBoxB magnetic field data box type
                 * @tparam T_DataBoxJ current density data box type
                 *
                 * @param fieldE electric field data box
                 * @param fieldB magnetic field data box
                 * @param fieldJ current density data box
                 * @param coeff coefficient value
                 */
                template<typename T_DataBoxE, typename T_DataBoxB, typename T_DataBoxJ>
                HDINLINE void operator()(
                    T_DataBoxE fieldE,
                    T_DataBoxB const,
                    T_DataBoxJ const fieldJ,
                    float_X const coeff)
                {
                    DataSpace<dim> const self;
                    fieldE(self) += coeff * fieldJ(self);
                }

                static pmacc::traits::StringProperty getStringProperties()
                {
                    pmacc::traits::StringProperty propList("name", "none");
                    return propList;
                }
            };

        } // namespace currentInterpolation
    } // namespace fields

    namespace traits
    {
        /* Get margin of the None current interpolation
         *
         * This class defines a LowerMargin and an UpperMargin.
         */
        template<>
        struct GetMargin<fields::currentInterpolation::None>
        {
        private:
            using MyInterpolation = fields::currentInterpolation::None;

        public:
            using LowerMargin = typename MyInterpolation::LowerMargin;
            using UpperMargin = typename MyInterpolation::UpperMargin;
        };

    } // namespace traits
} // namespace picongpu
