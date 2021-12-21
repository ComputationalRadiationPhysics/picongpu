/* Copyright 2020-2021 Sergei Bastrakov
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


namespace picongpu
{
    namespace traits
    {
        /** Trait for cell type of a field solver
         *
         * Defines the resulting type as ::type.
         * By default falls back to T_FieldSolver::CellType.
         *
         * Note: it was originally indented to be put to a new namespace
         * picongpu::fields::traits, but this was not possible due to conflicts
         * with pmacc names lookup.
         *
         * @tparam T_FieldSolver field solver type
         */
        template<typename T_FieldSolver>
        struct GetCellType
        {
            //! Cell type, one of fields::cellType:: types
            using type = typename T_FieldSolver::CellType;
        };

    } // namespace traits
} // namespace picongpu
