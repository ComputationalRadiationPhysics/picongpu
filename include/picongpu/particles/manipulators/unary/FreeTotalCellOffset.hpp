/* Copyright 2017-2023 Rene Widera, Axel Huebl
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

#include "picongpu/defines.hpp"
#include "picongpu/particles/functor/User.hpp"
#include "picongpu/particles/functor/misc/TotalCellOffset.hpp"
#include "picongpu/particles/manipulators/unary/FreeTotalCellOffset.def"

#include <string>


namespace picongpu
{
    namespace particles
    {
        namespace manipulators
        {
            namespace unary
            {
                namespace acc
                {
                    template<typename T_Functor>
                    struct FreeTotalCellOffset : private T_Functor
                    {
                        using Functor = T_Functor;

                        HDINLINE FreeTotalCellOffset(
                            Functor const& functor,
                            DataSpace<simDim> const& superCellToLocalOriginCellOffset)
                            : T_Functor(functor)
                            , m_superCellToLocalOriginCellOffset(superCellToLocalOriginCellOffset)
                        {
                        }

                        /** call user functor
                         *
                         * The random number generator is initialized with the first call.
                         *
                         * @tparam T_Particle type of the particle to manipulate
                         * @tparam T_Args type of the arguments passed to the user functor
                         * @tparam T_Worker lockstep worker type
                         *
                         * @param worker lockstep worker
                         * @param particle particle which is given to the user functor
                         * @return void is used to enable the operator if the user functor expects two arguments
                         */
                        template<typename T_Particle, typename T_Worker>
                        HDINLINE void operator()(T_Worker const&, T_Particle& particle)
                        {
                            DataSpace<simDim> const cellInSuperCell = pmacc::math::mapToND(
                                SuperCellSize::toRT(),
                                static_cast<int>(particle[localCellIdx_]));
                            Functor::operator()(m_superCellToLocalOriginCellOffset + cellInSuperCell, particle);
                        }

                    private:
                        DataSpace<simDim> const m_superCellToLocalOriginCellOffset;
                    };
                } // namespace acc

                template<typename T_Functor>
                struct FreeTotalCellOffset
                    : protected functor::User<T_Functor>
                    , private functor::misc::TotalCellOffset
                {
                    using CellOffsetFunctor = functor::misc::TotalCellOffset;
                    using Functor = functor::User<T_Functor>;

                    template<typename T_SpeciesType>
                    struct apply
                    {
                        using type = FreeTotalCellOffset;
                    };

                    /** constructor
                     *
                     * @param currentStep current simulation time step
                     */
                    HINLINE FreeTotalCellOffset(uint32_t currentStep, IdGenerator idGen)
                        : Functor(currentStep, idGen)
                        , CellOffsetFunctor(currentStep)
                    {
                    }

                    /** create functor for the accelerator
                     *
                     * @tparam T_Worker lockstep worker type
                     *
                     * @param worker lockstep worker
                     * @param localSupercellOffset offset (in superCells, without any guards) relative
                     *                             to the origin of the local domain
                     * @param blockCfg configuration of the worker
                     */
                    template<typename T_Worker>
                    HDINLINE auto operator()(T_Worker const& worker, DataSpace<simDim> const& localSupercellOffset)
                        const -> acc::FreeTotalCellOffset<Functor>
                    {
                        auto& cellOffsetFunctor = *static_cast<CellOffsetFunctor const*>(this);
                        return acc::FreeTotalCellOffset<Functor>(
                            *static_cast<Functor const*>(this),
                            cellOffsetFunctor(worker, localSupercellOffset));
                    }

                    HINLINE static std::string getName()
                    {
                        // we provide the name from the param class
                        return Functor::name;
                    }
                };

            } // namespace unary
        } // namespace manipulators
    } // namespace particles
} // namespace picongpu
