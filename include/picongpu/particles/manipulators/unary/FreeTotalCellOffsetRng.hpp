/*
 * SPDX-FileCopyrightText: Rene Widera, Alexander Grund, Axel Huebl, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/functor/User.hpp"
#include "picongpu/particles/functor/misc/Rng.hpp"
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
                    /** Device-side functor
                     *
                     * @tparam T_Functor user-defined unary functor
                     * @tparam T_RngType rng functor type
                     */
                    template<typename T_Functor, typename T_RngType>
                    struct FreeTotalCellOffsetRng : private T_Functor
                    {
                        using Functor = T_Functor;
                        using RngType = T_RngType;

                        HDINLINE FreeTotalCellOffsetRng(
                            Functor const& functor,
                            DataSpace<simDim> const& superCellToLocalOriginCellOffset,
                            RngType const& rng)
                            : T_Functor(functor)
                            , m_superCellToLocalOriginCellOffset(superCellToLocalOriginCellOffset)
                            , m_rng(rng)
                        {
                        }

                        /** call user functor
                         *
                         * @tparam T_Particle type of the particle to manipulate
                         * @tparam T_Worker lockstep worker type
                         *
                         * @param worker lockstep worker
                         * @param particle particle which is given to the user functor
                         */
                        template<typename T_Particle, typename T_Worker>
                        HDINLINE void operator()(T_Worker const&, T_Particle& particle)
                        {
                            DataSpace<simDim> const cellInSuperCell = pmacc::math::mapToND(
                                SuperCellSize::toRT(),
                                static_cast<int>(particle[localCellIdx_]));
                            Functor::operator()(m_superCellToLocalOriginCellOffset + cellInSuperCell, m_rng, particle);
                        }

                    private:
                        DataSpace<simDim> const m_superCellToLocalOriginCellOffset;
                        RngType m_rng;
                    };
                } // namespace acc

                template<typename T_Functor, typename T_Distribution>
                struct FreeTotalCellOffsetRng
                    : protected functor::User<T_Functor>
                    , private functor::misc::TotalCellOffset
                    , private functor::misc::Rng<T_Distribution>
                {
                    using CellOffsetFunctor = functor::misc::TotalCellOffset;
                    using Functor = functor::User<T_Functor>;

                    using RngGenerator = functor::misc::Rng<T_Distribution>;
                    using Distribution = T_Distribution;

                    template<typename T_SpeciesType>
                    struct apply
                    {
                        using type = FreeTotalCellOffsetRng;
                    };

                    /** constructor
                     *
                     * @param currentStep current simulation time step
                     */
                    HINLINE FreeTotalCellOffsetRng(uint32_t currentStep, IdGenerator idGen)
                        : Functor(currentStep, idGen)
                        , CellOffsetFunctor(currentStep)
                        , RngGenerator(currentStep)
                    {
                    }

                    /** Create functor for the accelerator
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
                        const
                    {
                        auto& cellOffsetFunctor = *static_cast<CellOffsetFunctor const*>(this);
                        auto const rng = (*static_cast<RngGenerator const*>(this))(worker, localSupercellOffset);
                        return acc::FreeTotalCellOffsetRng<Functor, std::decay_t<decltype(rng)>>(
                            *static_cast<Functor const*>(this),
                            cellOffsetFunctor(worker, localSupercellOffset),
                            rng);
                    }

                    HINLINE static std::string getName()
                    {
                        return Functor::name;
                    }
                };

            } // namespace unary
        } // namespace manipulators
    } // namespace particles
} // namespace picongpu
