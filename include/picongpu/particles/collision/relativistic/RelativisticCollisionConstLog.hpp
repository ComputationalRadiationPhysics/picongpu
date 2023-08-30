/* Copyright 2022 Rene Widera, Pawel Ordyna
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

#include "picongpu/particles/collision/kernels.def"
#include "picongpu/particles/collision/relativistic/RelativisticCollision.hpp"
#include "picongpu/particles/collision/relativistic/RelativisticCollisionConstLog.def"

#include <string>

namespace picongpu
{
    namespace particles
    {
        namespace collision
        {
            namespace relativistic
            {
                namespace acc
                {
                    //! Coulomb logarithm functor for a fixed logarithm defined at compile time
                    template<typename T_Param>
                    struct ConstCoulombLog
                    {
                        DINLINE float_COLL operator()(Variables const& v) const
                        {
                            return T_Param::coulombLog;
                        }
                    };

                } // namespace acc

                template<typename T_Param, bool ifDebug>
                struct RelativisticCollisionConstLogImpl
                {
                    template<typename T_Species0, typename T_Species1>
                    struct apply
                    {
                        using type = RelativisticCollisionConstLogImpl<T_Param, ifDebug>;
                    };
                    static constexpr bool ifDebug_m = ifDebug;
                    HINLINE RelativisticCollisionConstLogImpl(uint32_t currentStep){};

                    using AccFunctorImpl = acc::RelativisticCollision<acc::ConstCoulombLog<T_Param>, ifDebug>;
                    using AccFunctor = collision::acc::IBinary<AccFunctorImpl>;
                    // define kernel that should be used to call this functor
                    using CallingInterKernel = InterCollision<false>;
                    using CallingIntraKernel = IntraCollision<false>;

                    /** create device manipulator functor
                     *
                     * @param worker lockstep worker
                     * @param offset (in supercells, without any guards) to the origin of the local domain
                     * @param density0 cell density of the 1st species
                     * @param density1 cell density of the 2nd species
                     * @param potentialPartners number of potential collision partners for a macro particle in
                     *   the cell.
                     * @param coulombLog Coulomb logarithm
                     */
                    template<typename T_Worker>
                    HDINLINE auto operator()(
                        T_Worker const& worker,
                        DataSpace<simDim> const& offset,
                        float_X const& density0,
                        float_X const& density1,
                        uint32_t const& potentialPartners) const
                    {
                        using namespace picongpu::particles::collision::precision;
                        return AccFunctor{AccFunctorImpl{
                            math::pow(precisionCast<float_COLL>(density0), 2.0_COLL / 3.0_COLL),
                            math::pow(precisionCast<float_COLL>(density1), 2.0_COLL / 3.0_COLL),
                            potentialPartners}};
                    }

                    //! get the name of the functor
                    HINLINE static std::string getName()
                    {
                        return "RelativisticCollisionConstLog";
                    }
                };
            } // namespace relativistic
        } // namespace collision
    } // namespace particles
} // namespace picongpu
