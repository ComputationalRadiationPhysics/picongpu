/*
 * SPDX-FileCopyrightText: Rene Widera, Pawel Ordyna
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
                    HINLINE RelativisticCollisionConstLogImpl(uint32_t currentStep) {};

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
