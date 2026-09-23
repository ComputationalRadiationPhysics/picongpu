/* Copyright 2022-2024 Rene Widera, Pawel Ordyna, Filip Optolowicz
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

#include "picongpu/particles/fusion/kernels.def"
#include "picongpu/particles/fusion/relativistic/FusionAlgorithm.hpp"

#include <string>

namespace picongpu::particles::fusion::relativistic
{
    namespace acc
    {

        // clang-format off
        //! Compile-time Cross-section functor - Pade Polynomial
        template<typename T_Param>
        struct CalcCrossSection
        {
            DINLINE float_COLL operator()(float_COLL const& Energy) const
            {
                float_COLL const Snum =
                        ((((T_Param::A5) * Energy +
                            T_Param::A4) * Energy +
                            T_Param::A3) * Energy +
                            T_Param::A2) * Energy +
                            T_Param::A1;
                float_COLL const Sden =
                        (((((T_Param::B4) * Energy +
                            T_Param::B3) * Energy +
                            T_Param::B2) * Energy +
                            T_Param::B1) * Energy +
                            1._COLL);
                float_COLL const S = Snum / Sden;
                float_COLL const Eexp = Energy * math::exp(T_Param::BG / math::sqrt(Energy));
                return S / Eexp;
            }
        };

        // clang-format on

    } // namespace acc

    template<typename T_Param>
    struct FusionFunctorImpl
    {
        template<typename T_Species0, typename T_Species1, typename T_Species2, typename T_Species3>
        struct apply
        {
            using type = FusionFunctorImpl<T_Param>;
        };

        HINLINE FusionFunctorImpl(uint32_t currentStep) {};

        using AccFunctorImpl = acc::FusionAlg<acc::CalcCrossSection<T_Param>>;
        using AccFunctor = fusion::acc::IBinary<AccFunctorImpl>;
        using CallingInterKernel = InterCollision;
        using CallingIntraKernel = IntraCollision;

        /**
         * @brief This operator creates and returns an instance of `AccFunctor` which is
         * an alias for the `IBinary` interface with `FusionAlg` as its implementation.
         *
         * The `FusionAlg` is instantiated with the `CalcCrossSection` functor, which
         * is responsible for calculating the fusion cross-section based on the reaction parameters
         * defined by `T_Param`. This allows the fusion algorithm to be a general template
         * that is configured at compile time with the specific reaction physics.
         *
         * This `operator()` is called to retrieve the functor object that will be used
         * by 'InterCollision.hpp/IntraCollision.hpp' to execute the fusion kernel on the GPU.
         *
         * @return An instance of the `AccFunctor`, ready for use in a fusion kernel.
         */
        HDINLINE auto operator()()
        {
            using namespace picongpu::particles::collision::precision;
            return AccFunctor{AccFunctorImpl{}};
        }

        //! get the name of the functor
        HINLINE static std::string getName()
        {
            return "FusionFunctor";
        }
    };
} // namespace picongpu::particles::fusion::relativistic
