/* Copyright 2019-2024 Rene Widera, Pawel Ordyna
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
#include "picongpu/particles/fusion/InterCollision.hpp"
// in later pull request:
// #include "picongpu/particles/fusion/IntraCollision.hpp"

#include <pmacc/particles/meta/FindByNameOrType.hpp>

#include <boost/mpl/apply.hpp>

#include <cstdio>

namespace picongpu
{
    namespace particles
    {
        namespace fusion
        {
            namespace detail
            {
                // in-between species fusion
                template<
                    typename T_CollisionFunctor,
                    typename T_FilterPair,
                    typename T_ReactantSpecies1,
                    typename T_ReactantSpecies2,
                    typename T_ProductSpecies1,
                    typename T_ProductSpecies2,
                    uint32_t colliderId,
                    uint32_t pairId>
                struct WithPeer
                {
                    void operator()(std::shared_ptr<DeviceHeap> const& deviceHeap, uint32_t currentStep)
                    {
                        DataConnector& dc = Environment<>::get().DataConnector();
                        auto idProvider = dc.get<IdProvider>("globalId");

                        DoInterCollision<
                            T_CollisionFunctor,
                            T_FilterPair,
                            T_ReactantSpecies1,
                            T_ReactantSpecies2,
                            T_ProductSpecies1,
                            T_ProductSpecies2,
                            colliderId,
                            pairId>{}(deviceHeap, currentStep, idProvider->getDeviceGenerator());
                    }
                };

                // same species fusion
                template<
                    typename T_CollisionFunctor,
                    typename T_FilterPair,
                    typename T_ReactantSpecies,
                    typename T_ProductSpecies1,
                    typename T_ProductSpecies2,
                    uint32_t colliderId,
                    uint32_t pairId>
                struct WithPeer<
                    T_CollisionFunctor,
                    T_FilterPair,
                    T_ReactantSpecies,
                    T_ReactantSpecies,
                    T_ProductSpecies1,
                    T_ProductSpecies2,
                    colliderId,
                    pairId>
                {
                    // implementation in later pull request
                    void operator()(std::shared_ptr<DeviceHeap> const& deviceHeap, uint32_t currentStep)
                    {
                        // assert false for now
                        static_assert(sizeof(T_ReactantSpecies) == 0, "Intra-species fusion not implemented yet.");
                    }
                };
            } // namespace detail

            /* Runs the binary collision algorithm for a pair of colliding species.
             *
             * These struct chooses the InterCollision algorithm if the colliding
             * species are two different species and the IntraCollision algorithm if
             * they are identical.
             *
             * @tparam T_CollisionFunctor A binary particle functor defining a collision
             *    between two macro particles.
             * @tparam T_BaseSpecies First species in the collision pair.
             * @tparam T_PeerSpecies Second species in the collision pair.
             * @tparam T_Params A struct defining `coulombLog` for the collisions.
             * @tparam T_FilterPair A pair of particle filters, each for each species
             *     in the colliding pair.
             */
            template<
                typename T_CollisionFunctor,
                typename T_ReactantSpecies1,
                typename T_ReactantSpecies2,
                typename T_ProductSpecies1,
                typename T_ProductSpecies2,
                typename T_FilterPair,
                uint32_t colliderId,
                uint32_t pairId>
            struct WithPeer
            {
                void operator()(std::shared_ptr<DeviceHeap> const& deviceHeap, uint32_t currentStep)
                {
                    using ReactantSpecies1
                        = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_ReactantSpecies1>;

                    using ReactantSpecies2
                        = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_ReactantSpecies2>;

                    using ProductSpecies1
                        = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_ProductSpecies1>;

                    using ProductSpecies2
                        = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_ProductSpecies2>;

                    using CollisionFunctor = typename boost::mpl::apply4<
                        T_CollisionFunctor,
                        ReactantSpecies1,
                        ReactantSpecies2,
                        ProductSpecies1,
                        ProductSpecies2>::type;

                    detail::WithPeer<
                        CollisionFunctor,
                        T_FilterPair,
                        ReactantSpecies1,
                        ReactantSpecies2,
                        ProductSpecies1,
                        ProductSpecies2,
                        colliderId,
                        pairId>{}(deviceHeap, currentStep);
                }
            };

        } // namespace fusion
    } // namespace particles
} // namespace picongpu
