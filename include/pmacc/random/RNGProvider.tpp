/* Copyright 2015-2021 Alexander Grund
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/random/RNGProvider.hpp"
#include "pmacc/dimensions/DataSpaceOperations.hpp"
#include "pmacc/Environment.hpp"
#include "pmacc/mappings/threads/ForEachIdx.hpp"
#include "pmacc/mappings/threads/IdxConfig.hpp"
#include "pmacc/traits/GetNumWorkers.hpp"

#include <memory>


namespace pmacc
{
    namespace random
    {
        namespace kernel
        {
            template<uint32_t T_numWorkers, uint32_t T_blockSize, typename T_RNGMethod>
            struct InitRNGProvider
            {
                template<typename T_RNGBox, typename T_Space, typename T_Acc>
                DINLINE void operator()(T_Acc const& acc, T_RNGBox rngBox, uint32_t seed, const T_Space size) const
                {
                    using namespace mappings::threads;

                    constexpr uint32_t numWorkers = T_numWorkers;
                    uint32_t const workerIdx = cupla::threadIdx(acc).x;

                    using SupercellDomCfg = IdxConfig<T_blockSize, numWorkers>;

                    // each virtual worker initialize one rng state
                    ForEachIdx<SupercellDomCfg> forEachCell(workerIdx);

                    forEachCell([&](uint32_t const linearIdx, uint32_t const) {
                        uint32_t const linearTid = cupla::blockIdx(acc).x * T_blockSize + linearIdx;
                        if(linearTid >= size.productOfComponents())
                            return;

                        T_Space const cellIdx = DataSpaceOperations<T_Space::dim>::map(size, linearTid);
                        T_RNGMethod().init(acc, rngBox(cellIdx), seed, linearTid);
                    });
                }
            };

        } // namespace kernel

        template<uint32_t T_dim, class T_RNGMethod>
        RNGProvider<T_dim, T_RNGMethod>::RNGProvider(const Space& size, const std::string& uniqueId)
            : m_size(size)
            , m_uniqueId(uniqueId.empty() ? getName() : uniqueId)
            , buffer(new Buffer(size))
        {
            if(m_size.productOfComponents() == 0)
                throw std::invalid_argument("Cannot create RNGProvider with zero size");
        }

        template<uint32_t T_dim, class T_RNGMethod>
        void RNGProvider<T_dim, T_RNGMethod>::init(uint32_t seed)
        {
            const uint32_t blockSize = 256;

            constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<blockSize>::value;

            const uint32_t gridSize = (m_size.productOfComponents() + blockSize - 1u) / blockSize; // Round up

            auto bufferBox = buffer->getDeviceBuffer().getDataBox();

            PMACC_KERNEL(kernel::InitRNGProvider<numWorkers, blockSize, RNGMethod>{})
            (gridSize, numWorkers)(bufferBox, seed, m_size);
        }

        template<uint32_t T_dim, class T_RNGMethod>
        typename RNGProvider<T_dim, T_RNGMethod>::Handle RNGProvider<T_dim, T_RNGMethod>::createHandle(
            const std::string& id)
        {
            auto provider = Environment<>::get().DataConnector().get<RNGProvider>(id, true);
            Handle result(provider->getDeviceDataBox());
            Environment<>::get().DataConnector().releaseData(id);
            return result;
        }

        template<uint32_t T_dim, class T_RNGMethod>
        template<class T_Distribution>
        typename RNGProvider<T_dim, T_RNGMethod>::template GetRandomType<T_Distribution>::type RNGProvider<
            T_dim,
            T_RNGMethod>::createRandom(const std::string& id)
        {
            typedef typename GetRandomType<T_Distribution>::type ResultType;
            return ResultType(createHandle());
        }

        template<uint32_t T_dim, class T_RNGMethod>
        typename RNGProvider<T_dim, T_RNGMethod>::Buffer& RNGProvider<T_dim, T_RNGMethod>::getStateBuffer()
        {
            return *buffer;
        }

        template<uint32_t T_dim, class T_RNGMethod>
        typename RNGProvider<T_dim, T_RNGMethod>::DataBoxType RNGProvider<T_dim, T_RNGMethod>::getDeviceDataBox()
        {
            return buffer->getDeviceBuffer().getDataBox();
        }

        template<uint32_t T_dim, class T_RNGMethod>
        std::string RNGProvider<T_dim, T_RNGMethod>::getName()
        {
            /* generate a unique name (for this type!) to use as a default ID */
            return std::string("RNGProvider") + char('0' + dim) /* valid for 0..9 */
                + RNGMethod::getName();
        }

        template<uint32_t T_dim, class T_RNGMethod>
        SimulationDataId RNGProvider<T_dim, T_RNGMethod>::getUniqueId()
        {
            return m_uniqueId;
        }

        template<uint32_t T_dim, class T_RNGMethod>
        void RNGProvider<T_dim, T_RNGMethod>::synchronize()
        {
            buffer->deviceToHost();
        }

    } // namespace random
} // namespace pmacc
