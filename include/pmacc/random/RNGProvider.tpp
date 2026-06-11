/*
 * SPDX-FileCopyrightText: Alexander Grund, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/lockstep/lockstep.hpp"
#include "pmacc/random/RNGProvider.hpp"

#include <memory>

namespace pmacc
{
    namespace random
    {
        namespace kernel
        {
            template<uint32_t T_blockSize, typename T_RNGMethod>
            struct InitRNGProvider
            {
                template<typename T_Worker, typename T_RNGBox, typename T_Space>
                DINLINE void operator()(T_Worker const& worker, T_RNGBox rngBox, uint32_t seed, T_Space const size)
                    const
                {
                    // each virtual worker initialize one rng state
                    auto forEachCell = lockstep::makeForEach<T_blockSize>(worker);

                    forEachCell(
                        [&](int32_t const linearIdx)
                        {
                            int32_t const linearTid = worker.blockDomIdxND().x() * T_blockSize + linearIdx;
                            if(linearTid >= size.productOfComponents())
                                return;

                            T_Space const cellIdx = math::mapToND(size, linearTid);
                            T_RNGMethod().init(worker, rngBox(cellIdx), seed, linearTid);
                        });
                }
            };

        } // namespace kernel

        template<uint32_t T_dim, class T_RNGMethod>
        RNGProvider<T_dim, T_RNGMethod>::RNGProvider(Space const& size, std::string const& uniqueId)
            : m_size(size)
            , buffer(std::make_unique<Buffer>(size))
            , m_uniqueId(uniqueId.empty() ? getName() : uniqueId)
        {
            if(m_size.productOfComponents() == 0)
                throw std::invalid_argument("Cannot create RNGProvider with zero size");
        }

        template<uint32_t T_dim, class T_RNGMethod>
        void RNGProvider<T_dim, T_RNGMethod>::init(uint32_t seed)
        {
            constexpr uint32_t blockSize = 256;

            uint32_t const gridSize = (m_size.productOfComponents() + blockSize - 1u) / blockSize; // Round up

            auto bufferBox = buffer->getDeviceBuffer().getDataBox();

            PMACC_LOCKSTEP_KERNEL(kernel::InitRNGProvider<blockSize, RNGMethod>{})
                .template config<blockSize>(gridSize)(bufferBox, seed, m_size);
        }

        template<uint32_t T_dim, class T_RNGMethod>
        typename RNGProvider<T_dim, T_RNGMethod>::Handle RNGProvider<T_dim, T_RNGMethod>::createHandle(
            std::string const& id)
        {
            auto provider = Environment<>::get().DataConnector().get<RNGProvider>(id);
            Handle result(provider->getDeviceDataBox());
            return result;
        }

        template<uint32_t T_dim, class T_RNGMethod>
        template<class T_Distribution>
        typename RNGProvider<T_dim, T_RNGMethod>::template GetRandomType<T_Distribution>::type RNGProvider<
            T_dim,
            T_RNGMethod>::createRandom(std::string const& id)
        {
            using ResultType = typename GetRandomType<T_Distribution>::type;
            return ResultType(createHandle());
        }

        template<uint32_t T_dim, class T_RNGMethod>
        typename RNGProvider<T_dim, T_RNGMethod>::Buffer& RNGProvider<T_dim, T_RNGMethod>::getStateBuffer()
        {
            return *buffer;
        }

        template<uint32_t T_dim, class T_RNGMethod>
        typename RNGProvider<T_dim, T_RNGMethod>::Space RNGProvider<T_dim, T_RNGMethod>::getSize() const
        {
            return m_size;
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

        template<uint32_t T_dim, class T_RNGMethod>
        void RNGProvider<T_dim, T_RNGMethod>::syncToDevice()
        {
            buffer->hostToDevice();
        }

    } // namespace random
} // namespace pmacc
