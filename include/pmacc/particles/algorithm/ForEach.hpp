/* Copyright 2017-2022 Axel Huebl, Rene Widera, Sergei Bastrakov
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

#include "pmacc/Environment.hpp"
#include "pmacc/attribute/Constexpr.hpp"
#include "pmacc/lockstep.hpp"
#include "pmacc/lockstep/Config.hpp"
#include "pmacc/lockstep/ForEach.hpp"
#include "pmacc/mappings/kernel/AreaMapping.hpp"
#include "pmacc/particles/algorithm/detail/ForEach.hpp"
#include "pmacc/particles/frame_types.hpp"

#include <cstdint>
#include <type_traits>
#include <utility>


namespace pmacc::particles::algorithm
{
    namespace acc
    {
        //! Policy to forward iterate over particles/frames of a supercell.
        struct Forward
        {
            template<typename T_ParBox>
            static DINLINE auto getFirst(T_ParBox& pb, DataSpace<T_ParBox::Dim> const& superCellIdx)
            {
                return pb.getFirstFrame(superCellIdx);
            }

            template<typename T_ParBox>
            static DINLINE auto next(T_ParBox& pb, typename T_ParBox::FramePtr const& framePtr)
            {
                return pb.getNextFrame(framePtr);
            }

            template<typename T_ForEach>
            static DINLINE auto createCtx(T_ForEach& forEachParticleInFrame)
            {
                return lockstep::makeVar<uint32_t>(forEachParticleInFrame, 0u);
            }
        };

        //! Policy to reverse iterate over particles/frames of a supercell.
        struct Reverse
        {
            template<typename T_ParBox>
            static DINLINE auto getFirst(T_ParBox& pb, DataSpace<T_ParBox::Dim> const& superCellIdx)
            {
                return pb.getLastFrame(superCellIdx);
            }

            template<typename T_ParBox>
            static DINLINE auto next(T_ParBox& pb, typename T_ParBox::FramePtr const& framePtr)
            {
                return pb.getPreviousFrame(framePtr);
            }

            template<typename T_ForEach>
            static DINLINE auto createCtx(T_ForEach& forEachParticleInFrame)
            {
                return lockstep::makeVar<lcellId_t>(forEachParticleInFrame, lcellId_t(0u));
            }
        };

        /** Execute a particle or frame functor for each particle/frame.
         *
         * @tparam T_AccessTag Type of the tag to mark if the particle or frame interface will be used.
         *                     valid options: detail::CallParticleFunctor or detail::CallFrameFunctor
         * @tparam T_Order Iteration order. valid options: Forward or Reverse
         * @tparam T_ParBox Type of the particle box to traverse.
         * @tparam T_numWorkers Number of lockstep workers.
         */
        template<typename T_AccessTag, typename T_Order, typename T_ParBox, uint32_t T_numWorkers>
        class ForEachParticle
        {
        private:
            static constexpr uint32_t dim = T_ParBox::Dim;

            using SuperCellSize = typename T_ParBox::FrameType::SuperCellSize;
            static constexpr uint32_t frameSize = pmacc::math::CT::volume<SuperCellSize>::type::value;
            static constexpr uint32_t numWorkers = T_numWorkers;

            /** Number of frames to skip.
             *
             * If we have more workers than particles in a frame we skip frames depending on the worker id.
             */
            static constexpr uint32_t frameDistance = (numWorkers + frameSize - 1u) / frameSize;

            using ForEachParticleInFrame
                = lockstep::ForEach<lockstep::Config<frameSize * frameDistance, T_numWorkers, 1>>;

            DataSpace<dim> const m_superCellIdx;
            ForEachParticleInFrame const forEachParticleInFrame;
            T_ParBox m_particlesBox;

        public:
            /** Provide the lockstep foreach executor to execute a functor for each particle within a set of frames.
             *
             * Depending of the number of workers used it could be that more than one frame is executed by all workers
             * in parallel. The frame functor interface is therefore a context with one frame per virtual worker.
             * It is guaranteed that the executor fits to the frameCtx provided by the frame functor interface. @see
             * detail::FrameFunctorInterface
             *
             * @return Lockstep foreach executor to operate on frames.
             */
            DINLINE auto lockstepForEach() const
            {
                return forEachParticleInFrame;
            }

            /** Construct algoritm to operate on particles in a superCell
             *
             * @param workerIdx workerIdx index of the worker: range [0;workerSize)
             * @param particlesBox particles memory
             *                     It is not allowed concurrently to add or remove particles during the
             *                     execution of this algorithm.
             * @param superCellIdx index of the superCell where particles should be processed
             */
            DINLINE ForEachParticle(
                uint32_t const workerIdx,
                T_ParBox const& particlesBox,
                DataSpace<dim> const& superCellIdx)
                : m_superCellIdx(superCellIdx)
                , forEachParticleInFrame(workerIdx)
                , m_particlesBox(particlesBox)
            {
            }

            /** Execute unary functor for each particle.
             *
             * @attention There is no guarantee in which order particles will be executed.
             *            If the particle functor is used the algorithm assumes that the frame structure is following
             *            the rule that all frames but the last are fully filled and in the last frame the low indices
             *            are contiguously filled.
             *
             * @tparam T_Acc alpaka accelerator type
             * @tparam T_Functor unary particle functor or frame functor following the interface concept
             *                           detail::FrameFunctorInterface or detail::ParticleFunctorInterface
             * @param acc alpaka accelerator
             * @param unaryFunctor Functor executed for each particle/frame.
             *                     The caller must ensure that calling the functor in parallel with
             *                     different workers is data race free.
             *                     - particle functor: 'void operator()(T_Acc const &, ParticleType)'.
             *                       It is not allowed to call a synchronization function within the functor.
             *                     - frame functor: 'void operator()(T_Acc const &, FrameCtx)'.
             *                       Calling synchronization within the functor is allowed.
             */
            template<typename T_Acc, typename T_Functor>
            DINLINE void operator()(T_Acc const& acc, T_Functor&& unaryFunctor) const
            {
                if constexpr(numWorkers > frameSize)
                {
                    PMACC_CASSERT_MSG(
                        __number_of_workers_must_be_either_less_or_a_multiple_of_framesize,
                        (numWorkers % frameSize) == 0);
                }

                constexpr bool isSupportedOrder = std::is_same_v<Reverse, T_Order> || std::is_same_v<Forward, T_Order>;
                static_assert(isSupportedOrder, "Unsupported order policy");

                auto& superCell = m_particlesBox.getSuperCell(m_superCellIdx);
                uint32_t const numParticlesInSupercell = superCell.getNumParticles();

                // end kernel if we have no particles
                if(numParticlesInSupercell == 0)
                    return;

                // Information used to know if a particle within a frame is active without checking the multiMask.
                auto frameConditionData = T_Order::createCtx(forEachParticleInFrame);

                // get starting frame
                auto framePtrCtx = forEachParticleInFrame(
                    [&](lockstep::Idx const idx)
                    {
                        auto frame = T_Order::getFirst(m_particlesBox, m_superCellIdx);

                        if constexpr(std::is_same_v<Reverse, T_Order>)
                        {
                            if(frame.isValid())
                                frameConditionData[idx] = superCell.getSizeLastFrame();
                        }

                        /* Same as frameDistance, variable is required to workaround an nvcc compiler bug.
                         * Without redefinition within the lambda the variable will be undefined.
                         */
                        constexpr uint32_t frameDist = (numWorkers + frameSize - 1u) / frameSize;
                        if constexpr(frameDist > 1)
                        {
                            uint32_t const skipFrames = idx / frameSize;
                            /* select N-th (N=frameIdx) frame from the list */
                            for(uint32_t i = 1; i <= skipFrames && frame.isValid(); ++i)
                            {
                                frame = T_Order::next(m_particlesBox, frame);
                            }
                            if constexpr(std::is_same_v<Reverse, T_Order>)
                                frameConditionData[idx] = skipFrames != 0 ? frameSize : frameConditionData[idx];
                        }
                        return frame;
                    });

                bool hasMoreWork = true;
                // iterate over all frames in the supercell
                do
                {
                    if constexpr(std::is_same_v<T_AccessTag, detail::CallParticleFunctor>)
                    {
                        // loop over all particles in the frame
                        forEachParticleInFrame(
                            [&](lockstep::Idx const linearIdx)
                            {
                                auto const particleIdx = linearIdx % frameSize;

                                auto particle = framePtrCtx[linearIdx][particleIdx];
                                if constexpr(std::is_same_v<Reverse, T_Order>)
                                {
                                    bool const isParticle = particleIdx < frameConditionData[linearIdx];
                                    if(!isParticle)
                                        particle.setHandleInvalid();
                                }
                                else if constexpr(std::is_same_v<Forward, T_Order>)
                                {
                                    uint32_t const parIdxInSupercell = frameConditionData[linearIdx] + linearIdx;
                                    bool const isParticle = parIdxInSupercell < numParticlesInSupercell;
                                    if(!isParticle)
                                        particle.setHandleInvalid();
                                }


                                PMACC_CASSERT_MSG(
                                    __unaryParticleFunctor_must_return_void,
                                    std::is_void_v<decltype(unaryFunctor(acc, particle))>);
                                if(particle.isHandleValid())
                                {
                                    auto particleFunctor
                                        = detail::makeParticleFunctorInterface(std::forward<T_Functor>(unaryFunctor));
                                    particleFunctor(acc, particle);
                                }
                            });
                    }
                    else if constexpr(std::is_same_v<T_AccessTag, detail::CallFrameFunctor>)
                    {
                        auto frameFunctor = detail::makeFrameFunctorInterface(std::forward<T_Functor>(unaryFunctor));
                        frameFunctor(acc, framePtrCtx);
                    }

                    hasMoreWork = false;

                    // get next frame
                    forEachParticleInFrame(
                        [&](lockstep::Idx const idx)
                        {
                            if constexpr(std::is_same_v<Reverse, T_Order>)
                                frameConditionData[idx] = frameSize;
                            else if constexpr(std::is_same_v<Forward, T_Order>)
                                frameConditionData[idx] += frameSize * frameDistance;

                            for(uint32_t i = 0; i < frameDistance; ++i)
                                if(framePtrCtx[idx].isValid())
                                    framePtrCtx[idx] = T_Order::next(m_particlesBox, framePtrCtx[idx]);

                            hasMoreWork = hasMoreWork || framePtrCtx[idx].isValid();
                        });

                } while(hasMoreWork);
            }

            DINLINE bool hasParticles() const
            {
                return numParticles() != 0u;
            }

            DINLINE uint32_t numParticles() const
            {
                auto& superCell = m_particlesBox.getSuperCell(m_superCellIdx);
                return superCell.getNumParticles();
            }
        };

        /** Creates an executor to iterate over all particles in a supercell.
         *
         * @return ForEach executor which can be invoked with a particle functor as argument. @see
         * detail::ParticleFunctorInterface
         *
         * @{
         * @tparam T_numWorkers number of lockstep workers
         * @tparam T_ParBox Type of the particle box.
         * @param workerIdx worker index
         * @param particlesBox particle box
         * @param superCellIdx supercell index
         */
        template<uint32_t T_numWorkers, typename T_ParBox>
        DINLINE auto makeForEach(
            uint32_t workerIdx,
            T_ParBox const& particlesBox,
            DataSpace<T_ParBox::Dim> const& superCellIdx)
        {
            return ForEachParticle<detail::CallParticleFunctor, Reverse, T_ParBox, T_numWorkers>(
                workerIdx,
                particlesBox,
                superCellIdx);
        }

        /** Creates an executor to iterate over all frames in a supercell.
         *
         * @tparam T_Order Iteration order. valid options: Forward or Reverse
         * @return ForEach executor which can be invoked with a frame functor as argument. @see
         * detail::FrameFunctorInterface
         */
        template<uint32_t T_numWorkers, typename T_Order, typename T_ParBox>
        DINLINE auto makeForEachFrame(
            uint32_t workerIdx,
            T_ParBox const& particlesBox,
            DataSpace<T_ParBox::Dim> const& superCellIdx)
        {
            return ForEachParticle<detail::CallFrameFunctor, T_Order, T_ParBox, T_numWorkers>(
                workerIdx,
                particlesBox,
                superCellIdx);
        }
        /** Creates an executor to iterate over all particles in a supercell.
         *
         * @tparam T_Order Iteration order. valid options: Forward or Reverse
         * @return ForEach executor which can be invoked with a particle functor as argument. @see
         * detail::ParticleFunctorInterface
         */
        template<uint32_t T_numWorkers, typename T_Order, typename T_ParBox>
        DINLINE auto makeForEach(
            uint32_t workerIdx,
            T_ParBox const& particlesBox,
            DataSpace<T_ParBox::Dim> const& superCellIdx)
        {
            return ForEachParticle<detail::CallParticleFunctor, T_Order, T_ParBox, T_numWorkers>(
                workerIdx,
                particlesBox,
                superCellIdx);
        }
        /** @} */

        namespace detail
        {
            /** operate on particles of a species
             *
             * @tparam T_numWorkers number of workers
             */
            template<uint32_t T_numWorkers>
            struct KernelForEachParticle
            {
                /** operate on particles
                 *
                 * @tparam T_Acc alpaka accelerator type
                 * @tparam T_Functor type of the functor to operate on a particle
                 * @tparam T_Mapping mapping functor type
                 * @tparam T_ParBox pmacc::ParticlesBox, type of the species box
                 *
                 * @param acc alpaka accelerator
                 * @param functor functor to operate on a particle
                 *                must fulfill the interface pmacc::functor::Interface<F, 1u, void>
                 * @param mapper functor to map a block to a supercell
                 * @param pb particles species box
                 */
                template<typename T_Acc, typename T_Functor, typename T_Mapping, typename T_ParBox>
                DINLINE void operator()(T_Acc const& acc, T_Functor functor, T_Mapping const mapper, T_ParBox pb) const
                {
                    using SuperCellSize = typename T_ParBox::FrameType::SuperCellSize;
                    constexpr uint32_t dim = SuperCellSize::dim;
                    constexpr uint32_t numWorkers = T_numWorkers;

                    uint32_t const workerIdx = cupla::threadIdx(acc).x;

                    DataSpace<dim> const superCellIdx(mapper.getSuperCellIndex(DataSpace<dim>(cupla::blockIdx(acc))));

                    auto forEachParticle = makeForEach<numWorkers>(workerIdx, pb, superCellIdx);

                    // end kernel if we have no particles
                    if(!forEachParticle.hasParticles())
                        return;

                    // offset of the superCell (in cells, without any guards) to the origin of the local
                    // domain
                    DataSpace<dim> const localSuperCellOffset = superCellIdx - mapper.getGuardingSuperCells();

                    auto accFunctor = functor(acc, localSuperCellOffset, lockstep::Worker<numWorkers>{workerIdx});

                    forEachParticle(acc, accFunctor);
                }
            };

        } // namespace detail
    } // namespace acc

    /** Run a unary functor for each particle of a species in the given area
     *
     * Has a version for a fixed area, and for a user-provided mapper factory.
     * They differ only in how the area is defined.
     *
     * @warning Does NOT fill gaps automatically! If the
     *          operation deactivates particles or creates "gaps" in any
     *          other way, CallFillAllGaps needs to be called for the
     *          species manually afterwards!
     *
     * @tparam T_Species type of the species
     * @tparam T_Functor unary particle functor type which follows the interface of
     *                   pmacc::functor::Interface<F, 1u, void>
     *
     * @param species species to operate on
     * @param functor operation which is applied to each particle of the species
     *
     * @{
     */

    /** Version for a custom area mapper factory
     *
     * @tparam T_AreaMapperFactory factory type to construct an area mapper that defines the area to
     * process, adheres to the AreaMapperFactory concept
     *
     * @param areaMapperFactory factory to construct an area mapper,
     *                          the area is defined by the constructed mapper object
     */
    template<typename T_Species, typename T_Functor, typename T_AreaMapperFactory>
    HINLINE void forEach(T_Species&& species, T_Functor functor, T_AreaMapperFactory const& areaMapperFactory)
    {
        using MappingDesc = decltype(species.getCellDescription());
        using SuperCellSize = typename MappingDesc::SuperCellSize;
        constexpr uint32_t numWorkers
            = pmacc::traits::GetNumWorkers<pmacc::math::CT::volume<SuperCellSize>::type::value>::value;

        auto const mapper = areaMapperFactory(species.getCellDescription());
        PMACC_KERNEL(acc::detail::KernelForEachParticle<numWorkers>{})
        (mapper.getGridDim(), numWorkers)(std::move(functor), mapper, species.getDeviceParticlesBox());
    }

    /** Version for a fixed area
     *
     * @tparam T_area area to process particles in
     */
    template<uint32_t T_area, typename T_Species, typename T_Functor>
    HINLINE void forEach(T_Species&& species, T_Functor functor)
    {
        auto const areaMapperFactory = AreaMapperFactory<T_area>{};
        forEach(species, functor, areaMapperFactory);
    }

    /** @} */

} // namespace pmacc::particles::algorithm
