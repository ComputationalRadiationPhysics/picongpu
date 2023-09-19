/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch, Felix Schmitt, Sergei Bastrakov
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/fields/FieldJ.hpp"
#include "picongpu/fields/incidentField/Traits.hpp"
#include "picongpu/plugins/ILightweightPlugin.hpp"
#include "picongpu/plugins/output/GatherSlice.hpp"
#include "picongpu/plugins/output/header/MessageHeader.hpp"
#include "picongpu/simulation/control/MovingWindow.hpp"

#include <pmacc/algorithms/GlobalReduce.hpp>
#include <pmacc/algorithms/math/defines/pi.hpp>
#include <pmacc/assert.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/dimensions/DataSpaceOperations.hpp>
#include <pmacc/kernel/atomic.hpp>
#include <pmacc/lockstep.hpp>
#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/mappings/kernel/MappingDescription.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/boxes/DataBox.hpp>
#include <pmacc/memory/boxes/DataBoxDim1Access.hpp>
#include <pmacc/memory/boxes/PitchedBox.hpp>
#include <pmacc/memory/boxes/SharedBox.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/memory/shared/Allocate.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/particles/algorithm/ForEach.hpp>
#include <pmacc/particles/memory/boxes/ParticlesBox.hpp>

#include <cfloat>
#include <memory>
#include <string>


namespace picongpu
{
    // normalize EM fields to typical laser or plasma quantities
    //-1: Auto:     enable adaptive scaling for each output
    // 1: Laser:    [outdated]
    // 2: Drift:    [outdated]
    // 3: PlWave:   typical fields calculated out of the plasma freq.,
    //              assuming the wave moves approx. with c
    // 4: Thermal:  outdated
    // 5: BlowOut:  [outdated]
    // 6: Custom:   user-provided normalization factors via visPreview::customNormalizationSI
    // 7: Incident: typical fields calculated out of the incident field amplitude,
    //              uses max amplitude from all enabled incident field profile types ignoring Free
    ///  @return float3_X( tyBField, tyEField, tyCurrent )

    template<int T>
    struct typicalFields
    {
        HDINLINE static float3_X get()
        {
            return float3_X(float_X(1.0), float_X(1.0), float_X(1.0));
        }
    };

    template<>
    struct typicalFields<-1>
    {
        HDINLINE static float3_X get()
        {
            return float3_X(float_X(1.0), float_X(1.0), float_X(1.0));
        }
    };

    /* outdated drift normalization */
    template<>
    struct typicalFields<2>;

    template<>
    struct typicalFields<3>
    {
        HDINLINE static float3_X get()
        {
#if !(EM_FIELD_SCALE_CHANNEL1 == 3 || EM_FIELD_SCALE_CHANNEL2 == 3 || EM_FIELD_SCALE_CHANNEL3 == 3)
            return float3_X(float_X(1.0), float_X(1.0), float_X(1.0));
#else
            constexpr auto baseCharge = BASE_CHARGE;
            const float_X lambda_pl = pmacc::math::Pi<float_X>::doubleValue * SPEED_OF_LIGHT
                * sqrt(BASE_MASS * EPS0 / BASE_DENSITY / baseCharge / baseCharge);
            const float_X tyEField = lambda_pl * BASE_DENSITY / 3.0f / EPS0;
            const float_X tyBField = tyEField * MUE0_EPS0;
            const float_X tyCurrent = tyBField / MUE0;

            return float3_X(tyBField, tyEField, tyCurrent);
#endif
        }
    };

    /* outdated ELECTRON_TEMPERATURE normalization */
    template<>
    struct typicalFields<4>;

    //! Specialization for custom normalization
    template<>
    struct typicalFields<6>
    {
        HDINLINE static float3_X get()
        {
#if !(EM_FIELD_SCALE_CHANNEL1 == 6 || EM_FIELD_SCALE_CHANNEL2 == 6 || EM_FIELD_SCALE_CHANNEL3 == 6)
            return float3_X(float_X(1.0), float_X(1.0), float_X(1.0));
#else
            // Convert customNormalizationSI to internal units
            using visPreview::customNormalizationSI;
            constexpr auto normalizationB = static_cast<float_X>(customNormalizationSI[0] / UNIT_BFIELD);
            constexpr auto normalizationE = static_cast<float_X>(customNormalizationSI[1] / UNIT_EFIELD);
            constexpr auto normalizationCurrent
                = static_cast<float_X>(customNormalizationSI[2] / (UNIT_CHARGE / UNIT_TIME));
            return float3_X{normalizationB, normalizationE, normalizationCurrent};
#endif
        }
    };

    //! Specialization for incident field normalization
    template<>
    struct typicalFields<7>
    {
        //! Get normalization values
        HDINLINE static float3_X get()
        {
#if !(EM_FIELD_SCALE_CHANNEL1 == 7 || EM_FIELD_SCALE_CHANNEL2 == 7 || EM_FIELD_SCALE_CHANNEL3 == 7)
            return float3_X::create(1.0_X);
#else
            constexpr auto baseCharge = BASE_CHARGE;
            const float_X tyCurrent = particles::TYPICAL_PARTICLES_PER_CELL
                * static_cast<float_X>(particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE) * math::abs(baseCharge)
                / DELTA_T;
            const float_X tyEField = getAmplitude() + FLT_MIN;
            const float_X tyBField = tyEField * MUE0_EPS0;
            return float3_X(tyBField, tyEField, tyCurrent);
#endif
        }

    private:
        //! Get laser E amplitude in internal units
        HDINLINE static float_X getAmplitude()
        {
            using Profiles = fields::incidentField::UniqueEnabledProfiles;
            meta::ForEach<Profiles, CalculateMaxAmplitude<boost::mpl::_1>> calculateMaxAmplitude;
            auto maxAmplitude = 0.0_X;
            calculateMaxAmplitude(maxAmplitude);
            return maxAmplitude;
        }

        /** Functor to calculate max amplitude between the given value and given profile
         *
         * @tparam T_Profile incident field profile
         */
        template<typename T_Profile>
        struct CalculateMaxAmplitude
        {
            /** Call update E with the given parameters
             *
             * @param[out] maxAmplitude current value of max amplitude, can be updated by the functor
             */
            HDINLINE void operator()(float_X& maxAmplitude) const
            {
                auto const amplitude = fields::incidentField::amplitude<T_Profile>;
                if(amplitude > maxAmplitude)
                    maxAmplitude = amplitude;
            }
        };
    };

    /** Check if an offset is part of the slicing domain
     *
     * Check if a N dimensional local domain offset is equal to a scalar offset of
     * a given dimension.
     * The results can be taken to decide if a cell is within a slice of a volume.
     */
    template<uint32_t T_dim = simDim>
    struct IsPartOfSlice;

    template<>
    struct IsPartOfSlice<DIM3>
    {
        /** perform check
         *
         * @param cellOffset cell offset relative to the origin of the local domain
         * @param sliceDim dimension of the slice
         * @param localDomainOffset local domain offset relative to the origin of the global domain
         *                          (in the slice dimension)
         * @param sliceOffset cell offset of the slice relative to the origin of the global domain
         *                         ( in the slice dimension)
         * @return true if cellOffset is part of the slicing domain, else false
         *
         * @return always true
         */
        template<typename T_Space>
        HDINLINE bool operator()(
            T_Space const& cellOffset,
            uint32_t const sliceDim,
            uint32_t const localDomainOffset,
            uint32_t const sliceOffset)
        {
            // offset of the cell relative to the global origin
            uint32_t const localCellOffset = cellOffset[sliceDim] + localDomainOffset;
            return localCellOffset == sliceOffset;
        }
    };

    template<>
    struct IsPartOfSlice<DIM2>
    {
        /** perform check
         *
         * @return always true
         */
        template<typename T_Space>
        HDINLINE bool operator()(T_Space const&, uint32_t const, uint32_t const, uint32_t const)
        {
            return true;
        }
    };

    //! derives two dimensional field from a slice of field
    struct KernelPaintFields
    {
        /** derive field values
         *
         * @tparam T_EBox pmacc::DataBox, electric field box type
         * @tparam T_BBox pmacc::DataBox, magnetic field box type
         * @tparam T_JBox particle current box type
         * @tparam T_Mapping mapper functor type
         * @tparam T_Worker lockstep worker type
         *
         * @param worker lockstep worker
         * @param fieldE electric field
         * @param fieldB magnetic field
         * @param fieldJ field with particle current
         * @param image[in,out] two dimensional image (without guarding cells)
         * @param transpose indices to transpose dimensions range per dimension [0,simDim)
         * @param slice offset (in cells) of the slice in the dimension sliceDim relative to
         *              the origin of the global domain
         * @param localDomainOffset offset (in cells) of the local domain relative to the
         *                          origin of the global domain
         * @param sliceDim dimension to slice range [0,simDim)
         * @param mapper functor to map a block to a supercell
         */
        template<typename T_EBox, typename T_BBox, typename T_JBox, typename T_Mapping, typename T_Worker>
        DINLINE void operator()(
            T_Worker const& worker,
            T_EBox const fieldE,
            T_BBox const fieldB,
            T_JBox const fieldJ,
            DataBox<PitchedBox<float3_X, DIM2>> image,
            DataSpace<DIM2> const transpose,
            int const slice,
            uint32_t const localDomainOffset,
            uint32_t const sliceDim,
            T_Mapping mapper) const
        {
            using SuperCellSize = typename T_Mapping::SuperCellSize;

            constexpr uint32_t cellsPerSupercell = pmacc::math::CT::volume<SuperCellSize>::type::value;

            DataSpace<simDim> const suplercellIdx
                = mapper.getSuperCellIndex(DataSpace<simDim>(cupla::blockIdx(worker.getAcc())));
            // offset of the supercell (in cells) to the origin of the local domain
            DataSpace<simDim> const supercellCellOffset(
                (suplercellIdx - mapper.getGuardingSuperCells()) * SuperCellSize::toRT());

            // each cell in a supercell is handled as a virtual worker
            auto forEachCell = lockstep::makeForEach<cellsPerSupercell>(worker);

            forEachCell(
                [&](uint32_t const linearIdx)
                {
                    // cell index within the superCell
                    DataSpace<simDim> const cellIdx
                        = DataSpaceOperations<simDim>::template map<SuperCellSize>(linearIdx);
                    // offset to the origin of the local domain + guarding cells
                    DataSpace<simDim> const cellOffset(suplercellIdx * SuperCellSize::toRT() + cellIdx);
                    // cell offset without guarding cells
                    DataSpace<simDim> const realCell(supercellCellOffset + cellIdx);
                    // offset within the two dimensional result buffer
                    DataSpace<DIM2> const imageCell(realCell[transpose.x()], realCell[transpose.y()]);

                    bool const isCellOnSlice = IsPartOfSlice<>{}(realCell, sliceDim, localDomainOffset, slice);

                    /* if the virtual worker is not calculating a cell out of the
                     * selected slice then exit
                     */
                    if(!isCellOnSlice)
                        return;

                    // set fields of this cell to vars
                    typename T_BBox::ValueType field_b = fieldB(cellOffset);
                    typename T_EBox::ValueType field_e = fieldE(cellOffset);
                    typename T_JBox::ValueType field_j = fieldJ(cellOffset);

                    // multiply with the area size of each plane to get current
                    auto field_current = field_j * float3_X::create(CELL_VOLUME) / cellSize;

                    /* reset picture to black
                     *   color range for each RGB channel: [0.0, 1.0]
                     */
                    float3_X pic(
                        /* typical values of the fields to normalize them to [0,1]
                         * typicalFields<>::get()[...] means: [0] = BField normalization,
                         * [1] = EField normalization, [2] = Current normalization
                         */
                        visPreview::preChannel1(
                            field_b / typicalFields<EM_FIELD_SCALE_CHANNEL1>::get()[0],
                            field_e / typicalFields<EM_FIELD_SCALE_CHANNEL1>::get()[1],
                            field_current / typicalFields<EM_FIELD_SCALE_CHANNEL1>::get()[2]),
                        visPreview::preChannel2(
                            field_b / typicalFields<EM_FIELD_SCALE_CHANNEL2>::get()[0],
                            field_e / typicalFields<EM_FIELD_SCALE_CHANNEL2>::get()[1],
                            field_current / typicalFields<EM_FIELD_SCALE_CHANNEL2>::get()[2]),
                        visPreview::preChannel3(
                            field_b / typicalFields<EM_FIELD_SCALE_CHANNEL3>::get()[0],
                            field_e / typicalFields<EM_FIELD_SCALE_CHANNEL3>::get()[1],
                            field_current / typicalFields<EM_FIELD_SCALE_CHANNEL3>::get()[2]));

                    // draw to (perhaps smaller) image cell
                    image(imageCell) = pic;
                });
        }
    };

    //! derives two dimensional field from a particle slice
    struct KernelPaintParticles3D
    {
        /** derive particle values
         *
         * @tparam T_ParBox pmacc::ParticlesBox, particle box type
         * @tparam T_Mapping mapper functor type
         * @tparam T_Worker lockstep worker type
         *
         * @param acc alpaka accelerator
         * @param pb particle memory
         * @param image[in,out] two dimensional image (without guarding cells)
         * @param transpose indices to transpose dimensions range per dimension [0,simDim)
         * @param slice offset (in cells) of the slice in the dimension sliceDim relative to
         *              the origin of the global domain
         * @param localDomainOffset offset (in cells) of the local domain relative to the
         *                          origin of the global domain
         * @param sliceDim dimension to slice range [0,simDim)
         * @param mapper functor to map a block to a supercell
         */
        template<typename T_ParBox, typename T_Mapping, typename T_Worker>
        DINLINE void operator()(
            T_Worker const& worker,
            T_ParBox pb,
            DataBox<PitchedBox<float3_X, DIM2>> image,
            DataSpace<DIM2> const transpose,
            int const slice,
            uint32_t const localDomainOffset,
            uint32_t const sliceDim,
            T_Mapping mapper) const
        {
            using SuperCellSize = typename T_Mapping::SuperCellSize;

            constexpr uint32_t numCellsPerSupercell = pmacc::math::CT::volume<SuperCellSize>::type::value;

            auto onlyMaster = lockstep::makeMaster(worker);

            // each virtual worker works on a cell in the supercell
            auto forEachCell = lockstep::makeForEach<numCellsPerSupercell>(worker);

            /* is 1 if a offset of a cell in the supercell is equal the slice (offset)
             * else 0
             */
            PMACC_SMEM(worker, superCellParticipate, int);

            /* true if the virtual worker is processing a pixel within the resulting image,
             * else false
             */
            auto isImageThreadCtx = lockstep::makeVar<bool>(forEachCell, false);

            DataSpace<simDim> const suplercellIdx
                = mapper.getSuperCellIndex(DataSpace<simDim>(cupla::blockIdx(worker.getAcc())));
            // offset of the supercell (in cells) to the origin of the local domain
            DataSpace<simDim> const supercellCellOffset(
                (suplercellIdx - mapper.getGuardingSuperCells()) * SuperCellSize::toRT());

            onlyMaster([&]() { superCellParticipate = 0; });

            worker.sync();

            forEachCell(
                [&](uint32_t const idx, bool& isImageThread)
                {
                    // cell index within the superCell
                    DataSpace<simDim> const cellIdx = DataSpaceOperations<simDim>::template map<SuperCellSize>(idx);

                    // cell offset to origin of the local domain
                    DataSpace<simDim> const realCell(supercellCellOffset + cellIdx);

                    bool const isCellOnSlice = IsPartOfSlice<>{}(realCell, sliceDim, localDomainOffset, slice);

                    if(isCellOnSlice)
                    {
                        // atomic avoids: WAW Error in cuda-memcheck racecheck
                        kernel::atomicAllExch(worker, &superCellParticipate, 1, ::alpaka::hierarchy::Threads{});
                        isImageThread = true;
                    }
                },
                isImageThreadCtx);

            worker.sync();

            if(superCellParticipate == 0)
                return;

            // slice is always two dimensional
            using SharedMem = DataBox<PitchedBox<float_X, DIM2>>;

            float_X* shBlock = ::alpaka::getDynSharedMem<float_X>(worker.getAcc());

            // shared memory box for particle counter
            SharedMem counter(PitchedBox<float_X, DIM2>(
                (float_X*) shBlock,
                DataSpace<DIM2>(),
                // pitch in byte
                SuperCellSize::toRT()[transpose.x()] * sizeof(float_X)));

            forEachCell(
                [&](uint32_t const idx, bool const isImageThread)
                {
                    /* cell index within the superCell */
                    DataSpace<simDim> const cellIdx = DataSpaceOperations<simDim>::template map<SuperCellSize>(idx);

                    DataSpace<DIM2> const localCell(cellIdx[transpose.x()], cellIdx[transpose.y()]);

                    if(isImageThread)
                    {
                        counter(localCell) = float_X(0.0);
                    }
                },
                isImageThreadCtx);

            // wait that shared memory  is set to zero
            worker.sync();

            auto forEachParticle = pmacc::particles::algorithm::acc::makeForEach(worker, pb, suplercellIdx);

            forEachParticle(
                [&supercellCellOffset, &counter, &transpose, sliceDim, slice, localDomainOffset](
                    auto const& lockstepWorker,
                    auto& particle)
                {
                    int const linearCellIdx = particle[localCellIdx_];
                    // we only draw the first slice of cells in the super cell (z == 0)
                    DataSpace<simDim> const particleCellOffset(
                        DataSpaceOperations<simDim>::template map<SuperCellSize>(linearCellIdx));
                    bool const isParticleOnSlice = IsPartOfSlice<>{}(
                        particleCellOffset + supercellCellOffset,
                        sliceDim,
                        localDomainOffset,
                        slice);
                    if(isParticleOnSlice)
                    {
                        DataSpace<DIM2> const reducedCell(
                            particleCellOffset[transpose.x()],
                            particleCellOffset[transpose.y()]);
                        cupla::atomicAdd(
                            lockstepWorker.getAcc(),
                            &(counter(reducedCell)),
                            // normalize the value to avoid bad precision for large macro particle weightings
                            particle[weighting_]
                                / static_cast<float_X>(particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE),
                            ::alpaka::hierarchy::Threads{});
                    }
                });

            // wait that all worker finsihed the reduce operation
            worker.sync();

            forEachCell(
                [&](uint32_t const idx, bool const isImageThread)
                {
                    if(isImageThread)
                    {
                        // cell index within the superCell
                        DataSpace<simDim> const cellIdx
                            = DataSpaceOperations<simDim>::template map<SuperCellSize>(idx);
                        // cell offset to origin of the local domain
                        DataSpace<simDim> const realCell(supercellCellOffset + cellIdx);
                        // index in image
                        DataSpace<DIM2> const imageCell(realCell[transpose.x()], realCell[transpose.y()]);

                        DataSpace<DIM2> const localCell(cellIdx[transpose.x()], cellIdx[transpose.y()]);

                        /** Note: normally, we would multiply by particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE
                         * again. BUT: since we are interested in a simple value between 0 and 1, we stay with this
                         * number (normalized to the order of macro particles) and devide by the number of typical
                         * macro particles per cell
                         */
                        float_X value = counter(localCell) / float_X(particles::TYPICAL_PARTICLES_PER_CELL);
                        if(value > 1.0)
                            value = 1.0;


                        visPreview::preParticleDensCol::addRGB(
                            image(imageCell),
                            value,
                            visPreview::preParticleDens_opacity);

                        // cut to [0, 1]
                        for(uint32_t d = 0; d < DIM3; ++d)
                        {
                            if(image(imageCell)[d] < float_X(0.0))
                                image(imageCell)[d] = float_X(0.0);
                            if(image(imageCell)[d] > float_X(1.0))
                                image(imageCell)[d] = float_X(1.0);
                        }
                    }
                },
                isImageThreadCtx);
        }
    };

    namespace vis_kernels
    {
        /** divide each cell by a value
         *
         * @tparam T_blockSize number of elements which will be handled
         *                     within a kernel block
         */
        template<uint32_t T_blockSize>
        struct DivideAnyCell
        {
            /** derive particle values
             *
             * @tparam T_Mem pmacc::DataBox, type of the on dimensional memory
             * @tparam T_Type divisor type
             * @tparam T_Worker lockstep worker type
             *
             * @param acc alpaka accelerator
             * @param mem memory[in,out] to manipulate, must provide the `operator[](int)`
             * @param n number of elements in mem
             * @param divisor divisor for the division
             */
            template<typename T_Mem, typename T_Type, typename T_Worker>
            DINLINE void operator()(T_Worker const& worker, T_Mem mem, uint32_t n, T_Type divisor) const
            {
                // each virtual worker works on a cell
                auto forEachCell = lockstep::makeForEach<T_blockSize>(worker);

                forEachCell(
                    [&](uint32_t const linearIdx)
                    {
                        uint32_t tid = cupla::blockIdx(worker.getAcc()).x * T_blockSize + linearIdx;
                        if(tid >= n)
                            return;

                        float3_X const FLT3_MIN = float3_X::create(FLT_MIN);
                        mem[tid] /= (divisor + FLT3_MIN);
                    });
            }
        };


        /** convert channel value to an RGB color
         *
         * @tparam T_blockSize number of elements which will be handled
         *                     within a kernel block
         */
        template<uint32_t T_blockSize>
        struct ChannelsToRGB
        {
            /** convert each element to an RGB color
             *
             * @tparam T_Mem pmacc::DataBox, type of the on dimensional memory
             * @tparam T_Worker lockstep worker type
             *
             * @param acc alpaka accelerator
             * @param mem memory[in,out] to manipulate, must provide the `operator[](int)`
             * @param n number of elements in mem
             */
            template<typename T_Mem, typename T_Worker>
            DINLINE void operator()(T_Worker const& worker, T_Mem mem, uint32_t n) const
            {
                // each virtual worker works on a cell
                auto forEachCell = lockstep::makeForEach<T_blockSize>(worker);

                forEachCell(
                    [&](uint32_t const linearIdx)
                    {
                        uint32_t const tid = cupla::blockIdx(worker.getAcc()).x * T_blockSize + linearIdx;
                        if(tid >= n)
                            return;

                        float3_X rgb(float3_X::create(0.0));

                        visPreview::preChannel1Col::addRGB(rgb, mem[tid].x(), visPreview::preChannel1_opacity);
                        visPreview::preChannel2Col::addRGB(rgb, mem[tid].y(), visPreview::preChannel2_opacity);
                        visPreview::preChannel3Col::addRGB(rgb, mem[tid].z(), visPreview::preChannel3_opacity);
                        mem[tid] = rgb;
                    });
            }
        };

    } // namespace vis_kernels

    /**
     * Visualizes simulation data by writing png files.
     * Visulization is performed in an additional thread.
     */
    template<class ParticlesType, class Output>
    class Visualisation : public ILightweightPlugin
    {
    private:
        using SuperCellSize = MappingDesc::SuperCellSize;


    public:
        using FrameType = typename ParticlesType::FrameType;
        using CreatorType = Output;

        Visualisation(
            std::string name,
            Output output,
            std::string notifyPeriod,
            DataSpace<DIM2> transpose,
            float_X slicePoint)
            : particleTag(ParticlesType::FrameType::getName())
            , m_notifyPeriod(notifyPeriod)
            , m_slicePoint(slicePoint)
            , pluginName(name)
            , m_transpose(transpose)
            , header(nullptr)
            , m_output(output)
            , isMaster(false)
            , reduce(1024)
        {
            sliceDim = 0;
            if(m_transpose.x() == 0 || m_transpose.y() == 0)
                sliceDim = 1;
            /* sliceDim can not be two if the simulation is 2D.
             * sliceDim is only required for a 3D simulation
             */
            if constexpr(simDim == DIM3)
                if((m_transpose.x() == 1 || m_transpose.y() == 1) && sliceDim == 1)
                    sliceDim = 2;

            Environment<>::get().PluginConnector().registerPlugin(this);
            Environment<>::get().PluginConnector().setNotificationPeriod(this, m_notifyPeriod);
        }

        ~Visualisation() override
        {
            /* wait that shared buffers can destroyed */
            m_output.join();
            if(!m_notifyPeriod.empty())
            {
                MessageHeader::destroy(header);
            }
        }

        std::string pluginGetName() const override
        {
            return "Visualisation";
        }

        void notify(uint32_t currentStep) override
        {
            PMACC_ASSERT(cellDescription != nullptr);
            const DataSpace<simDim> localSize(cellDescription->getGridLayout().getDataSpaceWithoutGuarding());
            Window window(MovingWindow::getInstance().getWindow(currentStep));

            /*sliceOffset is only used in 3D*/
            sliceOffset = (int) ((float_32) (window.globalDimensions.size[sliceDim]) * m_slicePoint)
                + window.globalDimensions.offset[sliceDim];

            if(!doDrawing())
            {
                return;
            }
            createImage(currentStep, window);
        }

        void setMappingDescription(MappingDesc* cellDescription) override
        {
            PMACC_ASSERT(cellDescription != nullptr);
            this->cellDescription = cellDescription;
        }

        void createImage(uint32_t currentStep, Window window)
        {
            DataConnector& dc = Environment<>::get().DataConnector();
            // Data does not need to be synchronized as visualization is
            // done at the device.
            auto fieldB = dc.get<FieldB>(FieldB::getName());
            auto fieldE = dc.get<FieldE>(FieldE::getName());
            auto fieldJ = dc.get<FieldJ>(FieldJ::getName());
            auto particles = dc.get<ParticlesType>(particleTag);

            /* wait that shared buffers can accessed without conflicts */
            m_output.join();

            uint32_t localDomainOffset = 0;
            if constexpr(simDim == DIM3)
                localDomainOffset = Environment<simDim>::get().SubGrid().getLocalDomain().offset[sliceDim];

            PMACC_ASSERT(cellDescription != nullptr);

            auto const mapper = makeAreaMapper<CORE + BORDER>(*cellDescription);

            auto workerCfg = lockstep::makeWorkerCfg(SuperCellSize{});

            // create image fields
            PMACC_LOCKSTEP_KERNEL(KernelPaintFields{}, workerCfg)
            (mapper.getGridDim())(
                fieldE->getDeviceDataBox(),
                fieldB->getDeviceDataBox(),
                fieldJ->getDeviceDataBox(),
                img->getDeviceBuffer().getDataBox(),
                m_transpose,
                sliceOffset,
                localDomainOffset,
                sliceDim,
                mapper);

            // find maximum for img.x()/y and z and return it as float3_X
            int elements = img->getGridLayout().getDataSpace().productOfComponents();

            // Add one dimension access to 2d DataBox
            using D1Box = DataBoxDim1Access<typename GridBuffer<float3_X, 2U>::DataBoxType>;
            D1Box d1access(img->getDeviceBuffer().getDataBox(), img->getGridLayout().getDataSpace());

            constexpr uint32_t cellsPerSupercell = pmacc::math::CT::volume<SuperCellSize>::type::value;

#if(EM_FIELD_SCALE_CHANNEL1 == -1 || EM_FIELD_SCALE_CHANNEL2 == -1 || EM_FIELD_SCALE_CHANNEL3 == -1)
            // reduce with functor max
            float3_X max = reduce(pmacc::math::operation::Max(), d1access, elements);
            // reduce with functor min
            // float3_X min = reduce(pmacc::math::operation::Min(),
            //                    d1access,
            //                    elements);
#    if(EM_FIELD_SCALE_CHANNEL1 != -1)
            max.x() = float_X(1.0);
#    endif
#    if(EM_FIELD_SCALE_CHANNEL2 != -1)
            max.y() = float_X(1.0);
#    endif
#    if(EM_FIELD_SCALE_CHANNEL3 != -1)
            max.z() = float_X(1.0);
#    endif

            /* We don't know the size of the supercell plane at compile time
             * (because of the runtime dimension selection in any plugin),
             * thus we must use a one dimension kernel and no mapper
             */
            PMACC_LOCKSTEP_KERNEL(vis_kernels::DivideAnyCell<cellsPerSupercell>{}, workerCfg)
            ((elements + cellsPerSupercell - 1u) / cellsPerSupercell)(d1access, elements, max);
#endif

            // convert channels to RGB
            PMACC_LOCKSTEP_KERNEL(vis_kernels::ChannelsToRGB<cellsPerSupercell>{}, workerCfg)
            ((elements + cellsPerSupercell - 1u) / cellsPerSupercell)(d1access, elements);

            // add density color channel
            DataSpace<simDim> blockSize(MappingDesc::SuperCellSize::toRT());
            DataSpace<DIM2> blockSize2D(blockSize[m_transpose.x()], blockSize[m_transpose.y()]);

            auto particleWorkerCfg = lockstep::makeWorkerCfg<ParticlesType::FrameType::frameSize>();
            // create image particles
            PMACC_LOCKSTEP_KERNEL(KernelPaintParticles3D{}, particleWorkerCfg)
            (mapper.getGridDim(), blockSize2D.productOfComponents() * sizeof(float_X))(
                particles->getDeviceParticlesBox(),
                img->getDeviceBuffer().getDataBox(),
                m_transpose,
                sliceOffset,
                localDomainOffset,
                sliceDim,
                mapper);

            // send the RGB image back to host
            img->deviceToHost();


            header->update(*cellDescription, window, m_transpose, currentStep);


            eventSystem::getTransactionEvent().waitForFinished(); // wait for copy picture

            DataSpace<DIM2> size = img->getGridLayout().getDataSpace();

            auto hostBox = img->getHostBuffer().getDataBox();

            if(picongpu::white_box_per_GPU)
            {
                hostBox({0, 0}) = float3_X(1.0, 1.0, 1.0);
                hostBox({0, size.y() - 1}) = float3_X(1.0, 1.0, 1.0);
                hostBox({size.x() - 1, 0}) = float3_X(1.0, 1.0, 1.0);
                hostBox({size.x() - 1, size.y() - 1}) = float3_X(1.0, 1.0, 1.0);
            }
            auto resultBox = gather(hostBox, *header);
            if(isMaster)
            {
                m_output(resultBox.shift(header->window.offset), header->window.size, *header);
            }
        }

        void init()
        {
            if(!m_notifyPeriod.empty())
            {
                PMACC_ASSERT(cellDescription != nullptr);
                const DataSpace<simDim> localSize(cellDescription->getGridLayout().getDataSpaceWithoutGuarding());

                Window window(MovingWindow::getInstance().getWindow(0));
                sliceOffset = (int) ((float_32) (window.globalDimensions.size[sliceDim]) * m_slicePoint)
                    + window.globalDimensions.offset[sliceDim];


                const DataSpace<simDim> gpus = Environment<simDim>::get().GridController().getGpuNodes();

                float_32 cellSizeArr[3] = {0, 0, 0};
                for(uint32_t i = 0; i < simDim; ++i)
                    cellSizeArr[i] = cellSize[i];

                header = MessageHeader::create();
                header->update(*cellDescription, window, m_transpose, 0, cellSizeArr, gpus);

                bool isDrawing = doDrawing();
                isMaster = gather.init(isDrawing);
                reduce.participate(isDrawing);

                /* create memory for the local picture if the gpu participate on the visualization */
                if(isDrawing)
                    img = std::make_unique<GridBuffer<float3_X, DIM2>>(header->node.maxSize);
            }
        }

        void pluginRegisterHelp(po::options_description& desc) override
        {
            // nothing to do here
        }

    private:
        bool doDrawing()
        {
            PMACC_ASSERT(cellDescription != nullptr);
            const DataSpace<simDim> globalRootCellPos(Environment<simDim>::get().SubGrid().getLocalDomain().offset);
            if constexpr(simDim == DIM3)
            {
                const bool tmp = globalRootCellPos[sliceDim]
                            + Environment<simDim>::get().SubGrid().getLocalDomain().size[sliceDim]
                        > sliceOffset
                    && globalRootCellPos[sliceDim] <= sliceOffset;
                return tmp;
            }
            return true;
        }


        MappingDesc* cellDescription;
        SimulationDataId particleTag;

        std::unique_ptr<GridBuffer<float3_X, DIM2>> img;

        int sliceOffset;
        std::string m_notifyPeriod;
        float_X m_slicePoint;

        std::string pluginName;


        DataSpace<DIM2> m_transpose;
        //! dimension to slice range [0,simDim)
        uint32_t sliceDim;

        MessageHeader* header;

        Output m_output;
        GatherSlice gather;
        bool isMaster;
        algorithms::GlobalReduce reduce;
    };


} // namespace picongpu
