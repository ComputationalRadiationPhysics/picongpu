/* Copyright 2015-2021 Alexander Grund, Pawel Ordyna
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

#include "picongpu/plugins/externalBeam/AxisSwap.hpp"
#include "picongpu/plugins/photonDetector/DetectorParams.hpp"
#include "picongpu/plugins/photonDetector/accumulation/accumulationPolicies.hpp"

#include <pmacc/algorithms/math.hpp>
#include <pmacc/dataManagement/ISimulationData.hpp>
#include <pmacc/debug/VerboseLog.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>


namespace picongpu
{
    namespace plugins
    {
        namespace photonDetector
        {
            namespace detail
            {
                //! Round a floating point value towards zero
                template<typename T>
                HDINLINE int roundTowardsZero(T const& val)
                {
                    int ceilVal = pmacc::math::float2int_rd(math::abs(val));
                    return val < 0 ? -1 * ceilVal : ceilVal;
                }

                /**
                 * Calculates the cell index on the detector that a photon reaches when it continues
                 * with the current speed
                 *
                 * @tparam T_Config detector configuration from photonDetector.param
                 */
                template<typename T_Config>
                struct GetTargetCellIdx
                {
                    using Config = T_Config;
                    static constexpr float_X cellWidth = float_X(Config::cellWidth / UNIT_LENGTH);
                    static constexpr float_X cellHeight = float_X(Config::cellHeight / UNIT_LENGTH);

                    /** GetTargetCellIdx constructor
                     *
                     * @param detectorParams detector parameters (see DetectorParams.def)
                     * @param simSize simulation size (shape) in cells (in the detector coordinate system)
                     * @param axisSwap used to switch between the PIConGPU and the detector coordinate system
                     */
                    HDINLINE GetTargetCellIdx(
                        DetectorParams const& detectorParams,
                        DataSpace<simDim> const& simSize,
                        externalBeam::AxisSwap const& axisSwap)
                        : detector_m(detectorParams)
                        , simSize_m(simSize)
                        , axisSwap_m(axisSwap)
                    {
                    }

                    /**
                     * Calculates the index on the detector and the time of impact
                     * Returns true, if the detector was hit, false otherwise.
                     * On false \ref targetIdx and \ref dt are undefined
                     *
                     * @param particle particle to detect
                     * @param globalIdx the global index of the particle cell (in the detector coordinate system)
                     * @param targetIdx a reference to a variable storing the result (index on the detector)
                     */
                    template<typename T_Acc, typename T_Particle>
                    DINLINE bool operator()(
                        T_Acc const& acc,
                        T_Particle const& particle,
                        DataSpace<DIM3> const& globalIdx,
                        DataSpace<DIM2>& targetIdx) const
                    {
                        /* Detector coordinate system viewed from the simulation:
                         *      __________________
                         *     |        ^         |
                         *     |       x|         |
                         *     |      z(x) --->   |
                         *     |            y     |
                         *     |__________________|
                         *
                         */

                        // Get the dimensions of a picongpu cell in the detector coordinate system (only axes parallel
                        // to the detector plane)
                        float_X const CELL_LENGTH_DET_X_DIR
                            = std::get<0>(getCellLengths(detector_m.detectorPlacement));
                        float_X const CELL_LENGTH_DET_Y_DIR
                            = std::get<1>(getCellLengths(detector_m.detectorPlacement));
                        // get the direction in which the particle is moving fro its momentum
                        float3_X momentum = particle[momentum_];
                        float3_X dirPicSystem(momentum / math::abs(momentum));
                        // switch to the detector coordinate system
                        float3_X dir = axisSwap_m.rotate(dirPicSystem);

                        // Not flying towards detector? -> exit
                        if(dir.z() <= 0)
                            return false;

                        /* Notice:
                         * `parataxis` was using `float2int_rd` so it was rounding down.
                         * I think we should be rounding towards zero instead but maybe I'm wrong.
                         */

                        // Calculate angle in "up" dimension (when viewed from front) -> X-dimension of detector
                        // Notice: the z axis is the axis perpendicular to the detector
                        float_X angleBack = math::atan2<trigo_X>(dir.x(), dir.z());
                        // Calculate cell index on detector by angle (histogram-like binning)
                        float_X cellIdxX = angleBack / detector_m.anglePerCell.x();
                        targetIdx.x() = roundTowardsZero(cellIdxX);
                        // The angles are taken from the center of the volume. If the volume is larger than 1 detector
                        // cell particles from the outside of the volume might hit another detector cell. So add this
                        // additional distance from the volume center Note: Offset is really small compared to index.
                        // So adding directly might produce slightly inaccurate results as e.g. 512(half size) + 1e-5(1
                        // simcell offset) is still 512 for float32 Hence we strip off the large (integer) part, use
                        // the remainder and then put integers together again (see Kahan summation)
                        cellIdxX -= targetIdx.x();
                        cellIdxX += (globalIdx.x() - simSize_m.x() / 2) * CELL_LENGTH_DET_X_DIR / cellHeight;
                        targetIdx.x() += roundTowardsZero(cellIdxX);

                        // Same for "right" dimension -> Y-dimension of detector
                        float_X angleDown = math::atan2<trigo_X>(dir.y(), dir.z());
                        float_X cellIdxY = angleDown / detector_m.anglePerCell.y();
                        targetIdx.y() = roundTowardsZero(cellIdxY);
                        cellIdxY -= targetIdx.y();
                        cellIdxY += (globalIdx.y() - simSize_m.y() / 2) * CELL_LENGTH_DET_Y_DIR / cellWidth;
                        targetIdx.y() += roundTowardsZero(cellIdxY);

                        // Shift the origin to the center of the detector
                        targetIdx += detector_m.size / 2;

                        // Check bounds
                        return targetIdx.x() >= 0 && targetIdx.x() < detector_m.size.x() && targetIdx.y() >= 0
                            && targetIdx.y() < detector_m.size.y();
                    }

                private:
                    PMACC_ALIGN(detector_m, const DetectorParams);
                    PMACC_ALIGN(simSize_m, const DataSpace<simDim>);
                    PMACC_ALIGN(axisSwap_m, const externalBeam::AxisSwap);
                };

                //! Unary particle functor for particle detection.
                template<typename T_GetTargetCellIdx, typename T_AccumFunctor>
                struct DetectParticle
                {
                public:
                    /** DetectParticle constructor
                     *
                     * @param getTargetCellIdx unary particle functor used to determine the detector cell
                     * @param accumFunctor unary particle functor used for detector value accumulation
                     * @param axisSwap used to switch between the PIConGPU and the detector coordinate system
                     */
                    HDINLINE DetectParticle(
                        T_GetTargetCellIdx const& getTargetCellIdx,
                        T_AccumFunctor const& accumFunctor,
                        externalBeam::AxisSwap const& axisSwap)
                        : getTargetCellIdx_m(getTargetCellIdx)
                        , accumFunctor_m(accumFunctor)
                        , axisSwap_m(axisSwap)
                    {
                    }

                    /** detect particle
                     *
                     * Check if/where the particle lands on the detector if so call the accumulation functor.
                     *
                     * @param acc alpaka accelerator
                     * @param particle particle to detect
                     * @param superCellPosition the position of the super cell with the particle in the global domain
                     * @param detectorBox detector buffer data box
                     */
                    template<typename T_Acc, typename T_Particle, typename T_DetectorBox>
                    DINLINE void operator()(
                        T_Acc const& acc,
                        T_Particle const& particle,
                        DataSpace<simDim> const& superCellPosition,
                        T_DetectorBox& detectorBox) const
                    {
                        // calculate global cell index
                        DataSpace<simDim> const localCell(
                            pmacc::DataSpaceOperations<simDim>::map<SuperCellSize>(particle[localCellIdx_]));
                        DataSpace<simDim> const globalCellIdx = superCellPosition + localCell;
                        // transform into the detector coordinate system
                        DataSpace<DIM3> const globalCellIdxDet = axisSwap_m.transformCellIdx(globalCellIdx);
                        // particle position on the detector, getTargetCellIdx_m writes here
                        DataSpace<DIM2> targetIdx;
                        // Get index on detector, if none found -> go out
                        if(getTargetCellIdx_m(acc, particle, globalCellIdxDet, targetIdx))
                            accumFunctor_m(acc, detectorBox, targetIdx, particle, globalCellIdxDet);
                    }

                private:
                    PMACC_ALIGN(getTargetCellIdx_m, const T_GetTargetCellIdx);
                    PMACC_ALIGN(accumFunctor_m, const T_AccumFunctor);
                    PMACC_ALIGN(axisSwap_m, const externalBeam::AxisSwap);
                };

            } // namespace detail

            /**
             * Planar detector for particles that will accumulate incoming photons with the given policy
             * @tparam T_Config:
             *      policy AccumulationPolicy Defines how the detected particles change a value at a detector cell.
             *          This type should define a meta function `apply` which takes a species type as
             *          a template parameter and returns the specialized type of a host side accumulation functor
             *          (device functor factory). See ./accumulation for examples.
             *      float_64 distance distance from the volume in meters
             *      float_64 cellWidth detector cell width
             *      float_64 cellHeight detector cell height
             *
             */
            template<typename T_Config, typename T_Species>
            struct PhotonDetectorImpl
            {
                using Config = T_Config;
                using Species = T_Species;
                using AccumPolicy = typename Config::AccumulationPolicy::template apply<Species>::type;
                AccumPolicy accumHostFunctor;


                //! distance of the detector from the simulation box
                static constexpr float_64 distance = float_64(Config::distance / UNIT_LENGTH);
                // detector cell dimensions
                static constexpr float_64 cellWidth = float_64(Config::cellWidth / UNIT_LENGTH);
                static constexpr float_64 cellHeight = float_64(Config::cellHeight / UNIT_LENGTH);
                //! type used to store detector cell values
                using Type = typename AccumPolicy::Type;

            private:
                using Buffer = pmacc::HostDeviceBuffer<Type, 2>;
                //! A memory Buffer on both device and host used to store detector data
                std::unique_ptr<Buffer> buffer;
                //! angle range covered by 1 cell
                pmacc::math::Vector<float_64, 2> anglePerCell_m;
                DetectorPlacement const detectorPlacement_m;

            public:
                //! type of the particle functor used to detect particles
                using DetectParticle
                    = detail::DetectParticle<detail::GetTargetCellIdx<Config>, typename AccumPolicy::AccFunctorType>;

                /** detector implementation constructor
                 *
                 * @param size the detector size (shape)
                 * @param detectorPlacement specifies the side of the simulation box against which the detector is
                 *  placed
                 */
                HINLINE PhotonDetectorImpl(DataSpace<DIM2> const& size, DetectorPlacement const detectorPlacement)
                    : detectorPlacement_m(detectorPlacement)
                    , accumHostFunctor(AccumPolicy())
                {
                    buffer = std::make_unique<Buffer>(DataSpace<DIM2>(size));
                    // initialize the device buffer with the initial value defined by the accumulation policy
                    // (usually 0)
                    buffer->getDeviceBuffer().setValue(AccumPolicy::initValue);
                    anglePerCell_m.x() = atan(cellWidth / distance);
                    anglePerCell_m.y() = atan(cellHeight / distance);
                }

                /** clear detector
                 *
                 * Resets all detector cells with the initial value defined by the accumulation policy (usually 0).
                 */
                HINLINE void resetDeviceBuffer()
                {
                    buffer->getDeviceBuffer().setValue(AccumPolicy::initValue);
                }

                HINLINE typename Buffer::DataBoxType getHostDataBox()
                {
                    return buffer->getHostBuffer().getDataBox();
                    ;
                }

                HINLINE typename Buffer::DataBoxType getDeviceDataBox()
                {
                    return buffer->getDeviceBuffer().getDataBox();
                }

                HINLINE void hostToDevice()
                {
                    buffer->hostToDevice();
                }

                HINLINE void deviceToHost()
                {
                    buffer->deviceToHost();
                }

                //! Get detector size (shape)
                HINLINE DataSpace<DIM2> getSize() const
                {
                    return buffer->getHostBuffer().getDataSpace();
                }

                //! Create a particle functor for detection in the DetectParticles kernel
                HINLINE DetectParticle getDetectParticle(uint32_t currentStep) const
                {
                    DataSpace<simDim> const simSize(Environment<simDim>::get().SubGrid().getTotalDomain().size);
                    /* The simulation size hast to be a 3D vector so that it can be transformed into the detector
                       coordinate system. */
                    DataSpace<DIM3> simSize3D;
                    switch(simDim)
                    {
                    case DIM2:
                        simSize3D = DataSpace<DIM3>{simSize.x(), simSize.y(), 1};
                        break;
                    case DIM3:
                        simSize3D = simSize;
                        break;
                    }
                    // Object used to switch between simulation and detector coordinate systems
                    externalBeam::AxisSwap const axisSwap = getAxisSwap(detectorPlacement_m);
                    // transform into detector system
                    DataSpace<DIM3> simSizeDet(axisSwap.rotate(simSize3D, true));
                    // additional detector description for the device functors
                    DetectorParams detParams(getSize(), precisionCast<float_X>(anglePerCell_m), detectorPlacement_m);
                    return DetectParticle(
                        detail::GetTargetCellIdx<Config>(detParams, simSizeDet, axisSwap),
                        accumHostFunctor(currentStep, detParams, simSizeDet),
                        axisSwap);
                }


                // TODO: add sth like validateConstraints from parataxis
            };

        } // namespace photonDetector
    } // namespace plugins
} // namespace picongpu
