/* Copyright 2023-2024 Tapish Narwal
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

#if(ENABLE_OPENPMD == 1)

#    include "picongpu/plugins/common/openPMDDefaultExtension.hpp"

#    include <pmacc/dimensions/DataSpace.hpp>

#    include <cstdint>
#    include <functional>
#    include <string>
#    include <tuple>

#    include <openPMD/Series.hpp>

namespace picongpu
{
    namespace plugins::binning
    {
        /** @brief Bin particles in enabled region
         *
         * All regions must be represented by a unique bit
         */
        enum ParticleRegion : uint32_t
        {
            /** Bounded - Particles inside the global simulation volume, 01 in binary, corresponds to the first bit */
            Bounded = 1 << 0,
            /** Leaving - Particles that have left the global simulation volume in this timestep, 10 in binary,
             * corresponds to the second bit */
            Leaving = 1 << 1
        };

        template<typename Parent, typename T_AxisTuple, typename T_DepositionData, typename T_Extras>
        struct BinningDataBase
        {
            using DepositionFunctorType = typename T_DepositionData::FunctorType;
            using DepositedQuantityType = typename T_DepositionData::QuantityType;
            // @todo infer type from functor
            // using DepositedQuantityType = std::invoke_result_t<TDepositedQuantityFunctor, particle, worker>;

            std::string binnerOutputName;
            T_AxisTuple axisTuple;
            T_DepositionData depositionData;
            T_Extras extraData;
            pmacc::DataSpace<std::tuple_size_v<T_AxisTuple>> axisExtentsND;

            /* Optional parameters not initialized by constructor.
             * Use the return value of addBinner() to modify them if needed. */
            std::function<void()> hostHook = [] {};
            bool timeAveraging = true;
            bool normalizeByBinVolume = true;
            std::string notifyPeriod = "1";
            uint32_t dumpPeriod = 0u;

            std::string openPMDInfix = "_%06T.";
            std::string openPMDExtension = openPMD::getDefaultExtension();
            std::function<void(::openPMD::Series& series, ::openPMD::Iteration& iteration, ::openPMD::Mesh& mesh)>
                writeOpenPMDFunctor = std::function<
                    void(::openPMD::Series& series, ::openPMD::Iteration& iteration, ::openPMD::Mesh& mesh)>();
            std::string openPMDJsonCfg = "{}";

            BinningDataBase(
                std::string const& binnerName,
                T_AxisTuple const& axes,
                T_DepositionData const& depositData,
                T_Extras const& extraData)
                : binnerOutputName{binnerName}
                , axisTuple{axes}
                , depositionData{depositData}
                , extraData{extraData}
            {
                std::apply(
                    [&](auto const&... tupleArgs)
                    {
                        uint32_t i = 0;
                        // This assumes getNBins() exists
                        ((axisExtentsND[i++] = tupleArgs.getNBins()), ...);
                    },
                    axisTuple);
            }

            static constexpr uint32_t getNAxes()
            {
                return std::tuple_size_v<T_AxisTuple>;
            }

            /** @brief Time average the accumulated data when doing the dump. Defaults to true. */
            Parent& setTimeAveraging(bool timeAv)
            {
                this->timeAveraging = timeAv;
                return *static_cast<Parent*>(this);
            }
            /** @brief Defaults to true */
            Parent& setNormalizeByBinVolume(bool normalize)
            {
                this->normalizeByBinVolume = normalize;
                return *static_cast<Parent*>(this);
            }
            /** @brief The periodicity of the output. Defaults to 1 */
            Parent& setNotifyPeriod(std::string notify)
            {
                this->notifyPeriod = std::move(notify);
                return *static_cast<Parent*>(this);
            }
            /** @brief The number of notify steps to accumulate over. Dump at the end. Defaults to 1. */
            Parent& setDumpPeriod(uint32_t dumpXNotifys)
            {
                this->dumpPeriod = dumpXNotifys;
                return *static_cast<Parent*>(this);
            }

            /** @brief The periodicity of the output. Defaults to 1 */
            Parent& setOpenPMDExtension(std::string extension)
            {
                this->openPMDExtension = std::move(extension);
                return *static_cast<Parent*>(this);
            }

            /** @brief The periodicity of the output. Defaults to 1 */
            Parent& setOpenPMDInfix(std::string infix)
            {
                this->openPMDInfix = std::move(infix);
                return *static_cast<Parent*>(this);
            }

            /** @brief The periodicity of the output. Defaults to 1 */
            Parent& setOpenPMDWriteFunctor(
                std::function<void(::openPMD::Series& series, ::openPMD::Iteration& iteration, ::openPMD::Mesh& mesh)>
                    writeOpenPMDFunctor)
            {
                this->writeOpenPMDFunctor = std::move(writeOpenPMDFunctor);
                return *static_cast<Parent*>(this);
            }

            /** @brief The periodicity of the output. Defaults to 1 */
            Parent& setOpenPMDJsonCfg(std::string cfg)
            {
                this->openPMDJsonCfg = std::move(cfg);
                return *static_cast<Parent*>(this);
            }

            /** @brief A hook to execute code at every notify, before binning is done
             * A potential use is to fill fieldTmp
             */
            Parent& setHostSideHook(std::function<void()> hookFunc)
            {
                this->hostHook = std::move(hookFunc);
                return *static_cast<Parent*>(this);
            }
        };


        template<typename T_AxisTuple, typename T_SpeciesTuple, typename T_DepositionData, typename T_Extras>
        struct ParticleBinningData
            : public BinningDataBase<
                  ParticleBinningData<T_AxisTuple, T_SpeciesTuple, T_DepositionData, T_Extras>,
                  T_AxisTuple,
                  T_DepositionData,
                  T_Extras>
        {
            T_SpeciesTuple speciesTuple;
            uint32_t particleRegion{ParticleRegion::Bounded};

            ParticleBinningData(
                std::string const& binnerName,
                T_AxisTuple const& axes,
                T_SpeciesTuple const& species,
                T_DepositionData const& depositData,
                T_Extras const& extraData)
                : BinningDataBase<ParticleBinningData, T_AxisTuple, T_DepositionData, T_Extras>(
                    binnerName,
                    axes,
                    depositData,
                    extraData)
                , speciesTuple{species}
            {
            }

            // enable a region in the bitmask
            ParticleBinningData& enableRegion(ParticleRegion const region)
            {
                particleRegion = particleRegion | region;
                return *this;
            }
            // disable a region in the bitmask
            ParticleBinningData& disableRegion(ParticleRegion const region)
            {
                particleRegion = particleRegion & ~region;
                return *this;
            }
            // Check if a region is enabled in the bitmask
            bool isRegionEnabled(ParticleRegion const region) const
            {
                return (particleRegion & region) != 0;
            }
        };

        template<typename T_AxisTuple, typename T_FieldsTuple, typename T_DepositionData, typename T_Extras>
        struct FieldBinningData
            : public BinningDataBase<
                  FieldBinningData<T_AxisTuple, T_FieldsTuple, T_DepositionData, T_Extras>,
                  T_AxisTuple,
                  T_DepositionData,
                  T_Extras>
        {
            T_FieldsTuple fieldsTuple;

            FieldBinningData(
                std::string const& binnerName,
                T_AxisTuple const& axes,
                T_FieldsTuple const& fields,
                T_DepositionData const& depositData,
                T_Extras const& extraData)
                : BinningDataBase<FieldBinningData, T_AxisTuple, T_DepositionData, T_Extras>(
                    binnerName,
                    axes,
                    depositData,
                    extraData)
                , fieldsTuple{fields}
            {
            }
        };

    } // namespace plugins::binning
} // namespace picongpu

#endif
