/*
 * SPDX-FileCopyrightText: Tapish Narwal
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#if (ENABLE_OPENPMD == 1)

#    include "picongpu/plugins/common/openPMDDefaultExtension.hpp"

#    include <pmacc/dimensions/DataSpace.hpp>
#    include <pmacc/math/operation/traits.hpp>
#    include <pmacc/mpi/GetMPI_Op.hpp>

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

        template<
            typename Child,
            typename T_BinaryOp,
            typename T_AxisTuple,
            typename T_DepositionData,
            typename T_Extras>
        requires requires {
            typename pmacc::math::operation::traits::AlpakaAtomicOp_t<T_BinaryOp>;
            pmacc::mpi::getMPI_Op<T_BinaryOp>();
            pmacc::math::operation::traits::NeutralElement_v<T_BinaryOp, typename T_DepositionData::QuantityType>;
        }
        struct BinningDataBase
        {
            using DepositedQuantityType = typename T_DepositionData::QuantityType;
            using ReductionOp = T_BinaryOp;
            // @todo infer type from functor
            // using DepositedQuantityType = std::invoke_result_t<TDepositedQuantityFunctor, particle, worker>;

            std::string binnerOutputName;
            T_AxisTuple axisTuple;
            T_DepositionData depositionData;
            T_Extras extraData;
            pmacc::DataSpace<std::tuple_size_v<T_AxisTuple>> axisExtentsND;

            /* Optional parameters not initialized by constructor.
             * Use the return value of add...Binner() to modify them if needed. */
            std::function<void()> hostHook = [] {};
            std::string notifyPeriod = "1";
            uint32_t dumpPeriod = 0u;

            std::string openPMDInfix = "_%06T.";
            std::string openPMDExtension = openPMD::getDefaultExtension();
            std::function<void(::openPMD::Series& series, ::openPMD::Iteration& iteration, ::openPMD::Mesh& mesh)>
                writeOpenPMDFunctor = std::function<
                    void(::openPMD::Series& series, ::openPMD::Iteration& iteration, ::openPMD::Mesh& mesh)>();
            std::string openPMDBackendConfig = "{}";

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

            // safely cast this to child
            Child& interpretAsChild()
            {
                return *static_cast<Child*>(this);
            }

            /** @brief The periodicity of the output. Defaults to 1 */
            Child& setNotifyPeriod(std::string notify)
            {
                notifyPeriod = std::move(notify);
                return interpretAsChild();
            }

            /** @brief The number of notify steps to do the reduction over. Dump at the end. Defaults to 1. */
            Child& setDumpPeriod(uint32_t dumpXNotifys)
            {
                dumpPeriod = dumpXNotifys;
                return interpretAsChild();
            }

            /** @brief Set the file extension for the openPMD output */
            Child& setOpenPMDExtension(std::string extension)
            {
                openPMDExtension = std::move(extension);
                return interpretAsChild();
            }

            /** @brief Set the infix for file names in openPMD output */
            Child& setOpenPMDInfix(std::string infix)
            {
                openPMDInfix = std::move(infix);
                return interpretAsChild();
            }

            /** @brief Call a functor to add custom data to openPMD output */
            Child& setOpenPMDWriteFunctor(
                std::function<void(::openPMD::Series& series, ::openPMD::Iteration& iteration, ::openPMD::Mesh& mesh)>
                    functor)
            {
                writeOpenPMDFunctor = std::move(functor);
                return interpretAsChild();
            }

            /** @brief Set backend-specific configuration for openPMD in JSON format (used when writing) */
            Child& setOpenPMDBackendConfig(std::string cfg)
            {
                openPMDBackendConfig = std::move(cfg);
                return interpretAsChild();
            }

            /** @brief A hook to execute code at every notify, before binning is done
             * A potential use is to fill fieldTmp
             */
            Child& setHostSideHook(std::function<void()> hookFunc)
            {
                hostHook = std::move(hookFunc);
                return interpretAsChild();
            }
        };

        template<
            typename T_BinaryOp,
            typename T_AxisTuple,
            typename T_SpeciesTuple,
            typename T_DepositionData,
            typename T_Extras>
        struct ParticleBinningData
            : public BinningDataBase<
                  ParticleBinningData<T_BinaryOp, T_AxisTuple, T_SpeciesTuple, T_DepositionData, T_Extras>,
                  T_BinaryOp,
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
                : BinningDataBase<ParticleBinningData, T_BinaryOp, T_AxisTuple, T_DepositionData, T_Extras>(
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

        template<
            typename T_BinaryOp,
            typename T_AxisTuple,
            typename T_SpeciesTuple,
            typename T_DepositionData,
            typename T_Extras>
        ParticleBinningData<T_BinaryOp, T_AxisTuple, T_SpeciesTuple, T_DepositionData, T_Extras>
        makeParticleBinningData(
            std::string const& binnerOutputName,
            T_AxisTuple const& axisTupleObject,
            T_SpeciesTuple const& speciesTupleObject,
            T_DepositionData const& depositionData,
            T_Extras const& extraData)
        {
            return ParticleBinningData<T_BinaryOp, T_AxisTuple, T_SpeciesTuple, T_DepositionData, T_Extras>(
                binnerOutputName,
                axisTupleObject,
                speciesTupleObject,
                depositionData,
                extraData);
        }

        template<
            typename T_BinaryOp,
            typename T_AxisTuple,
            typename T_FieldsTuple,
            typename T_DepositionData,
            typename T_Extras>
        struct FieldBinningData
            : public BinningDataBase<
                  FieldBinningData<T_BinaryOp, T_AxisTuple, T_FieldsTuple, T_DepositionData, T_Extras>,
                  T_BinaryOp,
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
                : BinningDataBase<FieldBinningData, T_BinaryOp, T_AxisTuple, T_DepositionData, T_Extras>(
                      binnerName,
                      axes,
                      depositData,
                      extraData)
                , fieldsTuple{fields}
            {
            }
        };

        template<
            typename T_BinaryOp,
            typename T_AxisTuple,
            typename T_FieldsTuple,
            typename T_DepositionData,
            typename T_Extras>
        FieldBinningData<T_BinaryOp, T_AxisTuple, T_FieldsTuple, T_DepositionData, T_Extras> makeFieldBinningData(
            std::string const& binnerName,
            T_AxisTuple const& axes,
            T_FieldsTuple const& fields,
            T_DepositionData const& depositData,
            T_Extras const& extraData)
        {
            return FieldBinningData<T_BinaryOp, T_AxisTuple, T_FieldsTuple, T_DepositionData, T_Extras>(
                binnerName,
                axes,
                fields,
                depositData,
                extraData);
        }

    } // namespace plugins::binning
} // namespace picongpu

#endif
