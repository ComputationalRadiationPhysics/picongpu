/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#include <alpaka/core/BoostPredef.hpp>

#if !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/idx/Traits.hpp>

#include <alpaka/vec/Vec.hpp>
#include <alpaka/core/Concepts.hpp>
#include <alpaka/core/Hip.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Unused.hpp>

#include <type_traits>

namespace alpaka
{
    namespace idx
    {
        namespace gb
        {
            //#############################################################################
            //! The HIP accelerator ND index provider.
            template<
                typename TDim,
                typename TIdx>
            class IdxGbHipBuiltIn : public concepts::Implements<ConceptIdxGb, IdxGbHipBuiltIn<TDim, TIdx>>
            {
            public:
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST_ACC IdxGbHipBuiltIn() = default;
                //-----------------------------------------------------------------------------
                IdxGbHipBuiltIn(IdxGbHipBuiltIn const &) = delete;
                //-----------------------------------------------------------------------------
                IdxGbHipBuiltIn(IdxGbHipBuiltIn &&) = delete;
                //-----------------------------------------------------------------------------
                auto operator=(IdxGbHipBuiltIn const & ) -> IdxGbHipBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                auto operator=(IdxGbHipBuiltIn &&) -> IdxGbHipBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                /*virtual*/ ALPAKA_FN_HOST_ACC ~IdxGbHipBuiltIn() = default;
            };
        }
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator index dimension get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                idx::gb::IdxGbHipBuiltIn<TDim, TIdx>>
            {
                using type = TDim;
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator grid block index get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetIdx<
                idx::gb::IdxGbHipBuiltIn<TDim, TIdx>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current block in the grid.
                template<
                    typename TWorkDiv>
                ALPAKA_FN_HOST_ACC static auto getIdx(
                    idx::gb::IdxGbHipBuiltIn<TDim, TIdx> const & idx,
                    TWorkDiv const &)
                -> vec::Vec<TDim, TIdx>
                {
                    alpaka::ignore_unused(idx);
                    return offset::getOffsetVecEnd<TDim>(
                        vec::Vec<std::integral_constant<typename TDim::value_type, 3>, TIdx>(
                            static_cast<TIdx>(hipBlockIdx_z),
                            static_cast<TIdx>(hipBlockIdx_y),
                            static_cast<TIdx>(hipBlockIdx_x)));
                }
            };

            //#############################################################################
            //! The GPU HIP accelerator grid block index idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                idx::gb::IdxGbHipBuiltIn<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
