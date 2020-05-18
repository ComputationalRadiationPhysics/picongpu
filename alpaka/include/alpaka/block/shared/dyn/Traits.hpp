/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Concepts.hpp>

#include <type_traits>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The grid block specifics
    namespace block
    {
        //-----------------------------------------------------------------------------
        //! The block shared memory operation specifics.
        namespace shared
        {
            //-----------------------------------------------------------------------------
            //! The block shared dynamic memory operation specifics.
            namespace dyn
            {
                struct ConceptBlockSharedDyn{};

                //-----------------------------------------------------------------------------
                //! The block shared dynamic memory operation traits.
                namespace traits
                {
                    //#############################################################################
                    //! The block shared dynamic memory get trait.
                    template<
                        typename T,
                        typename TBlockSharedMemDyn,
                        typename TSfinae = void>
                    struct GetMem;
                }

                //-----------------------------------------------------------------------------
                //! Returns the pointr to the block shared dynamic memory.
                //!
                //! \tparam T The element type.
                //! \tparam TBlockSharedMemDyn The block shared dynamic memory implementation type.
                //! \param blockSharedMemDyn The block shared dynamic memory implementation.
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename T,
                    typename TBlockSharedMemDyn>
                ALPAKA_FN_ACC auto getMem(
                    TBlockSharedMemDyn const & blockSharedMemDyn)
                -> T *
                {
                    using ImplementationBase = concepts::ImplementationBase<ConceptBlockSharedDyn, TBlockSharedMemDyn>;
                    return
                        traits::GetMem<
                            T,
                            ImplementationBase>
                        ::getMem(
                            blockSharedMemDyn);
                }
            }
        }
    }
}
