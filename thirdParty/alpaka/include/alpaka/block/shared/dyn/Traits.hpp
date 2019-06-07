/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#include <alpaka/meta/IsStrictBase.hpp>

#include <alpaka/core/Common.hpp>

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
                    return
                        traits::GetMem<
                            T,
                            TBlockSharedMemDyn>
                        ::getMem(
                            blockSharedMemDyn);
                }

                namespace traits
                {
                    //#############################################################################
                    //! The AllocVar trait specialization for classes with BlockSharedMemDynBase member type.
                    template<
                        typename T,
                        typename TBlockSharedMemDyn>
                    struct GetMem<
                        T,
                        TBlockSharedMemDyn,
                        typename std::enable_if<
                            meta::IsStrictBase<
                                typename TBlockSharedMemDyn::BlockSharedMemDynBase,
                                TBlockSharedMemDyn
                            >::value
                        >::type>
                    {
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_ACC static auto getMem(
                            TBlockSharedMemDyn const & blockSharedMemDyn)
                        -> T *
                        {
                            // Delegate the call to the base class.
                            return
                                block::shared::dyn::getMem<
                                    T>(
                                        static_cast<typename TBlockSharedMemDyn::BlockSharedMemDynBase const &>(blockSharedMemDyn));
                        }
                    };
                }
            }
        }
    }
}
