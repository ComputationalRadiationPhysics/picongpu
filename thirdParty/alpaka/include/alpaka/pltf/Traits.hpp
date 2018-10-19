/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include <alpaka/core/Common.hpp>
#include <alpaka/dev/Traits.hpp>

#include <boost/config.hpp>

#include <vector>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The platform specifics.
    namespace pltf
    {
        //-----------------------------------------------------------------------------
        //! The platform traits.
        namespace traits
        {
            //#############################################################################
            //! The platform type trait.
            template<
                typename T,
                typename TSfinae = void>
            struct PltfType;

            //#############################################################################
            //! The device count get trait.
            template<
                typename T,
                typename TSfinae = void>
            struct GetDevCount;

            //#############################################################################
            //! The device get trait.
            template<
                typename T,
                typename TSfinae = void>
            struct GetDevByIdx;
        }

        //#############################################################################
        //! The platform type trait alias template to remove the ::type.
        template<
            typename T>
        using Pltf = typename traits::PltfType<T>::type;

        //-----------------------------------------------------------------------------
        //! \return The device identified by its index.
        template<
            typename TPltf>
        ALPAKA_FN_HOST auto getDevCount()
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(traits::GetDevCount<TPltf>::getDevCount())
#endif
        {
            return
                traits::GetDevCount<
                    TPltf>
                ::getDevCount();
        }

        //-----------------------------------------------------------------------------
        //! \return The device identified by its index.
        template<
            typename TPltf>
        ALPAKA_FN_HOST auto getDevByIdx(
            std::size_t const & devIdx)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(traits::GetDevByIdx<TPltf>::getDevByIdx(devIdx))
#endif
        {
            return
                traits::GetDevByIdx<
                    TPltf>
                ::getDevByIdx(
                    devIdx);
        }

        //-----------------------------------------------------------------------------
        //! \return All the devices available on this accelerator.
        template<
            typename TPltf>
        ALPAKA_FN_HOST auto getDevs()
        -> std::vector<dev::Dev<TPltf>>
        {
            std::vector<dev::Dev<TPltf>> devs;

            std::size_t const devCount(getDevCount<TPltf>());
            for(std::size_t devIdx(0); devIdx < devCount; ++devIdx)
            {
                devs.push_back(getDevByIdx<TPltf>(devIdx));
            }

            return devs;
        }
    }
}
