/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <pmacc/Environment.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/traits/IsSpecializationOf.hpp>

#include <string>
#include <string_view>

namespace picongpu::plugins::binning
{
    /**
     *  @brief Struct to hold information about a field
     *
     *  The constructor takes a id which describes how to get the field from the DataConnector
     *  @tparam Field The type of the field
     *
     */
    template<typename Field>
    struct FieldInfo
    {
        using FieldType = Field;
        std::string id;

        FieldInfo(std::string_view id) : id(id)
        {
        }

        std::string getId() const
        {
            return id;
        }
    };

    decltype(auto) transformFieldInfo(auto&& arg)
    {
        if constexpr(pmacc::concepts::SpecializationOf<std::decay_t<decltype(arg)>, FieldInfo>)
        {
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();
            return dc
                .get<typename std::remove_cvref_t<decltype(arg)>::FieldType>(std::forward<decltype(arg)>(arg).getId())
                ->getDeviceDataBox();
        }
        else
        {
            return std::forward<decltype(arg)>(arg);
        }
    };

} // namespace picongpu::plugins::binning
