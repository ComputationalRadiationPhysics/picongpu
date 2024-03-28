/* Copyright 2024 Julian Lenz
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

#include <pmacc/math/Vector.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/meta/conversion/MakeSeq.hpp>

#include <boost/mp11/bind.hpp>

#include <numeric>
#include <type_traits>

#include <nlohmann/json.hpp>


namespace picongpu
{
    namespace traits
    {
        template<typename, typename = void>
        inline constexpr bool providesMetadata = false;

        template<typename T>
        inline constexpr bool providesMetadata<T, std::void_t<decltype(std::declval<T>().metadata())>> = true;

        template<typename, typename = void>
        inline constexpr bool providesMetadataAtCT = false;

        template<typename T>
        inline constexpr bool providesMetadataAtCT<T, std::void_t<decltype(T::metadata())>> = true;

        template<typename, typename = void>
        inline constexpr bool providesMetadataAtRT = false;

        template<typename T>
        inline constexpr bool
            providesMetadataAtRT<T, std::enable_if_t<providesMetadata<T> && !providesMetadataAtCT<T>>> = true;

        namespace detail
        {
            /**
             * An empty return type to distinguish the fallback implementation of GetMetadata
             *
             * It has a template argument, so that the static_assert in to_json can print the type of TObject.
             */
            template<typename TObject>
            struct ReturnTypeFromDefault
            {
            };

            /**
             * A template that's always false but technically depends on its template parameter, so that its
             * instantiated only in the second stage (and as such does not trigger a static_assert if that function is
             * not used).
             */
            template<typename>
            inline constexpr bool False = false;

            /**
             * Always-(CT-)failing conversion to json, so we get a nice error message at CT.
             */
            template<typename T>
            void to_json(nlohmann::json&, ReturnTypeFromDefault<T> const&)
            {
                static_assert(
                    False<T>,
                    "You're missing metadata for a type supposed to provide some. There are three alternatives for "
                    "you: Specialise GetMetadata<YourType>, add a .metadata() method to your type, or use "
                    "AllowMissingMetadata<YourType> during the registration. For more infos, see "
                    "docs/source/usage/metadata.rst.");
            }
        } // namespace detail

        /**
         * Main customisation point for the content of the metadata reported
         *
         * As a user, provide a template specialisation for the type in question in order to customise the information
         * reported by it. The defaults reach to the (static -- for CT) member function `metadata` to provide such
         * information. If none is provided by the class, this attempt fails at compiletime.
         *
         * @tparam TObject type of the object we want to provide information about
         * @tparam SFINAE parameter used to provide defaults for RT and CT, DO NOT TOUCH
         */
        template<typename TObject, typename = void>
        struct GetMetadata
        {
            detail::ReturnTypeFromDefault<TObject> description() const
            {
                return {};
            }
        };

        // doc-include-start: GetMetdata trait
        template<typename TObject>
        struct GetMetadata<TObject, std::enable_if_t<providesMetadataAtRT<TObject>>>
        {
            // Holds a constant reference to the RT instance it's supposed to report about.
            // Omit this for the CT specialisation!
            TObject const& obj;

            nlohmann::json description() const
            {
                return obj.metadata();
            }
        };

        template<typename TObject>
        struct GetMetadata<TObject, std::enable_if_t<providesMetadataAtCT<TObject>>>
        {
            // CT version has no members. Apart from that, the interface is identical to the RT version.

            nlohmann::json description() const
            {
                return TObject::metadata();
            }
        };
        // doc-include-end: GetMetdata trait

        /**
         * Policy to wrap another type to allow missing metadata for it.
         *
         * It doesn't do much by itself. Functionality is provided by the corresponding
         * GetMetadata<AllowMissingMetadata<...>> specialisation.
         *
         * @tparam TObject type to apply this policy to
         */
        // doc-include-start: AllowMissingMetadata
        template<typename TObject>
        struct AllowMissingMetadata
        {
            // it's probably a nice touch to provide this, so people don't need a lot of
            // template metaprogramming to get `TObject`
            using type = TObject;
        };

        template<typename TObject>
        struct GetMetadata<AllowMissingMetadata<TObject>> : GetMetadata<TObject>
        {
            nlohmann::json description() const
            {
                return handle(GetMetadata<TObject>::description());
            }

            static nlohmann::json handle(nlohmann::json const& result)
            {
                // okay, we've found metadata, so we return it
                return result;
            }

            static nlohmann::json handle(detail::ReturnTypeFromDefault<TObject> const& result)
            {
                // also okay, we couldn't find metadata, so we'll return an empty object
                return nlohmann::json::object();
            }
        };
        // doc-include-end: AllowMissingMetadata

        /**
         * Policy to provide context for incident fields
         *
         * Incident fields are lacking the knowledge of their own context (in particular from which direction they are
         * incident), so we don't let them report immediately but instead use this policy to add the necessary context.
         *
         * @tparam Profiles a pmacc::MakeSeq_t list of profiles (typically this will be
         * `picongpu::fields::incidentField::EnabledProfiles`).
         */
        template<typename BoundaryName, typename Profiles>
        struct IncidentFieldPolicy
        {
        };

        namespace detail
        {
            /**
             * Gather the metadata from a list of profiles into one annotated json object.
             *
             * @tparam T_Pack Any class template (typically pmacc::MakeSeq_t but could be anything, e.g., std::tuple),
             * not used directly, we're only interested in its template parameters
             * @tparam Profiles The list of CT profiles we're actually interested in and want to report about.
             */
            template<template<typename...> typename T_Pack, typename... Profiles>
            nlohmann::json gatherMetadata(T_Pack<Profiles...>)
            {
                std::vector<nlohmann::json> collection;
                (collection.push_back(GetMetadata<AllowMissingMetadata<Profiles>>{}.description()), ...);
                return collection;
            }
        } // namespace detail


        template<typename BoundaryName, typename Profiles>
        struct GetMetadata<IncidentFieldPolicy<BoundaryName, Profiles>>
        {
            nlohmann::json description() const
            {
                auto result = nlohmann::json::object();
                result["incidentField"][BoundaryName::str()] = detail::gatherMetadata(pmacc::MakeSeq_t<Profiles>{});
                return result;
            }
        };

    } // namespace traits
} // namespace picongpu

namespace pmacc::math
{
    /**
     * Provide conversion of pmacc::math::Vector to json.
     */
    template<typename T_Type, uint32_t T_dim, typename T_Navigator, typename T_Storage>
    void to_json(nlohmann::json& j, Vector<T_Type, T_dim, T_Navigator, T_Storage> const& vec)
    {
        std::vector<T_Type> stdvec{};
        for(size_t i = 0; i < T_dim; ++i)
        {
            stdvec.push_back(vec[i]);
        }
        j = stdvec;
    }
} // namespace pmacc::math
