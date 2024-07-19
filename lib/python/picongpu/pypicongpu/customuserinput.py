"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from .rendering import RenderedObject

import typing
import pydantic


class InterfaceCustomUserInput(RenderedObject, pydantic):
    """interface required from all custom input implementations"""

    def check_does_not_change_existing_key_values(self, firstDict: dict, secondDict: dict):
        """check that updating firstDict with secondDict will not change any value in firstDict"""
        for key in firstDict.keys():
            if (key in secondDict) and (firstDict[key] != secondDict[key]):
                raise ValueError("Key " + str(key) + " exist already, and specified values differ.")

    def check_tags(self, existing_tags: list[str], tags: list[str]):
        """
        check that all entries in tags are valid tags and that all tags in the union if the list elements are unique
        """
        if "" in tags:
            raise ValueError("tags must not be empty string!")
        for tag in tags:
            if tag in existing_tags:
                raise ValueError("duplicate tag provided!, tags must be unique!")

    def get_tags(self) -> list[str]:
        """get a list of all tags of this CustomUserInput"""
        raise NotImplementedError("Abstract interface only!")

    def _get_serialized(self) -> dict[str, typing.Any]:
        """get serialized representation of this Custom User Input implementation"""
        raise NotImplementedError("Abstract interface only!")

    def check(self) -> None:
        """throw error if self not correct/consistent"""
        raise NotImplementedError("Abstract Interface only!")

    def get_generic_rendering_context(self) -> dict[str, typing.Any]:
        """
        create and init a CustomUserInput instance and return the result of calling get_rendering_context() on it

        necessary since get_rendering_context() requires a schema to verify the result of _get_serialized() for the
            class in the registry which does not exist for user implementations

        along the lines of:
            return CustomUserInput(
                rendering_context=<serialized representation of implementation>,
                tags=<list of tags>
                ).get_rendering_context()
                )
        """
        raise NotImplementedError("Abstract Interface only!")


class CustomUserInput(InterfaceCustomUserInput):
    """
    container for easy passing of additional input as dict from user script to rendering context of simulation input
    """

    tags: typing.Optional[list[str]] = None
    """
    list of tags
    """

    rendering_context: typing.Optional[dict[str, typing.Any]] = None
    """
    accumulation variable of added dictionaries
    """

    def addToCustomInput(self, custom_input: dict[str, typing.Any], tag: str):
        """
        append dictionary to custom input dictionary
        """
        if tag == "":
            raise ValueError("tag must not be empty string!")
        if not custom_input:
            raise ValueError("custom input must contain at least 1 key")

        if (self.tags is None) and (self.rendering_context is None):
            self.tags = [tag]
            self.rendering_context = custom_input
        else:
            CustomUserInput.check_does_not_change_existing_key_values(self.rendering_context, custom_input)

            if tag in self.tags:
                raise ValueError("duplicate tag!")

            self.rendering_context.update(custom_input)
            self.tags.append(tag)

    def check(self) -> None:
        """no way to check without further knowledge, may be overwritten by the user"""
        pass

    def get_tags(self) -> list[str]:
        return self.tags

    def _get_serialized(self) -> dict[str, typing.Any]:
        self.check()
        return self.rendering_context

    def get_generic_rendering_context(self) -> dict[str, typing.Any]:
        return self.get_rendering_context()
