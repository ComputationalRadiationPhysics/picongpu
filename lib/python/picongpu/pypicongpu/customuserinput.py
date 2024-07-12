"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from .rendering import RenderedObject

import typeguard
import typing


class CustomUserInput(RenderedObject):
    """
    container for easy passing of additional input as dict from user script to rendering context of simulation input

    if additional
    """

    tags: typing.Optional[list[str]] = None
    """
    list of tags
    """

    rendering_context: typing.Optional[dict[str, typing.Any]] = None
    """
    accumulation variable of added dictionaries
    """

    def __checkDoesNotChangeExistingKeyValues(self, firstDict, secondDict):
        for key in firstDict.keys():
            if (key in secondDict) and (firstDict[key] != secondDict[key]):
                raise ValueError("Key " + str(key) + " exist already, and specified values differ.")

    @typeguard.typechecked
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
            self.__checkDoesNotChangeExistingKeyValues(self.rendering_context, custom_input)

            if tag in self.tags:
                raise ValueError("duplicate tag!")

            self.rendering_context.update(custom_input)
            self.tags.append(tag)

    def get_tags(self) -> list[str]:
        return self.tags

    def _get_serialized(self) -> dict:
        return self.rendering_context
