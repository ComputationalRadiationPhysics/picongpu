"""
This file is part of PIConGPU.
Copyright 2024-2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from typing import Any

from pydantic import BaseModel, Field, model_serializer

from .rendering import RenderedObject


class CustomUserInput(RenderedObject, BaseModel):
    """
    container for easy passing of additional input as dict from user script to rendering context of simulation input
    """

    tags: list[str] | None = Field(None, min_length=1)
    """
    list of tags
    """

    rendering_context: dict[str, Any] | None = None
    """
    accumulation variable of added dictionaries
    """

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

    def addToCustomInput(self, custom_input: dict[str, Any], tag: str):
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
            self.check_does_not_change_existing_key_values(self.rendering_context, custom_input)

            if tag in self.tags:
                raise ValueError("duplicate tag!")

            self.rendering_context.update(custom_input)
            self.tags.append(tag)

    def get_tags(self) -> list[str]:
        return self.tags

    @model_serializer(mode="plain")
    def _get_serialized(self) -> dict[str, Any] | None:
        return self.rendering_context
