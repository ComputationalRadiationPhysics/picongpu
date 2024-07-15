"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from .ionizationcurrent import IonizationCurrent


class None_(IonizationCurrent):
    picongpu_name: str = "None"
