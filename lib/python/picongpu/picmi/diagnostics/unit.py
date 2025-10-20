import numpy as np

class Unit:
    """
    class to describe units
    """
    
    # number of unit dimension
    N_unit_dim = 7
    
    """
    The unit index map associates names with array indices.
    The name definition can be found at https://en.wikipedia.org/wiki/SI_base_unit.
    The array index order follows that of PIConGPU:
    include/picongpu/plugins/binning/UnitConversion.hpp, lines 40–48.
    """    
    _unit_index_map = {
        "length": 0, "L": 0,
        "mass": 1, "M": 1,
        "time": 2, "T": 2,
        "electric current": 3, "I": 3,
        "thermodynamic temperature": 4, "Θ": 4,
        "amount of substance": 5, "N": 5,
        "luminous intensity": 6, "J": 6
    }    
    
    def __init__(self, name = None):
        """set unit vector either empty or by name"""
        self.unit_vector = np.zeros((self.N_unit_dim))

        if name != None:
            index = self._unit_index_map.get(name)
            if index is not None:
                self.unit_vector[index] = 1.0
            else:
                raise ValueError(f"Unknown unit name: {name}")

    def __getitem__(self, name):
        """access component by name"""
        index = self._unit_index_map.get(name)
        if index is None:
            raise KeyError(f"Unknown unit name: {name}")
        return self.unit_vector[index]

    def __iter__(self):
        """return iterator of unit based on PIConGPU order: LMTIΘNJ"""
        return iter(self.unit_vector)

    def __str__(self):
        """return string representation that only outputs relevant units"""
        output = ""
        # invert from (name to index) to (index to (short) name)
        inverted = {v: k for k, v in self._unit_index_map.items()}

        for i in range(self.N_unit_dim):
            if self.unit_vector[i] != 0.0:
                output += f"{inverted[i]}^{self.unit_vector[i]} "
        return output
    
    def __pow__(self, exponent):
        """rase unit to a power"""
        result = Unit()
        result.unit_vector = self.unit_vector * exponent
        return result
    
    def __mul__(self, factor):
        """multiply units with each other"""
        result = Unit()        
        result.unit_vector = self.unit_vector + factor.unit_vector
        return result
        
    def __truediv__(self, divisor):
        """divide one unit by another"""
        result = Unit()
        result.unit_vector = self.unit_vector - divisor.unit_vector
        return result 

# predefined units 
T = Unit("T")
M = Unit("M")
L = Unit("L")
I = Unit("I")
