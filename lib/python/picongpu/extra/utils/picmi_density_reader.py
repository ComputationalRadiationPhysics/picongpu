import numpy as np
import matplotlib.pyplot as plt
import sympy
import json


class CppFctReader:
    """class that alllows to evalute the PIConGPU free density
    from a c++ code string

    BE AWARE: the code assumes that:
      1.) a free formular density profile was used in PICMI
      2.) if sympy.Piecewise was used, that the c++ code first case, will not branch
    """

    def __init__(self, s, debug=False):
        """constructor
        s ... string from pypicongpu.json file
        debug ... bool for debug output
        """
        self.debug = debug
        self.cpp_code = s.replace("\n", " ")
        self.cpp_code = self.initial_clean(self.cpp_code)

    def evaluate(self, x, symbol="y"):
        """evalute expression
        x ... float: position [m] where to evaluate the density
        symbol ... string: symbol to replace (default: 'y')
        """
        return self.inner_evaluate(self.cpp_code, x, symbol=symbol) + 0.0  # add float for cast

    def test_for_cases(self, s):
        """internal method checking for c++ cases
        s ... string to check
        """
        index_first_q = s.find("?")
        index_first_c = s.find(":")
        if index_first_q == -1 and index_first_c == -1:
            return False
        else:
            return True

    def inner_evaluate(self, s, x, symbol):
        """evalute string - this is a recursevly called internal method
        s ... string code
        x ... float value to evalute density at
        symbol ... string symbol to replace
        """
        if self.test_for_cases(s):
            if self.debug:
                print("CASES")
            return self.eval_cases(s, x, symbol)

        else:
            s_sym = sympy.parsing.sympy_parser.parse_expr(s)
            res = s_sym.subs(symbol, x)
            if self.debug:
                print("eval:", res)
            return res

    def eval_cases(self, s, x, symbol):
        """handle c++ cases of form condition ? case1 : case 2
        s ... string code
        x ... float value to evalute
        symbol ... symbol to replace in string s
        """
        if self.debug:
            print("-->", s)
        s = self.clean_substring(s)
        if self.debug:
            print("==>", s)
        index_first_q = s.find("?")
        index_first_c = s.find(":")  # this assumes that there are no cases in the first branch

        condition = s[0:index_first_q]
        case1 = s[index_first_q + 1 : index_first_c]
        case2 = s[index_first_c + 1 :]

        if self.debug:
            print("if")
            print(condition)
            print("do")
            print(case1)
            print("else")
            print(case2)
            print("fi")

        if self.inner_evaluate(condition, x, symbol):
            return self.inner_evaluate(case1, x, symbol)
        else:
            return self.inner_evaluate(case2, x, symbol)

    def initial_clean(self, s):
        """initially clean string from c++/picongpu methods
        making it readably for sympy
        s ... string
        """
        if self.debug:
            print("clean before:", s)
        s = s.replace("^", "**")
        s = s.replace("pmacc::math::exp", "exp")
        s = s.replace("pmacc::math::pow", "pow")
        if self.debug:
            print("clean after:", s)
        return s

    def clean_substring(self, s):
        """clean any substrings from cases from paraneties and whitespace
        s ... string
        """
        while s[0].isspace():
            s = s[1:]

        while s[-1].isspace():
            s = s[:-1]

        if s[0] == "(" and s[-1] == ")":
            s = s[1:-1]
        return s


if __name__ == "__main__":
    # load pypicongpu.json, convert json to dict and extract equation for density
    # a pypicongpu.json is created for every PICMI call
    with open("pypicongpu.json") as file:
        sim_dict = json.load(file)
        density_fct_str = sim_dict["species_initmanager"]["operations"]["simple_density"][0]["profile"]["data"][
            "function_body"
        ]

    # create cpp_fct_reader class for later evaluation
    reader = CppFctReader(density_fct_str)

    # define positions where to evaluate the density
    x_array = np.linspace(0.0, 5.0e-3, 1000)
    n_array = np.zeros_like(x_array)

    # evalute density
    for i, x in enumerate(x_array):
        n_array[i] = reader.evaluate(x)

    # plot density
    plt.plot(x_array, n_array)
    plt.xlabel(r"$y \, \mathrm{[m]}$")
    plt.ylabel(r"$n \, \mathrm{[m^-3]}$")
    plt.yscale("log")
    plt.show()
