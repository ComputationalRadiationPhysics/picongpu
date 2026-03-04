from . import picmi
from . import pypicongpu


def load_tests(loader, standard_tests, pattern):
    standard_tests.addTests((loader.loadTestsFromModule(module, pattern=pattern) for module in (picmi, pypicongpu)))
    return standard_tests
