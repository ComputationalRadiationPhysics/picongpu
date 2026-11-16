from . import densityprofile as densityprofile
from . import momentum as momentum
from .setchargestate import SetChargeState as SetChargeState
from .simpledensity import SimpleDensity as SimpleDensity
from .simplemomentum import SimpleMomentum as SimpleMomentum

AnyOperation = SimpleDensity | SimpleMomentum | SetChargeState
