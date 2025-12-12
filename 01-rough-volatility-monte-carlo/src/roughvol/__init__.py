"""roughvol — a Monte Carlo engine for rough-volatility option pricing.

Monte Carlo methods, random-number generation, OOP model design and NumPy
vectorization, applied to one of the most active topics in quantitative
finance: rough stochastic volatility.

Modules
-------
- black_scholes : closed-form BS price + implied-vol inversion (validation anchor)
- models        : GBM and RoughBergomi simulatable model objects
- mc_pricer     : European pricing with antithetic + control-variate variance reduction
- smile         : implied-vol smiles and the at-the-money skew term structure
- analysis      : roughness estimation, convergence studies, surfaces, validation gates
- plotting      : the report figures
- style         : shared matplotlib theme and validated palette
"""

from . import analysis, black_scholes, mc_pricer, models, smile, style
from .mc_pricer import PriceEstimate, price_european
from .models import GBM, RoughBergomi

__all__ = [
    "analysis", "black_scholes", "mc_pricer", "models", "smile", "style",
    "GBM", "RoughBergomi", "price_european", "PriceEstimate",
]
__version__ = "0.2.0"
