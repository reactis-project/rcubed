"""ctmc_pbee: CTMC-based Performance-Based Earthquake Engineering.

Code accompanying Nardin, Marelli, Sudret, Broccardo, "A Generalized Framework
for Performance-Based Earthquake Engineering: Integrated Assessment of
Structural Reliability and Resilience" (STRUCS-S-26-00168).

Modules
-------
core        : reliability (F, beta) and resilience (R0, rho) of the CTMC, Eq. 8-27
generator   : generator matrices Q from hazard, fragilities and recovery (case study)
fragilities : simulated state-dependent fragility curves (case study)
plotting    : figures
utils       : helpers (tables, figure saving, nested dicts, comparisons)
paths       : repository folders (data/, output/, figures/)
"""

__version__ = "0.1.0"
