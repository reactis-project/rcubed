"""ctmc_pbee: CTMC-based Performance-Based Earthquake Engineering.

Code accompanying Nardin, Marelli, Sudret, Broccardo, "A Generalized Framework
for Performance-Based Earthquake Engineering: Integrated Assessment of
Structural Reliability and Resilience" (STRUCS-S-26-00168).

Modules
-------
core        : CTMC model (generator Q, pi(t), Eq. 8-9, 28-29) and parametric studies
metrics     : reliability (F, beta) and resilience (Q0, R0, rho) metrics, Eq. 10-19
spectral    : spectral analysis and approximation of R0, Eq. 20-27
generator   : generator matrices Q from hazard, fragilities and recovery (case study)
fragilities : simulated state-dependent fragility curves (case study)
case_study  : case study of the paper (Sec. 4): inputs, Figs. 9-13, Tables 4-5
plotting    : figures
utils       : helpers (tables, figure saving, nested dicts, comparisons)
paths       : repository folders (data/, output/)
"""

__version__ = "0.1.0"
