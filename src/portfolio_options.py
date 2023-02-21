RMS = [
    "MV",
    "MAD",
    "MSV",
    "FLPM",
    "SLPM",
    "CVaR",
    "EVaR",
    #"WR",
    "MDD",
    "ADD",
    "CDaR",
    "UCI",
    "EDaR",
    "RG",
    "TG",
    "TGRG",
    "CVRG",
]
rms_short = ["CVaR", "EVaR", "CDaR", "UCI", "EDaR", "FLPM"]

method_covs = [
    "hist",
    "ledoit",
    "oas",
    "shrunk",
    "gl",
    "ewma1",
    "ewma2",
    "jlogo",
    # "fixed",
    "spectral",
    "shrink",
]
mus = ["hist", "ewma1", "ewma2"]  # Method to estimate expected returns
kellys = ["exact", "approx", False]
