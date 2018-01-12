import psrc_urbansim.models
import orca

# models are defined in psrc_urbansim.models

# uncomment models you want to estimate

# REPM
#orca.run(["repmres_estimate"])
#orca.run(["repmnr_estimate"])

# HLCM
#orca.run(["hlcm_estimate"])

# WPLCM
orca.run(["wplcm_estimate"])

# ELCM
#orca.run(["elcm_estimate"])
