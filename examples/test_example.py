import matplotlib.pyplot as plt
import numpy as np
from virtual_accelerator.models.staged_model import get_cu_hxr_staged_model
import os

os.environ["LCLS_LATTICE"] = "/Users/smiskov/SLAC/lcls-lattice"
import warnings

warnings.filterwarnings("ignore")


# do a quadrupole scan using QUAD:IN20:525:BCTRL and plot the beam distribution at OTR3
model = get_cu_hxr_staged_model()

# for i in model.supported_variables:
#     print(i)
SCAN_QUAD_PV = "QUAD:IN20:525:BCTRL"
model.set({SCAN_QUAD_PV: -10.5})
b = model.get(["OTR4_beam"])

print(b["OTR4_beam"]["norm_emit_x"])

# OTR_IMAGE_PV = "OTRS:IN20:711:Image:ArrayData"
# SCAN_QUAD_PV = "QUAD:IN20:525:BCTRL"
#
# quad_values = np.linspace(-10, 10, 5)
#
# fig, ax = plt.subplots(1, len(quad_values), sharex=True, sharey=True)
#
# rms_sizes = []
# for i, quad_value in enumerate(quad_values):
#     print(f"Setting {SCAN_QUAD_PV} to {quad_value:.1f}")
#     model.set({SCAN_QUAD_PV: quad_value})
#     image = model.get([OTR_IMAGE_PV])[OTR_IMAGE_PV]
#     ax[i].imshow(-image, rasterized=True)
#     ax[i].set_title(f"{SCAN_QUAD_PV}={quad_value:.1f}", fontsize="x-small")
#
# fig.savefig("cu_hxr_staged_model_quad_scan.svg")
