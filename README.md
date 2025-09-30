### Summary
This repository contains code to apply Canonical Representational Mapping (CRM) to various simulations and datasets.

### Installation guide
Download this repository, and add to your MATLAB search path. All figures and panels were created with MATLAB scripts. The typical run time for the ECoG analysis is a few hours, all other figures can be generated within a few minutes. A small demo to test the software on simulated data is part of `Fig2_simulation.m`

## Requirements
No non-standard hardware is required. No GPU is required. Code is confirmed to work on MATLAB R2024b and R2025a. We used the FieldTrip toolbox. To install MATLAB, follow the instructions on https://www.mathworks.com/help/install/index.html

### Repository Content
    .
    ├── external                # Ancillary code
    ├── Fig2                    # Code for simulations in Fig. 2
    ├── Fig3                    # Code for analyses and plots in Fig. 3
    ├── Fig4                    # Code for analyses and plots in Fig. 4
    ├── Fig5                    # Code for analysis in Fig. 5
    ├── Solver                  # CRM Solver; implemented in matlab.
    ├── LICENSE                 # Text file: This work is licensed under a CC Attribution 4.0 International License.
    └── README.md               # Text file: Open Know-How manifest files.   

## Dependencies

Data for figure 3 can be found in: 
Zada, Z., Nastase, S.A., Aubrey, B. et al. The “Podcast” ECoG dataset for modeling neural activity during natural language comprehension. Sci Data 12, 1135 (2025). https://doi.org/10.1038/s41597-025-05462-2

Data for figure 4 are available from the authors of Reagh et al. on request: 
Reagh, Z.M., Ranganath, C. Flexible reuse of cortico-hippocampal representations during encoding and recall of naturalistic events. Nat Commun 14, 1279 (2023). https://doi.org/10.1038/s41467-023-36805-5

Data for figure 5 are available from the authors of Rosenblum et al. on request: 
Rosenblum, H.L., Kim, S.H., Stout, J.J., Klintsova, A.Y., Griffin A.L. Choice Behaviors and Prefrontal–Hippocampal Coupling Are Disrupted in a Rat Model of Fetal Alcohol Spectrum Disorders. J Neuroscience 45, 10 (2025). https://doi.org/10.1523/JNEUROSCI.1241-24.2025

ShadedErrorBar is from: 
https://www.mathworks.com/matlabcentral/fileexchange/26311-raacampbell-shadederrorbar


### Issue
Issues are tracked through this repository. For minor updates, just push. For major ones, use a PR. 

### License
This work is free: you can redistribute it and/or modify it under the terms of the Creative Commons Attribution 4.0 International license, version 4 of the License, or (at your option) any later version (CC-BY-4.0). This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY, to the extent permitted by law; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. A copy of the License is provided in this repository.  For more details, see <http://www.gnu.org/licenses/>.

