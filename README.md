# Superspreader Epidemic Model

A spatially structured SIR (Susceptible-Infected-Recovered) epidemic simulation that models the impact of superspreaders on disease transmission dynamics. This project implements and compares two distinct superspreader models: the **Strong Infectiousness Model** and the **Hub Model**.

## Overview

This simulation explores how superspreaders—individuals with enhanced transmission capabilities—affect epidemic spread patterns, percolation thresholds, and outbreak dynamics. The project reproduces and extends results from epidemiological research on superspreader phenomena, including comparisons with real-world data from the 2003 SARS outbreak in Singapore.

## Key Features

- **Two Superspreader Models**: Strong Infectiousness and Hub models with different transmission mechanisms
- **Spatial Epidemic Simulation**: 2D periodic boundary conditions with distance-dependent infection probabilities
- **Comprehensive Analysis**: Percolation theory, critical density calculations, and epidemic curve generation
- **Real Data Comparison**: Validation against SARS outbreak data from Singapore (2003)
- **Extensive Visualization**: 15 different plots and network visualizations

## Models

### Strong Infectiousness Model
- **Normal individuals**: Infection probability decreases quadratically with distance up to cutoff `r₀`
- **Superspreaders**: Constant maximum infection probability `w₀` within the same cutoff distance
- Models individuals with enhanced viral shedding or transmission efficiency

### Hub Model  
- **Normal individuals**: Same distance-dependent transmission as Strong model (cutoff `r₀`)
- **Superspreaders**: Extended transmission range with cutoff `rs = √6 × r₀`
- Models individuals with increased social connectivity or mobility

## Installation
### Setup
```bash
# Clone the repository
git clone https://github.com/fisherman611/superspreader-epidemic-model.git
cd superspreader-epidemic-model

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running Individual Simulations

```python
from models.SIR import SIRSimulation

# Initialize simulation with default parameters
sim = SIRSimulation(r0=1, w0=1, gamma=1, alpha=2)

# Run single simulation
result = sim.run_simulation(
    N=500,                    # Number of individuals
    lambda_val=0.2,          # Fraction of superspreaders
    model_type='strong_infectiousness',  # or 'hub'
    max_steps=100,           # Maximum simulation steps
    initial_pos=(5, 0)       # Patient zero position
)

# Access results
print(f"Total infected: {np.sum(result['states'] > 0)}")
print(f"Secondary infections: {result['secondary_infections']}")
```

### Generating All Figures

```bash
python utils/SIR_plots.py
```

This will generate all 15 figures used in the analysis:

## Generated Figures

| Figure | Description | Filename |
|--------|-------------|----------|
| 1 | Strong Infectiousness Model Probabilities | `fig1_strong_infection_prob.png` |
| 2 | Hub Model Probabilities | `fig2_hub_infection_prob.png` |
| 3 | Strong Model Percolation | `fig3_strong_percolation.png` |
| 4 | Hub Model Percolation | `fig4_hub_percolation.png` |
| 5 | Critical Density vs Superspreader Fraction | `fig5_critical_density.png` |
| 6 | Distance Evolution (Strong Model) | `fig6_strong_distance_evolution.png` |
| 7 | Propagation Velocity | `fig7_propagation_velocity.png` |
| 8 | Epidemic Curves Comparison | `fig8_epidemic_curves.png` |
| 9 | Strong Model Network | `fig9_strong_network.png` |
| 10 | Hub Model Network | `fig10_hub_network.png` |
| 11 | No Superspreaders Network | `fig11_no_superspreaders_network.png` |
| 12 | Secondary Infections (λ=0.0) | `fig12_no_superspreaders_distribution.png` |
| 13 | Secondary Infections (λ=0.2) | `fig13_superspreaders_distribution.png` |
| 14 | SARS Secondary Cases Data | `fig14_sars_secondary_cases.png` |
| 15 | SARS Epidemic Curves Comparison | `fig15_sars_epidemic_curves.png` |

## Parameters

### Core Parameters
- **`r0`**: Infection cutoff distance for normal individuals (default: 1.0)
- **`w0`**: Base infection probability scaling factor (default: 1.0)  
- **`gamma`**: Recovery probability per time step (default: 1.0)
- **`alpha`**: Distance decay exponent (default: 2.0)
- **`L`**: Simulation space size = 10 × r₀
- **`rs`**: Superspreader cutoff distance = √6 × r₀ (Hub model)

### Simulation Parameters
- **`N`**: Population size
- **`lambda_val`**: Fraction of superspreaders (0 ≤ λ ≤ 1)
- **`model_type`**: `'strong_infectiousness'` or `'hub'`
- **`max_steps`**: Maximum simulation duration
- **`initial_pos`**: Starting position of patient zero

## Key Results

### Critical Findings
1. **Percolation Thresholds**: Both models show reduced critical density with increasing superspreader fraction
2. **Hub Model Efficiency**: More effective at low superspreader fractions due to extended transmission range
3. **Propagation Velocity**: Increases monotonically with superspreader fraction in both models
4. **Secondary Infection Distribution**: Heavy-tailed distributions emerge with superspreaders present

### SARS Validation
The models successfully reproduce key features of the 2003 SARS outbreak in Singapore:
- Heterogeneous secondary infection distributions
- Epidemic curve shapes and timing
- Impact of superspreading events on outbreak dynamics

## Project Structure

```
superspreader-epidemic-model/
├── models/
│   └── SIR.py                    # Core simulation class
├── utils/
│   └── SIR_plots.py             # Visualization and analysis functions
├── figures/                      # Generated plot outputs
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

## Mathematical Background

### Infection Probability Functions

**Strong Infectiousness Model:**
```
w(r) = w₀ × (1 - r/r₀)² for r ≤ r₀ (normal)
w(r) = w₀               for r ≤ r₀ (superspreader)
w(r) = 0                for r > r₀
```

**Hub Model:**
```
w(r) = w₀ × (1 - r/r₀)²  for r ≤ r₀ (normal)
w(r) = w₀ × (1 - r/rs)²  for r ≤ rs (superspreader)  
w(r) = 0                 for r > respective cutoff
```

## Applications

This model framework can be applied to study:
- **COVID-19 superspreading events** and intervention strategies
- **Influenza outbreaks** in confined populations
- **Social network epidemics** and information spread
- **Vector-borne diseases** with heterogeneous transmission
- **Vaccination strategy optimization** in spatially structured populations

## References
[1] R. Fujie and T. Odagaki. Effects of superspreaders in spread of epidemic. Physica
A: Statistical Mechanics and its Applications, 374(2):843–852, 2007. ISSN 0378-4371. doi:
https://doi.org/10.1016/j.physa.2006.08.050. URL https://www.sciencedirect.com/science/article/pii/S0378437106008703.

[2] M. Lipsitch, T. Cohen, B. Cooper, J. M. Robins, S. Ma, L. James, G. Gopalakrishna, S. K.
Chew, C. C. Tan, M. H. Samore, D. Fisman, and M. Murray. Transmission dynamics and
control of severe acute respiratory syndrome. Science, 300(5627):1966–1970, 2003. doi:
10.1126/science.1086616. URL https://www.science.org/doi/abs/10.1126/science.1086616.

[3] World Health Organization. Severe acute respiratory syndrome (sars). https://www.who.int/health-topics/severe-acute-respiratory-syndrome, 2003.

## **License** 
This project is licensed under the [MIT License](LICENSE).
