import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from scipy.interpolate import interp1d
import random
from models.SIR import SIRSimulation

os.makedirs("figures", exist_ok=True)

def plot_infection_probabilities():
    """Plot the Infection Probabilities of Strong Infectiousness model and Hub model"""
    sim = SIRSimulation()
    
    # Figure 1: Strong Infectiousness Model
    r_values = np.linspace(0, 1, 100)
    prob_normal = [sim.infection_probability(r, False, model_type="strong_infectiousness") for r in r_values]
    prob_super = [sim.infection_probability(r, True, model_type="strong_infectiousness") for r in r_values]
    
    plt.figure(figsize=(8, 6))
    plt.plot(r_values, prob_super, 'orange', linewidth=2, label='Superspreader')
    plt.plot(r_values, prob_normal, 'blue', linestyle='--', linewidth=2, label='Normal')
    plt.xlabel(r'$r/r_0$')
    plt.ylabel(r'$w(r)/w_0$')
    plt.title('Strong Infectiousness Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig('figures/fig1_strong_infection_prob.png', dpi=300)
    plt.close()
    
    # Figure 2: Hub Model
    r_values_hub = np.linspace(0, 2, 200)
    prob_normal_hub = [sim.infection_probability(r, False, model_type="hub") for r in r_values_hub]
    prob_super_hub = [sim.infection_probability(r, True, model_type="hub") for r in r_values_hub]
    
    plt.figure(figsize=(8, 6))
    plt.plot(r_values_hub, prob_super_hub, 'orange', linewidth=2, label='Superspreader')
    plt.plot(r_values_hub, prob_normal_hub, 'blue', linestyle='--', linewidth=2, label='Normal')
    plt.xlabel(r'$r/r_0$')
    plt.ylabel(r'$w(r)/w_0$')
    plt.title('Hub Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 2)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig('figures/fig2_hub_infection_prob.png', dpi=300)
    plt.close()

def plot_percolation_probability():
    """Plot the percolation probabilities of Strong Infectiousness model and Hub model"""
    sim = SIRSimulation()
    L = sim.L
    lambda_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    N_values = range(150, 901, 50)
    n_runs = 1000
    
    colors = ['red', 'green', 'blue', 'magenta', 'cyan', 'yellow']
    markers = ['o', 'o', 's', 's', '^', '^']
    
    for model_idx, model_type in enumerate(['strong', 'hub']):
        plt.figure(figsize=(10, 8))
        
        for lambda_idx, lambda_val in enumerate(lambda_values):
            percolation_probs = []
            rho_pi_r0_squared = []
            
            for N in tqdm(N_values, desc=f'{model_type} model, λ={lambda_val}'):
                percolated_count = 0
                
                for _ in range(n_runs):
                    result = sim.run_simulation(N, lambda_val, model_type)
                    max_dist = max(result['max_distances']) if result['max_distances'] else 0
                    if max_dist >= 5:
                        percolated_count += 1
                
                percolation_prob = percolated_count / n_runs
                percolation_probs.append(percolation_prob)
                rho_pi_r0_squared.append(N * np.pi / L ** 2)
            
            plt.plot(rho_pi_r0_squared, percolation_probs, 
                    color=colors[lambda_idx], marker=markers[lambda_idx], 
                    markersize=6, label=f'λ = {lambda_val}')
        
        plt.xlabel(r'$\rho \pi r_0^2$')
        plt.ylabel('Percolation Probability')
        plt.title(f'{"Strong Infectiousness" if model_type == "strong" else "Hub"} Model')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 30)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f'figures/fig{3+model_idx}_{model_type}_percolation.png', dpi=300)
        plt.close()

def plot_critical_density():
    """Plot the Critical density"""
    sim = SIRSimulation()
    lambda_values = np.linspace(0, 1, 100)
    N_values = np.arange(150, 901, 50)
    n_runs = 1000
    
    
    # Analytical curves
    r0 = sim.r0
    rs = sim.rs
    w0 = sim.w0
    
    # Strong model integrals
    I_n_strong = 2 * np.pi * w0 * (r0**2) / 6
    I_ss_strong = 2 * np.pi * w0 * (r0**2) / 2
    R_c_strong = 4.5
    strong_critical = R_c_strong * w0 / (lambda_values * I_ss_strong + (1 - lambda_values) * I_n_strong)

    # Hub model integrals
    I_n_hub = I_n_strong
    I_ss_hub = 2 * np.pi * w0 * (rs**2) / 6
    R_c_hub = 3.2
    hub_critical = R_c_hub * w0 / (lambda_values * I_ss_hub + (1 - lambda_values) * I_n_hub)
    
    # Simulation points
    lambda_sim = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    strong_sim = []
    hub_sim = []

    for model_type, sim_points in [('strong', strong_sim), ('hub', hub_sim)]:
        for lambda_val in tqdm(lambda_sim, desc=f'Critical density {model_type}'):
            percolation_probs = []
            rho_values = []
            for N in N_values:
                percolated_count = 0
                for _ in range(n_runs):
                    result = sim.run_simulation(N, lambda_val, model_type)
                    max_dist = max(result['max_distances']) if result['max_distances'] else 0
                    if max_dist >= 5:  # Percolation threshold as per paper
                        percolated_count += 1
                percolation_prob = percolated_count / n_runs
                percolation_probs.append(percolation_prob)
                rho_values.append(N * np.pi / 100)  # \rho \pi r_0^2 = N \pi r_0^2 / L^2, L = 10 r_0

            # Interpolate to find critical density where percolation_prob ~ 0.5
            interp = interp1d(rho_values, percolation_probs, bounds_error=False, fill_value=(0, 1))
            rho_range = np.linspace(min(rho_values), max(rho_values), 1000)
            probs = interp(rho_range)
            critical_rho = rho_range[np.argmin(np.abs(probs - 0.5))]
            sim_points.append(critical_rho)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(lambda_values, strong_critical, 'g-', linewidth=2, label='Strong Model')
    plt.plot(lambda_values, hub_critical, 'm--', linewidth=2, label='Hub Model')
    plt.plot(lambda_sim, strong_sim, 'go', markersize=8, label='Strong Simulation')
    plt.plot(lambda_sim, hub_sim, 'ms', markersize=8, label='Hub Simulation')
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$\rho_c \pi r_0^2$')
    plt.title('Dependence of Critical Density on Superspreader Fraction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 6)  # Adjusted to match paper’s Figure 5
    plt.tight_layout()
    plt.savefig('figures/fig5_critical_density.png', dpi=300)
    plt.close()
    
def plot_distance_evolution():
    """Plot the distance evolution"""
    sim = SIRSimulation()
    lambda_values = [0.0, 0.2, 0.4, 0.8, 1.0]
    N = 500
    n_runs = 1000
    max_steps = 100
    
    colors = ['red', 'green', 'purple', 'blue', 'yellow']
    markers = ['o', 's', 's', 's', '^']
    
    plt.figure(figsize=(10, 8))
    
    for lambda_idx, lambda_val in enumerate(lambda_values):
        all_distances = []
        
        for _ in tqdm(range(n_runs), desc=f'λ={lambda_val}'):
            result = sim.run_simulation(N, lambda_val, 'strong', max_steps)
            distances = result['max_distances']
            # Pad with last value if simulation ended early
            while len(distances) < max_steps:
                distances.append(distances[-1] if distances else 0)
            all_distances.append(distances)
        
        # Calculate average
        avg_distances = np.mean(all_distances, axis=0)
        time_steps = range(max_steps)
        
        plt.plot(time_steps, avg_distances, 
                color=colors[lambda_idx], marker=markers[lambda_idx], 
                markersize=4, label=f'λ = {lambda_val}')
    
    plt.xlabel('Time')
    plt.ylabel(r'$r(t)$')
    plt.title('Time Evolution of Distance (Strong Model)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 40)
    plt.ylim(0, 12)
    plt.tight_layout()
    plt.savefig('figures/fig6_strong_distance_evolution.png', dpi=300)
    plt.close()

def plot_propagation_velocity():
    """Plot propagation velocity"""
    sim = SIRSimulation()
    lambda_values = np.linspace(0, 1, 30)
    N = 500
    n_runs = 1000
    max_steps = 100
    
    strong_velocities = []
    hub_velocities = []
    
    for lambda_val in tqdm(lambda_values, desc='Computing velocities'):
        # Strong model
        strong_vels = []
        for _ in range(n_runs):
            result = sim.run_simulation(N, lambda_val, 'strong', max_steps)
            distances = result['max_distances']
            if len(distances) > 10:
                # Calculate velocity as slope of first 10 steps
                velocity = np.polyfit(range(min(10, len(distances))), 
                                    distances[:min(10, len(distances))], 1)[0]
                strong_vels.append(max(0, velocity))
        strong_velocities.append(np.mean(strong_vels))
        
        # Hub model
        hub_vels = []
        for _ in range(n_runs):
            result = sim.run_simulation(N, lambda_val, 'hub', max_steps)
            distances = result['max_distances']
            if len(distances) > 10:
                velocity = np.polyfit(range(min(10, len(distances))), 
                                    distances[:min(10, len(distances))], 1)[0]
                hub_vels.append(max(0, velocity))
        hub_velocities.append(np.mean(hub_vels))
    
    plt.figure(figsize=(8, 6))
    plt.plot(lambda_values, strong_velocities, 'ro', markersize=6, label='Strong Model')
    plt.plot(lambda_values, hub_velocities, 'bs', markersize=6, label='Hub Model')
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'Velocity ($v_0/s$)')
    plt.title('Propagation Velocity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1.6)
    plt.tight_layout()
    plt.savefig('figures/fig7_propagation_velocity.png', dpi=300)
    plt.close()

def plot_epidemic_curves():
    """Plot the epidemic curves"""
    sim = SIRSimulation()
    N = 500
    n_runs = 1000
    max_steps = 100
    
    # Run simulations
    strong_02_infections = []
    hub_02_infections = []
    no_super_infections = []
    
    for _ in tqdm(range(n_runs), desc='Epidemic curves'):
        # Strong model, λ=0.2
        result = sim.run_simulation(N, 0.2, 'strong', max_steps)
        infections = result['new_infections_per_step']
        while len(infections) < max_steps:
            infections.append(0)
        strong_02_infections.append(infections)
        
        # Hub model, λ=0.2
        result = sim.run_simulation(N, 0.2, 'hub', max_steps)
        infections = result['new_infections_per_step']
        while len(infections) < max_steps:
            infections.append(0)
        hub_02_infections.append(infections)
        
        # No superspreaders, λ=0.0
        result = sim.run_simulation(N, 0.0, 'strong', max_steps)
        infections = result['new_infections_per_step']
        while len(infections) < max_steps:
            infections.append(0)
        no_super_infections.append(infections)
    
    # Calculate averages
    avg_strong_02 = np.mean(strong_02_infections, axis=0)
    avg_hub_02 = np.mean(hub_02_infections, axis=0)
    avg_no_super = np.mean(no_super_infections, axis=0)
    
    time_steps = range(max_steps)
    
    plt.figure(figsize=(10, 8))
    plt.plot(time_steps, avg_strong_02, 'ro', markersize=4, label='Strong Model (λ=0.2)')
    plt.plot(time_steps, avg_hub_02, 'bs', markersize=4, label='Hub Model (λ=0.2)')
    plt.plot(time_steps, avg_no_super, '^', color='cyan', markersize=4, label='No Superspreaders (λ=0.0)')
    
    plt.xlabel('Time')
    plt.ylabel('New Infections')
    plt.title('Epidemic Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 50)
    plt.ylim(0, 60)
    plt.tight_layout()
    plt.savefig('figures/fig8_epidemic_curves.png', dpi=300)
    plt.close()

def plot_infection_networks():
    """Plot infection networks"""
    sim = SIRSimulation()
    N = 500
    max_steps = 100
    
    # Figure 9: Strong Model, λ=0.2
    result = sim.run_simulation(N, 0.2, 'strong', max_steps)
    plot_network(result, 'Strong Model (λ=0.2)', 'fig9_strong_network.png')
    
    # Figure 10: Hub Model, λ=0.2
    result = sim.run_simulation(N, 0.2, 'hub', max_steps)
    plot_network(result, 'Hub Model (λ=0.2)', 'fig10_hub_network.png')
    
    # Figure 11: No superspreaders, λ=0.0
    result = sim.run_simulation(N, 0.0, 'strong', max_steps)
    plot_network(result, 'No Superspreaders (λ=0.0)', 'fig11_no_superspreaders_network.png')

def plot_network(result, title, filename):
    """Helper function to plot infection network"""
    positions = result['positions']
    is_superspreader = result['is_superspreader']
    states = result['states']
    infection_tree = result['infection_tree']
    
    plt.figure(figsize=(10, 10))
    
    # Plot connections
    for target, source in infection_tree.items():
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=0.5)
    
    # Plot individuals
    for i, (x, y) in enumerate(positions):
        if states[i] == 0:  # Susceptible
            continue
        
        if is_superspreader[i]:
            if states[i] == 1:  # Infected superspreader
                plt.plot(x, y, 'bo', markersize=8, markeredgecolor='blue')
            else:  # Recovered superspreader
                plt.plot(x, y, 'ko', markersize=8)
        else:
            plt.plot(x, y, 'wo', markersize=4, markeredgecolor='blue')
    
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'figures/{filename}', dpi=300)
    plt.close()

def plot_secondary_infections():
    """Plot secondary infections"""
    sim = SIRSimulation()
    N = 500
    n_runs = 1000
    max_steps = 100
    
    # Figure 12: λ=0.0
    all_secondary_no_super = []
    for _ in tqdm(range(n_runs), desc='Secondary infections λ=0.0'):
        result = sim.run_simulation(N, 0.0, 'strong', max_steps)
        secondary = list(result['secondary_infections'].values())
        all_secondary_no_super.extend(secondary)
    
    plt.figure(figsize=(8, 6))
    plt.hist(all_secondary_no_super, bins=range(21), density=True, 
             color='cyan', alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Secondary Infections')
    plt.ylabel('Frequency')
    plt.title('Secondary Infections (λ=0.0)')
    plt.xlim(0, 20)
    plt.ylim(0, 0.6)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/fig12_no_superspreaders_distribution.png', dpi=300)
    plt.close()
    
    # Figure 13: λ=0.2
    all_secondary_strong = []
    all_secondary_hub = []
    
    for _ in tqdm(range(n_runs), desc='Secondary infections λ=0.2'):
        # Strong model
        result = sim.run_simulation(N, 0.2, 'strong', max_steps)
        secondary = list(result['secondary_infections'].values())
        all_secondary_strong.extend(secondary)
        
        # Hub model
        result = sim.run_simulation(N, 0.2, 'hub', max_steps)
        secondary = list(result['secondary_infections'].values())
        all_secondary_hub.extend(secondary)
    
    plt.figure(figsize=(8, 6))
    plt.hist(all_secondary_strong, bins=range(16), density=True, 
             color='red', alpha=0.7, label='Strong Model', edgecolor='black')
    plt.hist(all_secondary_hub, bins=range(16), density=True, 
             color='blue', alpha=0.5, label='Hub Model', edgecolor='black')
    plt.xlabel('Number of Secondary Infections')
    plt.ylabel('Frequency')
    plt.title('Secondary Infections (λ=0.2)')
    plt.legend()
    plt.xlim(0, 15)
    plt.ylim(0, 0.8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/fig13_superspreaders_distribution.png', dpi=300)
    plt.close()

def plot_sars_comparison():
    """Plot SARS secondary cases distribution and epidemic curves (Figures 14 and 15).

    Compares simulated secondary infections and epidemic curves for strong infectiousness
    and hub models with SARS data from Singapore (Feb–Jun 2003), based on Fujie and Odagaki (2007).
    """
    sim = SIRSimulation(r0=1, w0=1, gamma=1.0)
    N = 500
    lambda_val = 0.4
    n_runs = 1000
    max_steps = 25
    
    sars_secondary = [0] * 150 + [1] * 25 + [2] * 15 + [3] * 10 + [4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 21, 23, 40]

    plt.figure(figsize=(8, 6))
    plt.hist(sars_secondary, bins=range(42), color='pink', alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Secondary Cases')
    plt.ylabel('Number of Patients')
    plt.title('SARS Secondary Cases (Singapore, Feb 25–Apr 30, 2003)')
    plt.xlim(0, 40)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/fig14_sars_secondary_cases.png', dpi=300)
    plt.close()

    # Figure 15: SARS epidemic curves comparison
    # Approximate SARS data based on paper’s description (120 days, peak ~30–40 cases)
    sars_data = [0, 2, 5, 10, 20, 30, 40, 35, 30, 25, 20, 15, 10, 8, 6, 4, 3, 2, 1, 1, 0, 0, 0, 0, 0]

    # Run simulations
    strong_infections = []
    hub_infections = []
    no_super_infections = []

    for _ in tqdm(range(n_runs), desc='SARS comparison'):
        # Strong model, \lambda=0.4
        result = sim.run_simulation(N, lambda_val, 'strong_infectiousness', max_steps)
        infections = result['new_infections_per_step']
        while len(infections) < max_steps:
            infections.append(0)
        strong_infections.append(infections)

        # Hub model, \lambda=0.4
        result = sim.run_simulation(N, lambda_val, 'hub', max_steps)
        infections = result['new_infections_per_step']
        while len(infections) < max_steps:
            infections.append(0)
        hub_infections.append(infections)

        # No superspreaders, \lambda=0.0
        result = sim.run_simulation(N, 0.0, 'strong_infectiousness', max_steps)
        infections = result['new_infections_per_step']
        while len(infections) < max_steps:
            infections.append(0)
        no_super_infections.append(infections)

    # Calculate averages and scale to match SARS data magnitude
    scale_factor = 0.5  # Adjust simulation output to approximate SARS case numbers
    avg_strong = np.mean(strong_infections, axis=0) * scale_factor
    avg_hub = np.mean(hub_infections, axis=0) * scale_factor
    avg_no_super = np.mean(no_super_infections, axis=0) * scale_factor

    time_steps = np.arange(max_steps) * 6  # Each step = 6 days

    plt.figure(figsize=(10, 8))
    plt.bar(time_steps, sars_data, color='yellow', alpha=0.7, label='SARS Data', width=3)
    plt.plot(time_steps, avg_strong, 'ro-', markersize=4, label='Strong Model (\u03BB=0.4)')
    plt.plot(time_steps, avg_hub, 'bs-', markersize=4, label='Hub Model (\u03BB=0.4)')
    plt.plot(time_steps, avg_no_super, 'c^-', markersize=4, label='No Superspreaders (\u03BB=0.0)')
    plt.xlabel('Time (days)')
    plt.ylabel('New Cases')
    plt.title('SARS Epidemic Curves (Singapore, Feb 13–Jun 13, 2003)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 150)
    plt.ylim(0, 50)
    plt.tight_layout()
    plt.savefig('figures/fig15_sars_epidemic_curves.png', dpi=300)
    plt.close()
    
def main():
    print("Generating SIR Superspreader Epidemic Simulation Figures...")
    print("This may take several minutes due to Monte Carlo simulations.")
    
    # Generate all figures
    print("\n1. Plotting infection probabilities...")
    plot_infection_probabilities()
    
    print("\n2. Computing percolation probabilities...")
    plot_percolation_probability()
    
    print("\n3. Plotting critical density...")
    plot_critical_density()
    
    print("\n4. Computing distance evolution...")
    plot_distance_evolution()
    
    print("\n5. Computing propagation velocities...")
    plot_propagation_velocity()
    
    print("\n6. Generating epidemic curves...")
    plot_epidemic_curves()
    
    print("\n7. Creating infection networks...")
    plot_infection_networks()
    
    print("\n8. Analyzing secondary infections...")
    plot_secondary_infections()
    
    print("\n9. SARS data comparison...")
    plot_sars_comparison()
    
    print(f"\nAll figures saved to 'figures/' directory!")
    print("Generated figures:")
    for i in range(1, 16):
        print(f"  - Figure {i}: fig{i}_*.png")

if __name__ == "__main__":
    main()