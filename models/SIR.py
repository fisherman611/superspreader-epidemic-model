import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from scipy.interpolate import interp1d
import random

class SIRSimulation:
    def __init__(self, r0:float =1, w0: float =1, gamma: float =1, alpha: float =2):
        """Initializes parameters for a spatially structured SIR model simulation
    
        Parameters: 
        
            r0 (float): Infection cutoff distance for normal individuals.
            L (float): Simulation space size (10 * r0)
            w0 (float): Base infection probability scaling factor.
            gamma (float): Recovery probability.
            alpha (float): Exponent for distance-dependent infection probability.
            rs (float): Superspreader cutoff distance (sqrt(6) * r0) for hub model
        """
        self.r0 = r0
        self.L = 10 * r0
        self.w0 = w0
        self.gamma = gamma
        self.alpha = alpha
        self.rs = np.sqrt(6) * r0
        
    def periodic_distance(self, x1: float, y1: float, x2: float, y2: float):
        """Calculate periodic distance with boundary conditions
        
        Args: 
            x1 (float): X-coordinate of first point
            y1 (float): Y-coordinate of first point 
            x2 (float): X-coordinate of second point 
            y2 (float): Y-coordinate of second point 
            
        Returns: 
            float: Euclidean distance considering periodic boundaries of size L.
        """
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        dx = min(dx, self.L - dx)
        dy = min(dy, self.L - dy)
        return np.sqrt(dx**2 + dy**2)
        
    def infection_probability(self, r: float , is_superspreader: bool, model_type: str ="hub"): 
        """Calculate infection probability
        
        Args: 
            r (float): Distance between individuals 
            is_superspreader (bool): True if the individuals is a superspreader
            model_type (str): Type of model, "hub" or "strong_infectiousness"
        
        Returns: 
            float: Infection probability
        """
        # Strong infectiousness model
        if model_type == "strong_infectiousness":
            if r > self.r0:
                return 0 
            if is_superspreader:
                return self.w0 
            else:
                return self.w0 * (1 - r/self.r0)**2
        
        # Hub model
        elif model_type == "hub":
            if is_superspreader:
                if r > self.rs:
                    return 0
                return self.w0 * (1 - r/self.rs)**2
            else:
                if r > self.r0:
                    return 0
                return self.w0 * (1 - r/self.r0)**2
    
    def run_simulation(self, N, lambda_val, model_type='strong_infectiousness', max_steps=100, initial_pos=(0, 0)):
        """Run a single epidemic simulation
        
        Args: 
            N (int): Number of individuals
            lambda_val (float): Fraction of superspreaders
            model_type (str): Type of model "hub" or "strong_infectiousness" 
            max_steps (int): Maximum simulation steps
            initial_pos (tuple): Initial infected position
            
        Returns: 
            dict: Simulation results including positions, states, infection tree, and metrics
        """
        # Initialize individuals
        positions = np.random.uniform(0, self.L, (N, 2))
        positions[0] = initial_pos  # Patient zero
        
        # Assign superspreader status
        is_superspreader = np.random.random(N) < lambda_val
        
        # States: 0=S, 1=I, 2=R
        states = np.zeros(N, dtype=int)
        states[0] = 1  # Initial infection
        
        infection_times = np.full(N, -1)
        infection_times[0] = 0
        
        # Track infection network
        infection_tree = {}
        secondary_infections = defaultdict(int)
        
        new_infections_per_step = []
        max_distances = []
        
        for step in range(max_steps):
            new_infections = 0
            infected_indices = np.where(states == 1)[0]
            
            if len(infected_indices) == 0:
                break
                
            # Calculate maximum distance from origin
            infected_positions = positions[states > 0]
            if len(infected_positions) > 0:
                distances = [self.periodic_distance(pos[0], pos[1], initial_pos[0], initial_pos[1]) 
                           for pos in infected_positions]
                max_distances.append(max(distances))
            else:
                max_distances.append(0)
            
            # Infection process
            for infector_idx in infected_indices:
                infector_pos = positions[infector_idx]
                infector_superspreader = is_superspreader[infector_idx]
                
                for target_idx in range(N):
                    if states[target_idx] == 0:  # Susceptible
                        target_pos = positions[target_idx]
                        distance = self.periodic_distance(
                            infector_pos[0], infector_pos[1],
                            target_pos[0], target_pos[1]
                        )
                        
                        # Calculate infection probability
                        if model_type == 'strong_infectiousness':
                            prob = self.infection_probability(distance, infector_superspreader, model_type="strong_infectiousness")
                        else:  # hub
                            prob = self.infection_probability(distance, infector_superspreader, model_type="hub")
                        
                        if np.random.random() < prob:
                            states[target_idx] = 1
                            infection_times[target_idx] = step + 1
                            infection_tree[target_idx] = infector_idx
                            secondary_infections[infector_idx] += 1
                            new_infections += 1
            
            # Recovery process
            for idx in infected_indices:
                if np.random.random() < self.gamma:
                    states[idx] = 2
            
            new_infections_per_step.append(new_infections)
        
        return {
            'positions': positions,
            'is_superspreader': is_superspreader,
            'states': states,
            'infection_times': infection_times,
            'infection_tree': infection_tree,
            'secondary_infections': dict(secondary_infections),
            'new_infections_per_step': new_infections_per_step,
            'max_distances': max_distances
        }