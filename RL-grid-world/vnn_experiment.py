import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import MiniBatchKMeans
import collections
import time
import json

# ==========================================
# 1. THE MODELS
# ==========================================

class InferenceNeuron:
    """
    A 'Smart' Neuron representing a hypersphere in the feature space.
    Combines:
    - Geometry (Center, Radius)
    - Bayesian Statistics (Online Mean/Variance update)
    - Markov Chain (Next state transition probabilities)
    - Physics (Lagrange multipliers for max radius constraints)
    - Biology (Energy/Metabolism/TTL)
    """
    def __init__(self, neuron_id, center, radius, max_radius=1.5):
        self.id = neuron_id
        
        # 1. GEOMETRY (Hard Sphere Logic)
        self.c = np.array(center, dtype=np.float32)
        self.r = radius
        
        # 2. MARKOV (Transition Memory)
        # Sparse dictionary: {target_neuron_id: count}
        # Only stores existing connections to save memory vs dense matrix
        self.next_links = {} 
        self.total_activations = 1 # Denominator for probability
        
        # 3. BAYES (Internal Statistics for updates)
        self.n_samples = 1 # Number of samples seen (Weight)
        self.variance_estimate = 0.1 # Initial uncertainty
        
        # 4. LAGRANGE (Physical Constraints)
        self.max_r = max_radius # Hard physical limit on growth
        
        # 5. METABOLISM (TTL)
        self.energy = 1.0 

    def match(self, x):
        """Hard Check: Is the point x within the sphere? (Geometric Membership)"""
        dist = np.linalg.norm(x - self.c)
        return dist <= self.r

    def update_bayes(self, x):
        """
        Online Learning (Simplified Welford's algorithm).
        Updates the center (mean) based on new evidence.
        """
        self.n_samples += 1
        self.n_samples += 1
        # Bayesian natural decay / learning rate 1/N
        # FIX: "Sticky Sphere" prevention. We anchor the center more firmly.
        learning_rate = 0.1 / self.n_samples 
        
        # Move center towards new data (Moving Average)
        diff = x - self.c
        self.c += learning_rate * diff
        
        # Update radius: if point was near the edge, expand slightly (Heuristic)
        dist = np.linalg.norm(x - self.c)
        target_r = dist * 1.1 
        if target_r > self.r:
             # Smooth expansion update
             self.r = 0.9 * self.r + 0.1 * target_r

        self.constrain_lagrange()
        self.energy = 1.0 # Refill energy on activation

    def constrain_lagrange(self):
        """
        Enforces physical laws. If constraints are violated, clamp values.
        """
        # Constraint: Radius cannot exceed max_r
        if self.r > self.max_r:
            self.r = self.max_r
            # Penalty: If trying to overgrow, increase uncertainty (variance)
            self.variance_estimate += 0.01

    def update_markov(self, next_neuron_id):
        """
        Learns temporal sequence.
        If THIS neuron fired at t, and THAT neuron fired at t+1, increment link.
        """
        if next_neuron_id not in self.next_links:
            self.next_links[next_neuron_id] = 0
        self.next_links[next_neuron_id] += 1
        self.total_activations += 1

    def predict_next(self):
        """
        Uses Markov chain to predict the next likely neuron ID.
        Returns: ID of the most probable next neuron.
        """
        if not self.next_links:
            return None
        
        # Fast Argmax over dictionary
        best_next = max(self.next_links, key=self.next_links.get)
        return best_next

    def decay(self):
        """Lifecycle: Decay energy if not used. Returns True if dead."""
        self.energy -= 0.01
        return self.energy <= 0

class VolumetricBrain:
    """
    The 'Living Inference Engine' (VL-VNN).
    Manages the population of InferenceNeurons (Geometric Prototypes).
    Handles: 
    - Spatial Detection (Set Theory / Geometric Membership)
    - One-Shot Learning (Neuron Creation)
    - Volumetric Logic (Bayesian + Lagrange + Markov)
    """
    def __init__(self, init_radius=0.1, max_radius=0.3):
        self.neurons = {} # id -> InferenceNeuron
        self.next_id = 0
        self.last_active_id = None
        self.init_radius = init_radius
        self.max_radius = max_radius
        
    def process(self, x):
        """
        Main Engine Step:
        1. Match/Create Neuron (Geometric Membership)
        2. Update Bayes Center
        3. Update Markov Links
        4. Predict Next State
        5. Prune Dead Neurons
        """
        match_id = None
        closest_dist = float('inf')
        
        # 1. Search for matching neuron (Geometric Membership / Set Theory)
        # IMPROVEMENT: "Core vs Periphery" logic. 
        # Only match if point is in the "Core" (inner 80%). 
        # If in "Periphery" (outer 20%), treat as No Match to force creation (Overlap).
        for nid, neuron in self.neurons.items():
            dist = np.linalg.norm(x - neuron.c)
            # Standard Strict check: dist <= neuron.r * 0.8
            if dist <= neuron.r * 0.8: 
                match_id = nid
                break
        
        # If no match logic (OOD), create new neuron (One-Shot Learning)
        if match_id is None:
            match_id = self.next_id
            new_neuron = InferenceNeuron(match_id, x, self.init_radius, self.max_radius)
            self.neurons[match_id] = new_neuron
            self.next_id += 1
        else:
            # 2. Update existing (Bayes)
            self.neurons[match_id].update_bayes(x)
            
        current_neuron = self.neurons[match_id]
        
        # 3. Update Markov Link (Previous -> Current)
        if self.last_active_id is not None and self.last_active_id in self.neurons:
            self.neurons[self.last_active_id].update_markov(match_id)
            
        self.last_active_id = match_id
        
        # 4. Predict Next Position
        pred_id = current_neuron.predict_next()
        prediction = None
        if pred_id is not None and pred_id in self.neurons:
            prediction = self.neurons[pred_id].c.copy()
        else:
            prediction = current_neuron.c.copy()
            
        # 5. Metabolism / Pruning
        dead_ids = []
        for nid, neuron in self.neurons.items():
            if nid == match_id: continue 
            if neuron.decay():
                dead_ids.append(nid)
        
        for nid in dead_ids:
            del self.neurons[nid]
            
        return prediction

    def get_parameter_count(self):
        # Approx: Neurons * (Coords + Radius + Energy) + Links
        # Coords(2) + R(1) + E(1) + Stats(2) = 6 floats per neuron
        # Links: 2 ints (id, count) per link
        n_count = len(self.neurons)
        link_count = sum(len(n.next_links) for n in self.neurons.values())
        return n_count * 6 + link_count * 2

class StandardVNN_Baseline:
    """
    Standard VNN assumed as Static K-Means.
    No Markov recurrence (Memoryless).
    """
    def __init__(self, n_clusters=20, buffer_size=100):
        self.model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=buffer_size, n_init='auto')
        self.buffer = []
        self.buffer_size = buffer_size
        self.is_fitted = False
        
    def process(self, x, t):
        self.buffer.append(x)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
            
        # Periodic retrain (every 50 steps)
        if t % 50 == 0 and len(self.buffer) >= 20:
            self.model.partial_fit(np.array(self.buffer))
            self.is_fitted = True
            
        if not self.is_fitted:
            return x # Naive assumption
            
        # Prediction: Without Markov, best guess is the cluster centroid matching input
        # Effectively: Denoising / Vector Quantization
        cluster_idx = self.model.predict(np.array([x]))[0]
        return self.model.cluster_centers_[cluster_idx]

class MLP_Baseline:
    """
    Traditional MLP Regressor.
    Trained on sliding window to predict x(t+1) from x(t).
    Slow adaptation compared to online methods.
    """
    def __init__(self, hidden_dim=(50,50), buffer_size=200):
        self.model = MLPRegressor(hidden_layer_sizes=hidden_dim, max_iter=1, warm_start=True)
        self.X_buffer = []
        self.Y_buffer = [] # Targets (x_t+1)
        self.buffer_size = buffer_size
        self.last_x = None
        self.is_fitted = False
        
    def process(self, x, t):
        prediction = np.zeros(2)
        
        # Predict first (before knowing ground truth for this step, 
        # but here we predict x_t+1 given x_t.
        # Wait, the main loop asks for prediction of next step.
        # So we predict P(t+1) using current X(t).
        
        if self.is_fitted:
            prediction = self.model.predict(np.array([x]))[0]
        else:
            prediction = x # Fallback
            
        # Store training data for next time
        if self.last_x is not None:
            self.X_buffer.append(self.last_x)
            self.Y_buffer.append(x) # x is target for last_x
            
            if len(self.X_buffer) > self.buffer_size:
                self.X_buffer.pop(0)
                self.Y_buffer.pop(0)
        
        self.last_x = x
        
        # Retrain every 50 steps
        if t % 50 == 0 and len(self.X_buffer) > 20:
            self.model.fit(np.array(self.X_buffer), np.array(self.Y_buffer))
            self.is_fitted = True
            
        return prediction

# ==========================================
# 2. DATA GENERATION (Simulated Environment)
# ==========================================

def generate_trajectory(n_steps=2000):
    t = np.linspace(0, 4*np.pi, n_steps)
    data = []
    
    # Phase 1: Noisy Circle
    # Phase 2: Morphs to Figure-8 (Symbolic Concept Drift)
    
    for i in range(n_steps):
        noise = np.random.normal(0, 0.05, 2)
        
        if i < 1000:
            # Circle
            x = np.cos(t[i])
            y = np.sin(t[i])
        else:
            # Figure 8 (Lemniscate) using transition
            # Smooth transition factor alpha
            alpha = min(1.0, (i - 1000) / 200.0)
            
            # Circle comps
            xc = np.cos(t[i])
            yc = np.sin(t[i])
            
            # Fig8 comps
            # x = sin(t), y = sin(t)*cos(t)
            # Adjust freq for continuity
            t8 = t[i]
            x8 = np.sin(t8)
            y8 = np.sin(t8) * np.cos(t8)
            
            x = (1 - alpha) * xc + alpha * x8
            y = (1 - alpha) * yc + alpha * y8
            
        data.append(np.array([x, y]) + noise)
        
    return np.array(data)

# ==========================================
# 3. EXECUTION LOOP
# ==========================================

def run_experiment():
    print("Initializing Experiment...")
    start_time = time.perf_counter()
    data = generate_trajectory()
    steps = len(data)
    
    # Instantiate Models
    vnn = VolumetricBrain()
    baseline_std = StandardVNN_Baseline()
    baseline_mlp = MLP_Baseline()
    
    # Metrics
    preds_vnn = []
    metrics_vnn = [] 
    preds_std = []
    metrics_std = []
    preds_mlp = []
    metrics_mlp = []
    
    errors_vnn = []
    errors_std = []
    errors_mlp = []
    
    param_counts_vnn = []
    
    print(f"Running Simulation over {steps} steps...")
    
    for t in range(steps - 1):
        x_t = data[t]
        ground_truth_next = data[t+1]
        
        # 1. Inference VNN (Volumetric Brain)
        t0 = time.perf_counter()
        p_vnn = vnn.process(x_t)
        t1 = time.perf_counter()
        
        # Store speed metric
        metrics_vnn.append({'speed': t1 - t0})
        
        # 2. Baseline Standard (KMeans)
        t0 = time.perf_counter()
        p_std = baseline_std.process(x_t, t)
        t1 = time.perf_counter()
        metrics_std.append({'speed': t1 - t0})
        
        # 3. Baseline MLP
        t0 = time.perf_counter()
        p_mlp = baseline_mlp.process(x_t, t)
        t1 = time.perf_counter()
        metrics_mlp.append({'speed': t1 - t0})
        
        # Record Predictions
        preds_vnn.append(p_vnn)
        preds_std.append(p_std)
        preds_mlp.append(p_mlp)
        
        # Calculate Errors (MSE for this step)
        errors_vnn.append(np.sum((p_vnn - ground_truth_next)**2))
        errors_std.append(np.sum((p_std - ground_truth_next)**2))
        errors_mlp.append(np.sum((p_mlp - ground_truth_next)**2))
        
        # Metrics
        param_counts_vnn.append(vnn.get_parameter_count())
    
    # Cumulative Errors
    cum_err_vnn = np.cumsum(errors_vnn)
    cum_err_std = np.cumsum(errors_std)
    cum_err_mlp = np.cumsum(errors_mlp)
    
    print("Simulation Complete.")
    
    # ==========================================
    # 4. VISUALIZATION
    # ==========================================
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2)
    
    # Plot 1: Trajectory Tracking (Ground Truth vs VNN)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Trajectory Tracking: Ground Truth vs VL-VNN")
    
    # Plot Ground Truth (faded)
    gt_x = data[:-1, 0]
    gt_y = data[:-1, 1]
    ax1.plot(gt_x, gt_y, 'k--', alpha=0.3, label='Ground Truth (Trajectory)')
    
    # Plot VNN Predictions
    p_vnn_arr = np.array(preds_vnn)
    ax1.plot(p_vnn_arr[:, 0], p_vnn_arr[:, 1], 'b-', linewidth=1, alpha=0.8, label='VL-VNN Prediction')
    
    # Highlight Drift Zone
    ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax1.text(0.05, 0.05, 'Phase 1: Circle', transform=ax1.transAxes)
    ax1.text(0.55, 0.05, 'Phase 2: Fig-8', transform=ax1.transAxes)
    ax1.legend()

    # Plot 2: Brain Topology (Neurons & Links)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title(f"VL-VNN Brain Topology ({len(vnn.neurons)} Active Neurons)")
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    
    # Draw Neurons
    for nid, neuron in vnn.neurons.items():
        circle = plt.Circle(neuron.c, neuron.r, color='r', alpha=0.1 + (0.4 * neuron.energy))
        ax2.add_patch(circle)
        ax2.plot(neuron.c[0], neuron.c[1], 'r.', markersize=2)
        
        # Draw Links
        for target_id, count in neuron.next_links.items():
            if target_id in vnn.neurons:
                target_c = vnn.neurons[target_id].c
                # Line width proportional to link strength
                lw = min(2.0, count * 0.1)
                if lw > 0.2:
                    ax2.plot([neuron.c[0], target_c[0]], [neuron.c[1], target_c[1]], 
                             'g-', alpha=0.3, linewidth=lw)
    
    ax2.set_aspect('equal')

    # Plot 3: Cumulative Prediction Error
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("Cumulative Prediction Error (MSE)")
    ax3.plot(cum_err_vnn, 'b-', label='Volumetric VNN (Ours)')
    ax3.plot(cum_err_std, 'g--', label='Standard VNN (KMeans)')
    ax3.plot(cum_err_mlp, 'm--', label='MLP (Retrained)')
    ax3.set_xlabel("Time Steps")
    ax3.set_ylabel("Cumulative Error")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Active Parameter/Memory Usage
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title("Active Parameters (Memory Efficiency)")
    ax4.plot(param_counts_vnn, 'b-', label='VL-VNN Active Params')
    ax4.axhline(y=100*2 + 100*50, color='m', linestyle='--', label='MLP (Fixed ~5000)') # Approx
    ax4.set_xlabel("Time Steps")
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('vnn_experiment_results.png')
    print("Results saved to vnn_experiment_results.png")
    # plt.show() # Uncomment if running interactively

    # ==========================================
    # 5. JSON RESULTS
    # ==========================================
    
    avg_speed_vnn = np.mean([m['speed'] for m in metrics_vnn]) * 1000
    avg_speed_std = np.mean([m['speed'] for m in metrics_std]) * 1000
    avg_speed_mlp = np.mean([m['speed'] for m in metrics_mlp]) * 1000
    total_time = time.perf_counter() - start_time

    results = {
        "metrics": {
            "volumetric_vnn": {
                "final_cumulative_mse": float(cum_err_vnn[-1]),
                "avg_active_params": float(np.mean(param_counts_vnn)),
                "avg_inference_speed_ms": float(avg_speed_vnn)
            },
            "standard_vnn": {
                "final_cumulative_mse": float(cum_err_std[-1]),
                "avg_inference_speed_ms": float(avg_speed_std)
            },
            "mlp_baseline": {
                "final_cumulative_mse": float(cum_err_mlp[-1]),
                "avg_inference_speed_ms": float(avg_speed_mlp)
            }
        },
        "experiment_info": {
            "steps": steps,
            "drift_phase_start": 1000,
            "total_execution_time_sec": float(total_time)
        }
    }
    
    print("\n--- EXPERIMENT RESULTS (JSON) ---")
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    run_experiment()
