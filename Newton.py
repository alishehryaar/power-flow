import numpy as np

# Bus data [V_nominal=1.0 pu, Angle_initial=0 radians]
# P_spec = Net Active Power (Generation - Load)
# Q_spec = Net Reactive Power (Generation - Load)
BUSSES = {
    'Bus1': {'V_nom': 1.06, 'Angle_init': 0.0, 'type': 'slack', 'P_spec': 0.0, 'Q_spec': 0.0},
    'Bus2': {'V_nom': 1.00, 'Angle_init': 0.0, 'type': 'pv',    'P_spec': 0.20, 'Q_spec': 0.0},
    'Bus3': {'V_nom': 1.00, 'Angle_init': 0.0, 'type': 'pq',    'P_spec': -0.45, 'Q_spec': -0.15},
    'Bus4': {'V_nom': 1.00, 'Angle_init': 0.0, 'type': 'pq',    'P_spec': -0.40, 'Q_spec': -0.05},
    'Bus5': {'V_nom': 1.00, 'Angle_init': 0.0, 'type': 'pq',    'P_spec': -0.60, 'Q_spec': -0.10}
}

# Transmission line data (Resistance R and Reactance X in p.u.)
LINES = [
    {'from': 'Bus1', 'to': 'Bus2', 'R': 0.02, 'X': 0.06},
    {'from': 'Bus1', 'to': 'Bus3', 'R': 0.08, 'X': 0.24},
    {'from': 'Bus2', 'to': 'Bus3', 'R': 0.06, 'X': 0.18},
    {'from': 'Bus2', 'to': 'Bus4', 'R': 0.06, 'X': 0.18},
    {'from': 'Bus2', 'to': 'Bus5', 'R': 0.04, 'X': 0.12},
    {'from': 'Bus3', 'to': 'Bus4', 'R': 0.01, 'X': 0.03},
    {'from': 'Bus4', 'to': 'Bus5', 'R': 0.08, 'X': 0.24}
]

def calculate_ybus(lines, n_buses, name_to_index):
    Ybus = np.zeros((n_buses, n_buses), dtype=complex)
    
    for line in lines:
        i = name_to_index[line['from']]
        j = name_to_index[line['to']]
        
        # Series admittance y = 1 / (R + jX)
        y_series = 1.0 / complex(line['R'], line['X'])
        
        # Off-diagonal elements
        Ybus[i, j] -= y_series
        Ybus[j, i] -= y_series
        
        # Diagonal elements (sum of connected admittances)
        Ybus[i, i] += y_series
        Ybus[j, j] += y_series
        
    return Ybus

def run_newton_raphson():
    bus_names = list(BUSSES.keys())
    n_buses = len(bus_names)
    name_to_index = {name: i for i, name in enumerate(bus_names)}
    
    # 1. Build Y-bus
    Ybus = calculate_ybus(LINES, n_buses, name_to_index)
    G = np.real(Ybus)
    B = np.imag(Ybus)

    # 2. Initialize State Vectors
    V = np.array([BUSSES[b]['V_nom'] for b in bus_names])
    theta = np.zeros(n_buses)
    
    P_spec = np.array([BUSSES[b]['P_spec'] for b in bus_names])
    Q_spec = np.array([BUSSES[b]['Q_spec'] for b in bus_names])

    # Identify bus indices by type
    slack_idx = [0]
    pv_idx = [1]
    pq_idx = [2, 3, 4]
    
    # Indices for equations
    # dP equations for all except Slack (buses 1, 2, 3, 4)
    # dQ equations for PQ buses only (buses 2, 3, 4)
    ang_indices = pv_idx + pq_idx  
    vol_indices = pq_idx           

    MAX_ITER = 10
    TOLERANCE = 1e-5
    
    print("--- Starting Newton-Raphson ---")
    
    for k in range(MAX_ITER):
        # 3. Calculate P_calc and Q_calc for current iteration
        P_calc = np.zeros(n_buses)
        Q_calc = np.zeros(n_buses)
        
        for i in range(n_buses):
            for j in range(n_buses):
                delta_theta = theta[i] - theta[j]
                P_calc[i] += V[i] * V[j] * (G[i, j] * np.cos(delta_theta) + B[i, j] * np.sin(delta_theta))
                Q_calc[i] += V[i] * V[j] * (G[i, j] * np.sin(delta_theta) - B[i, j] * np.cos(delta_theta))

        # 4. Calculate Mismatches
        dP = P_spec[ang_indices] - P_calc[ang_indices]
        dQ = Q_spec[vol_indices] - Q_calc[vol_indices]
        mismatch = np.concatenate((dP, dQ))
        
        max_mismatch = np.max(np.abs(mismatch))
        print(f"Iter {k+1} | Max Mismatch: {max_mismatch:.6f}")
        
        if max_mismatch < TOLERANCE:
            print("Converged!")
            break

        # 5. Build 7x7 Jacobian Matrix dynamically
        # J11 = dP/dTheta (4x4) | J12 = dP/dV (4x3)
        # J21 = dQ/dTheta (3x4) | J22 = dQ/dV (3x3)
        J11 = np.zeros((len(ang_indices), len(ang_indices)))
        J12 = np.zeros((len(ang_indices), len(vol_indices)))
        J21 = np.zeros((len(vol_indices), len(ang_indices)))
        J22 = np.zeros((len(vol_indices), len(vol_indices)))

        for row, i in enumerate(ang_indices):
            for col, j in enumerate(ang_indices):
                if i == j:
                    J11[row, col] = -Q_calc[i] - (V[i]**2) * B[i, i]
                else:
                    delta_th = theta[i] - theta[j]
                    J11[row, col] = V[i] * V[j] * (G[i, j] * np.sin(delta_th) - B[i, j] * np.cos(delta_th))
            
            for col, j in enumerate(vol_indices):
                if i == j:
                    J12[row, col] = (P_calc[i] / V[i]) + V[i] * G[i, i]
                else:
                    delta_th = theta[i] - theta[j]
                    J12[row, col] = V[i] * (G[i, j] * np.cos(delta_th) + B[i, j] * np.sin(delta_th))

        for row, i in enumerate(vol_indices):
            for col, j in enumerate(ang_indices):
                if i == j:
                    J21[row, col] = P_calc[i] - (V[i]**2) * G[i, i]
                else:
                    delta_th = theta[i] - theta[j]
                    J21[row, col] = -V[i] * V[j] * (G[i, j] * np.cos(delta_th) + B[i, j] * np.sin(delta_th))
            
            for col, j in enumerate(vol_indices):
                if i == j:
                    J22[row, col] = (Q_calc[i] / V[i]) - V[i] * B[i, i]
                else:
                    delta_th = theta[i] - theta[j]
                    J22[row, col] = V[i] * (G[i, j] * np.sin(delta_th) - B[i, j] * np.cos(delta_th))

        # Assemble the full 7x7 Jacobian
        J = np.vstack((np.hstack((J11, J12)), 
                       np.hstack((J21, J22))))

        # 6. Solve the Linear System: J * deltaX = mismatch
        delta_x = np.linalg.solve(J, mismatch)

        # 7. Update State Vectors
        dTheta = delta_x[:len(ang_indices)]
        dV = delta_x[len(ang_indices):]

        theta[ang_indices] += dTheta
        V[vol_indices] += dV

    print("\n--- Final Results ---")
    for i, name in enumerate(bus_names):
        print(f"{name} | V = {V[i]:.4f} p.u. | Angle = {np.degrees(theta[i]):.4f} deg")

if __name__ == "__main__":
    run_newton_raphson()