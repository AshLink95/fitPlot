#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv
import re

# Predefined common fitting functions
def lorentzian(x, A, x0, gamma, c):
    """Lorentzian: A * γ² / ((x - x₀)² + γ²) + c"""
    return A * (gamma**2 / ((x - x0)**2 + gamma**2)) + c

def gaussian(x, A, mu, sigma, c):
    """Gaussian: A * exp(-((x - mu)² / (2σ²))) + c"""
    return A * np.exp(-((x - mu)**2 / (2 * sigma**2))) + c

def exponential(x, A, k, c):
    """Exponential: A * exp(k*x) + c"""
    return A * np.exp(k * x) + c

def exponential_decay(x, A, k, c):
    """Exponential Decay: A * exp(-k*x) + c"""
    return A * np.exp(-k * x) + c

def power_law(x, A, n, c):
    """Power law: A * x^n + c"""
    return A * x**n + c

def linear(x, m, b):
    """Linear: m*x + b"""
    return m * x + b

def polynomial_2(x, a, b, c):
    """Quadratic: a*x² + b*x + c"""
    return a * x**2 + b * x + c

def polynomial_3(x, a, b, c, d):
    """Cubic: a*x³ + b*x² + c*x + d"""
    return a * x**3 + b * x**2 + c * x + d

def sinusoidal(x, A, omega, phi, c):
    """Sinusoidal: A * sin(omega*x + phi) + c"""
    return A * np.sin(omega * x + phi) + c

def damped_oscillation(x, A, omega, gamma, phi, c):
    """Damped oscillation: A * exp(-gamma*x) * sin(omega*x + phi) + c"""
    return A * np.exp(-gamma * x) * np.sin(omega * x + phi) + c

def logarithmic(x, a, b, c):
    """Logarithmic: a * log(x + b) + c"""
    # Ensure x + b is always positive
    return a * np.log(np.maximum(x + b, 1e-10)) + c

def stretched_exponential(x, A, tau, beta, c):
    """Stretched Exponential: A * exp(-(x/tau)^beta) + c"""
    return A * np.exp(-((x / tau) ** beta)) + c

# Dictionary of predefined functions
PREDEFINED_FUNCTIONS = {
    '1': ('Lorentzian', lorentzian, ['A', 'x0', 'gamma', 'c']),
    '2': ('Gaussian', gaussian, ['A', 'mu', 'sigma', 'c']),
    '3': ('Exponential Growth', exponential, ['A', 'k', 'c']),
    '4': ('Exponential Decay', exponential_decay, ['A', 'k', 'c']),
    '5': ('Power Law', power_law, ['A', 'n', 'c']),
    '6': ('Linear', linear, ['m', 'b']),
    '7': ('Quadratic', polynomial_2, ['a', 'b', 'c']),
    '8': ('Cubic', polynomial_3, ['a', 'b', 'c', 'd']),
    '9': ('Sinusoidal', sinusoidal, ['A', 'omega', 'phi', 'c']),
    '10': ('Damped Oscillation', damped_oscillation, ['A', 'omega', 'gamma', 'phi', 'c']),
    '11': ('Logarithmic', logarithmic, ['a', 'b', 'c']),
    '12': ('Stretched Exponential', stretched_exponential, ['A', 'tau', 'beta', 'c']),
}

def create_custom_function(expression, param_names):
    """Create a function from a string expression"""
    def custom_func(x, *params):
        # Create a namespace with numpy functions
        namespace = {
            'x': x,
            'np': np,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'exp': np.exp,
            'log': np.log,
            'sqrt': np.sqrt,
            'abs': np.abs,
        }
        # Add parameters to namespace
        for name, value in zip(param_names, params):
            namespace[name] = value
        
        return eval(expression, namespace)
    
    return custom_func

def load_csv_data(filename):
    """Load data from CSV file. Expected format: x1, y1, x2, y2, ..."""
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        
    # Transpose to get columns
    cols = list(zip(*rows))
    
    # Determine number of datasets (pairs of columns)
    n_datasets = len(cols) // 2
    
    for i in range(n_datasets):
        x_data = np.array([float(val) for val in cols[2*i]])
        y_data = np.array([float(val) for val in cols[2*i + 1]])
        data.append((x_data, y_data))
    
    return data, n_datasets

def get_manual_data(n_points):
    """Get data manually from user input"""
    print(f"Enter {n_points} x values (space-separated):")
    x_data = np.array([float(val) for val in input().split()])
    print(f"Enter {n_points} y values (space-separated):")
    y_data = np.array([float(val) for val in input().split()])
    return x_data, y_data

def create_formula_string(func_name, param_names, params):
    """Create a formatted formula string for the legend"""
    # Format numbers nicely
    def fmt(val):
        if abs(val) < 0.001 or abs(val) > 9999:
            return f"{val:.2e}"
        else:
            return f"{val:.3f}"
    
    if 'Logarithmic' in func_name:
        a, b, c = params
        return f"{fmt(a)}·ln(x+{fmt(b)})+{fmt(c)}"
    
    elif 'Exponential Decay' in func_name:
        A, k, c = params
        return f"{fmt(A)}·e^(-{fmt(k)}x)+{fmt(c)}"
    
    elif 'Exponential Growth' in func_name:
        A, k, c = params
        return f"{fmt(A)}·e^({fmt(k)}x)+{fmt(c)}"
    
    elif 'Linear' in func_name:
        m, b = params
        return f"{fmt(m)}x+{fmt(b)}"
    
    elif 'Quadratic' in func_name:
        a, b, c = params
        return f"{fmt(a)}x²+{fmt(b)}x+{fmt(c)}"
    
    elif 'Cubic' in func_name:
        a, b, c, d = params
        return f"{fmt(a)}x³+{fmt(b)}x²+{fmt(c)}x+{fmt(d)}"
    
    elif 'Gaussian' in func_name:
        A, mu, sigma, c = params
        return f"{fmt(A)}·e^(-(x-{fmt(mu)})²/(2·{fmt(sigma)}²))+{fmt(c)}"
    
    elif 'Lorentzian' in func_name:
        A, x0, gamma, c = params
        return f"{fmt(A)}·γ²/((x-{fmt(x0)})²+γ²)+{fmt(c)}, γ={fmt(gamma)}"
    
    elif 'Sinusoidal' in func_name:
        A, omega, phi, c = params
        return f"{fmt(A)}·sin({fmt(omega)}x+{fmt(phi)})+{fmt(c)}"
    
    elif 'Damped Oscillation' in func_name:
        A, omega, gamma, phi, c = params
        return f"{fmt(A)}·e^(-{fmt(gamma)}x)·sin({fmt(omega)}x+{fmt(phi)})+{fmt(c)}"
    
    elif 'Power Law' in func_name:
        A, n, c = params
        return f"{fmt(A)}·x^{fmt(n)}+{fmt(c)}"
    
    elif 'Stretched Exponential' in func_name:
        A, tau, beta, c = params
        return f"{fmt(A)}·e^(-(x/{fmt(tau)})^{fmt(beta)})+{fmt(c)}"
    
    elif 'Custom' in func_name:
        # For custom functions, just show parameter values
        param_str = ", ".join([f"{name}={fmt(val)}" for name, val in zip(param_names, params)])
        return param_str
    
    else:
        # Default: just show parameter values
        param_str = ", ".join([f"{name}={fmt(val)}" for name, val in zip(param_names, params)])
        return param_str

def display_function_menu():
    """Display available fitting functions"""
    print("\n=== Available Fitting Functions ===")
    for key, (name, func, params) in PREDEFINED_FUNCTIONS.items():
        print(f"{key}. {func.__doc__}")
    print("13. Custom function (enter your own expression)")
    print("===================================")

def get_fitting_function(dataset_num):
    """Get the fitting function from user for a specific dataset"""
    print(f"\n--- Function for Dataset {dataset_num} ---")
    choice = input(f"\nSelect a function for dataset {dataset_num} (1-13): ").strip()
    
    if choice in PREDEFINED_FUNCTIONS:
        name, func, param_names = PREDEFINED_FUNCTIONS[choice]
        print(f"\nSelected: {name}")
        print(f"Parameters: {', '.join(param_names)}")
        return func, param_names, name
    
    elif choice == '13':
        print("\nEnter custom function:")
        print("Example: A * exp(-k*x) + c")
        print("Available: sin, cos, tan, exp, log, sqrt, abs, np (numpy)")
        expression = input("Function f(x) = ").strip()
        
        param_input = input("Enter parameter names (space-separated, e.g., A k c): ").strip()
        param_names = [p.strip() for p in param_input.split()]
        
        func = create_custom_function(expression, param_names)
        return func, param_names, f"Custom: {expression}"
    
    else:
        print("Invalid choice. Using Linear as default.")
        return PREDEFINED_FUNCTIONS['6'][1], PREDEFINED_FUNCTIONS['6'][2], PREDEFINED_FUNCTIONS['6'][0]

def get_initial_guess(param_names, x_data, y_data, func_name):
    """Get initial parameter guesses from user or use defaults"""
    print(f"\nEnter initial guesses for parameters: {', '.join(param_names)}")
    print("Press Enter to use automatic guesses")
    guess_input = input("Initial guesses (space-separated): ").strip()
    
    if guess_input:
        try:
            return [float(val) for val in guess_input.split()]
        except:
            print("Invalid input. Using automatic guesses.")
    
    # Automatic guesses
    x_min, x_max = min(x_data), max(x_data)
    x_mid = (x_max + x_min) / 2
    x_range = x_max - x_min
    y_min, y_max = min(y_data), max(y_data)
    y_base = y_min
    y_peak = y_max
    y_range = y_peak - y_base
    y_mean = np.mean(y_data)
    
    # Smart defaults based on function type
    if 'Logarithmic' in func_name:
        # For logarithmic: a * log(x + b) + c
        # Estimate slope from first and last points
        if x_max > x_min and y_max != y_min:
            # Calculate approximate derivative
            dy_dx_approx = (y_data[-1] - y_data[0]) / (x_data[-1] - x_data[0])
            # For log function, dy/dx = a/(x+b), so a ≈ dy_dx * x
            a_guess = dy_dx_approx * x_mid
        else:
            a_guess = 1.0
        b_guess = 0.01  # Very small positive to avoid log(0)
        c_guess = y_data[0] - a_guess * np.log(x_data[0] + b_guess) if x_data[0] + b_guess > 0 else y_min
        return [a_guess, b_guess, c_guess]
    
    elif 'Exponential Decay' in func_name:
        # For exponential decay: A * exp(-k*x) + c
        c_guess = y_data[-1]  # Asymptote at end
        A_guess = y_data[0] - c_guess  # Amplitude
        # Estimate decay constant from half-life
        if A_guess != 0:
            half_point = c_guess + A_guess / 2
            # Find x where y is approximately half_point
            half_idx = np.argmin(np.abs(y_data - half_point))
            x_half = x_data[half_idx]
            k_guess = np.log(2) / x_half if x_half > 0 else 0.5
        else:
            k_guess = 0.5
        return [A_guess, k_guess, c_guess]
    
    elif 'Exponential Growth' in func_name:
        # For exponential growth: A * exp(k*x) + c
        c_guess = y_data[0]
        A_guess = (y_data[-1] - c_guess) / np.exp(x_data[-1]) if x_data[-1] != 0 else 1.0
        k_guess = 0.1
        return [A_guess, k_guess, c_guess]
    
    elif 'Linear' in func_name:
        # For linear: m*x + b
        m_guess = y_range / x_range if x_range != 0 else 1.0
        b_guess = y_data[0] - m_guess * x_data[0]
        return [m_guess, b_guess]
    
    elif 'Quadratic' in func_name:
        # For quadratic: a*x² + b*x + c
        # Use three points to estimate
        a_guess = 0.0
        b_guess = y_range / x_range if x_range != 0 else 0.0
        c_guess = y_data[0]
        return [a_guess, b_guess, c_guess]
    
    elif 'Sinusoidal' in func_name or 'Damped' in func_name:
        # For sinusoidal: A * sin(omega*x + phi) + c
        A_guess = y_range / 2
        omega_guess = 2 * np.pi / x_range if x_range != 0 else 1.0
        phi_guess = 0.0
        c_guess = y_mean
        if 'Damped' in func_name:
            gamma_guess = 0.1
            return [A_guess, omega_guess, gamma_guess, phi_guess, c_guess]
        return [A_guess, omega_guess, phi_guess, c_guess]
    
    elif 'Gaussian' in func_name or 'Lorentzian' in func_name:
        # For Gaussian/Lorentzian: A * ... + c
        A_guess = y_range
        x0_guess = x_mid
        width_guess = x_range / 10 if x_range != 0 else 1.0
        c_guess = y_base
        return [A_guess, x0_guess, width_guess, c_guess]
    
    else:
        # Generic defaults
        A_guess = y_range if y_range != 0 else 1.0
        width_guess = x_range / 10 if x_range != 0 else 1.0
        n_params = len(param_names)
        defaults = [A_guess, x_mid, width_guess, y_base, 1.0]
        return defaults[:n_params] if n_params <= 5 else [1.0] * n_params

# Main program
print("=== General Curve Fitting Tool ===\n")

# Ask about CSV first
use_csv = input("Use csv? (yes/no): ").lower().strip()

datasets = []
n = 0

if use_csv in ['yes', 'y']:
    csv_file = input("Enter CSV filename: ")
    try:
        datasets, n = load_csv_data(csv_file)
        print(f"Successfully loaded {n} datasets from {csv_file}")
        for i, (x, y) in enumerate(datasets):
            print(f"  Dataset {i+1}: {len(x)} points")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        print("Falling back to manual input...")
        use_csv = 'no'

if use_csv not in ['yes', 'y']:
    # Get number of datasets for manual entry
    n = input("How many data sets are you fitting? ")
    if not n.isdigit():
        print("Invalid input. Exiting.")
        exit()
    n = int(n)
    
    # Manual data entry
    for i in range(n):
        print(f"\n--- Dataset {i+1} ---")
        x_points = input("How many points in this dataset? ")
        if not x_points.isdigit():
            print("Invalid input. Exiting.")
            exit()
        x_points = int(x_points)
        
        x_data, y_data = get_manual_data(x_points)
        datasets.append((x_data, y_data))

# Get fitting functions for each dataset
fitting_functions = []
display_function_menu()
for i in range(n):
    fit_func, param_names, func_name = get_fitting_function(i + 1)
    fitting_functions.append((fit_func, param_names, func_name))

# Get labels for datasets
data_labels = []
fct_labels = []
for i in range(n):
    data_label = input(f"\nEnter label for dataset {i+1} (or press Enter for default): ").strip()
    fct_label = input(f"\nEnter label for function {i+1} (or press Enter for default): ").strip()
    if not data_label:
        label = f"Dataset {i+1}"
    data_labels.append(data_label)
    if not fct_label:
        fct_label = f"f_{i+1}(x)="
    fct_labels.append(fct_label)

# Get axis labels
x_label = input("\nEnter x-axis label (or press Enter for 'X'): ").strip() or "X"
y_label = input("Enter y-axis label (or press Enter for 'Y'): ").strip() or "Y"
plot_title = input("Enter plot title (or press Enter for default): ").strip() or "Multi-Function Curve Fits"

# Fit and plot
colors = ['r', 'b', 'g', 'm', 'c', 'y', 'k', 'orange', 'purple', 'brown']
plt.figure(figsize=(12, 7))

fitted_params = []

for i, (x_data, y_data) in enumerate(datasets):
    color = colors[i % len(colors)]
    fit_func, param_names, func_name = fitting_functions[i]
    
    print(f"\n{'='*60}")
    print(f"Fitting {data_labels[i]} with {func_name}")
    print('='*60)
    
    # Get initial guess
    p0 = get_initial_guess(param_names, x_data, y_data, func_name)
    print(f"Using initial guesses: {dict(zip(param_names, p0))}")
    
    try:
        # Fit the data with multiple attempts if needed
        attempts = [
            (p0, 50000),  # First attempt with smart guesses
            ([1.0] * len(param_names), 50000),  # Second attempt with all 1.0
            ([0.1] * len(param_names), 100000),  # Third attempt with small values
        ]
        
        fit_successful = False
        last_error = None
        
        for attempt_num, (initial_guess, max_iter) in enumerate(attempts):
            try:
                if attempt_num > 0:
                    print(f"  Retry attempt {attempt_num} with different initial guesses...")
                params, pcov = curve_fit(fit_func, x_data, y_data, p0=initial_guess, maxfev=max_iter)
                fit_successful = True
                break
            except Exception as e:
                last_error = e
                continue
        
        if not fit_successful:
            raise last_error
        
        # Verify the fit actually worked by checking if parameters are reasonable
        if np.any(np.isnan(params)) or np.any(np.isinf(params)):
            raise ValueError("Fit produced invalid parameters (NaN or Inf)")
        
        # Generate smooth curve
        x_fit = np.linspace(min(x_data), max(x_data), 400)
        y_fit = fit_func(x_fit, *params)
        
        # Create formula string for legend
        formula = create_formula_string(func_name, param_names, params)
        
        # Plot
        plt.plot(x_data, y_data, color + 'o', label=f'{data_labels[i]} Data', markersize=6, alpha=0.7)
        plt.plot(x_fit, y_fit, color + '-', label=f'{fct_labels[i]}: {formula}', linewidth=2)
        
        fitted_params.append(params)
        
        # Print fit parameters
        print(f"\n✓ Fit successful for {data_labels[i]}:")
        print(f"  Function: {func_name}")
        for param_name, param_value in zip(param_names, params):
            print(f"  {param_name} = {param_value:.6f}")
        
        # Calculate R-squared
        residuals = y_data - fit_func(x_data, *params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        print(f"  R² = {r_squared:.6f}")
        
        # Warn if fit is poor
        if r_squared < 0.8:
            print(f"  ⚠ Warning: Low R² value. Consider trying a different function or initial guesses.")
        
        # Calculate standard errors
        perr = np.sqrt(np.diag(pcov))
        print(f"  Standard errors:")
        for param_name, error in zip(param_names, perr):
            print(f"    σ_{param_name} = {error:.6f}")
        
    except Exception as e:
        print(f"\n✗ Error fitting {data_labels[i]}: {e}")
        print("Try adjusting initial guesses or using a different function.")

plt.xlabel(x_label, fontsize=12)
plt.ylabel(y_label, fontsize=12)
plt.title(plot_title, fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=9, loc='best')
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("=== Fitting Complete ===")
print("="*60)
