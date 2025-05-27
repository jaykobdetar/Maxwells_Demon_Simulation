![image](https://github.com/user-attachments/assets/b1329760-89fe-4ab9-b0c1-a2fbef681c11)
![image](https://github.com/user-attachments/assets/a41d143d-54d5-4f69-9fd4-98ca54d29c3c)
![image](https://github.com/user-attachments/assets/d9e52ac8-1f7c-4e1c-ad92-c77bc3df6ec0)
A 2D browser-based demonstration of Maxwell's Demon operating on an ideal gas. Explore thermodynamics, entropy, and the information–energy tradeoff in real time.

## Files

* **Maxwells_Demon.html**: Entry point; loads the simulation canvas and UI.
* **style.css**: Basic styling for layout and controls.
* **simulation.js**: Core physics engine (velocity sampling, collision handling, entropy & temperature calculations, demon logic).

## Features

* **Maxwell–Boltzmann Sampling**: Particles initialized with velocities drawn from a 2D Maxwell–Boltzmann distribution at the chosen temperature.
* **Elastic Collisions**: Perfectly elastic particle–particle and particle–wall collisions.
* **Entropy & Temperature**: Real-time calculation using the 2D Sackur–Tetrode formula and kinetic-energy-based temperature measurement.
* **Demon Control**: Toggle the demon on/off to see natural equilibration vs. active sorting with realistic information costs.
* **Units & Scaling**: Pixel-based units mapped to real-world nanometers and meters per second; time magnification for clear visualization.

## Getting Started

### Prerequisites

* A modern web browser (Chrome, Firefox, Safari, or Edge).

### Usage

1. Clone or download the repository.
2. Open `MaxwellsDemon.html` in your browser.
3. Use the on-screen controls to adjust temperature, toggle the demon mode, or reset the simulation.

## Physics Details

* **Velocity Initialization**: Uses Box–Muller to generate Gaussian components, normalized so σ = 1 at 300 K.
* **Temperature Computation**: In 2D, ⟨KE⟩ = kT; code converts total kinetic energy to a temperature via T = (totalKE/N)/k\_B, scaled back to Kelvin.
* **Entropy Calculation**: Implements the 2D ideal-gas Sackur–Tetrode: S = Nk\_B\[ln(A/(Nλ²)) + 1], with λ² ∝ 1/T.
* **Demon Information Cost**: The demon operates with realistic measurement inefficiencies and memory constraints. Information erasure cost is tracked as bits × k\_B ln 2, with additional inefficiencies for imperfect measurement and erasure.

## Unit System & Normalization

The simulation uses a carefully chosen normalized unit system to balance computational efficiency with physical accuracy. Here's how units are mapped:

### Normalized Units

The code uses normalized units where:
- **k_B = 1.0** (Normalized Boltzmann constant)
- **Mass = 1.0** (Normalized particle mass)
- **Reference Temperature T_REF = 300 K**

At the reference temperature, k_B·T_REF = 1 in energy units.

### Physical Mapping

The normalized units map to physical units as follows:

#### Length Scale
- **1 pixel = 10 nanometers**
- Chamber width ~800 pixels represents ~8 micrometers
- This scale is appropriate for modeling a microscopic gas chamber

#### Velocity Scale
- **VELOCITY_SCALE = 100 pixels/second**
- At 300K, this represents the thermal velocity of argon atoms
- Physical argon thermal velocity: ~400 m/s
- Mapping: 100 px/s in simulation = 400 m/s in reality

#### Time Scale
- **1 second of simulation time = 25 nanoseconds real time**
- This time magnification allows visualization of molecular motion
- Collision frequency and thermal equilibration occur on appropriate timescales

### Key Constants

The simulation defines several important constants for the 2D system:

```javascript
// Thermal de Broglie wavelength for argon at 300K
// λ_thermal = h/√(2πmkT) ≈ 1.6×10⁻¹¹ m
const LAMBDA_THERMAL_PIXELS = 0.0016; // in pixel units
const H2_OVER_2PI_MK = 2.56e-6; // px² units, used in entropy calculations
```

### Temperature and Energy

- **Temperature**: Displayed in Kelvin, calculated from average kinetic energy
- **Energy**: In 2D, ⟨KE⟩ = kT per particle (not 3kT/2 as in 3D)
- **Information cost**: k_B·T·ln(2) per bit, displayed in units of k_B·T

### Entropy Units

- Entropy is displayed in units of k_B
- The 2D Sackur-Tetrode equation uses area (in px²) instead of volume
- Thermal wavelength λ scales as 1/√T, reflecting quantum mechanical effects

### Conversion Formulas

To convert simulation values to physical units:

1. **Position**: multiply pixels by 10⁻⁸ meters
2. **Velocity**: multiply (px/s) by 4 m/s
3. **Time**: multiply simulation seconds by 25×10⁻⁹ seconds
4. **Energy**: multiply by k_B·T_REF = 4.14×10⁻²¹ Joules
5. **Entropy**: multiply by k_B = 1.38×10⁻²³ J/K

### Why These Units?

This unit system was chosen to:
- Keep numerical values in a computationally friendly range (avoiding overflow/underflow)
- Maintain physically accurate ratios between quantities
- Allow direct visualization of molecular-scale phenomena
- Preserve the correct thermodynamic relationships in 2D

The normalization ensures that at room temperature (300K), particles have unit thermal energy and unit-order velocities, making the simulation both numerically stable and physically meaningful.
