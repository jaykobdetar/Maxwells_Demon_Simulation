![image](https://github.com/user-attachments/assets/b1329760-89fe-4ab9-b0c1-a2fbef681c11)
![image](https://github.com/user-attachments/assets/68733c7e-7b4d-4450-82e0-2c4c73be7ff4)
![image](https://github.com/user-attachments/assets/d9e52ac8-1f7c-4e1c-ad92-c77bc3df6ec0)



A 2D browser-based demonstration of Maxwell’s Demon operating on an ideal gas. Explore thermodynamics, entropy, and the information–energy tradeoff in real time.

## Files

* **Maxwells_Demon.html**: Entry point; loads the simulation canvas and UI.
* **style.css**: Basic styling for layout and controls.
* **simulation.js**: Core physics engine (velocity sampling, collision handling, entropy & temperature calculations, demon logic).

## Features

* **Maxwell–Boltzmann Sampling**: Particles initialized with velocities drawn from a 2D Maxwell–Boltzmann distribution at the chosen temperature.
* **Elastic Collisions**: Perfectly elastic particle–particle and particle–wall collisions.
* **Entropy & Temperature**: Real-time calculation using the 2D Sackur–Tetrode formula and kinetic-energy-based temperature measurement.
* **Demon Control**: Toggle between an ideal (zero-cost) demon and a realistic demon with Landauer-limit erasure accounting.
* **Units & Scaling**: Pixel-based units mapped to real-world nanometers and meters per second; time magnification for clear visualization.

## Getting Started

### Prerequisites

* A modern web browser (Chrome, Firefox, Safari, or Edge).

### Usage

1. Clone or download the repository.
2. Open `MaxwellsDemon.html` in your browser.
3. Use the on-screen controls to adjust temperature, toggle the demon mode, or reset the simulation.

## Physics Details

* **Velocity Initialization**: Uses Box–Muller to generate Gaussian components, normalized so σ = 1 at 300 K.
* **Temperature Computation**: In 2D, ⟨KE⟩ = kT; code converts total kinetic energy to a temperature via T = (totalKE/N)/k\_B, scaled back to Kelvin.
* **Entropy Calculation**: Implements the 2D ideal-gas Sackur–Tetrode: S = Nk\_B\[ln(A/(Nλ²)) + 1], with λ² ∝ 1/T.
* **Demon Information Cost**: Erasure cost is tracked as bits × k\_B ln 2; ideal measurements are reversible (zero cost).

## Units & Scaling

* **Length**: 1 px = 10 nm.
* **Velocity**: 100 px/s ≃ 400 m/s at 300 K.
* **Time**: 1 s of simulation ≃ 25 ns real time.

