![image](https://github.com/user-attachments/assets/2f645eec-b383-40ca-a9a7-85db73a3716b)
![image](https://github.com/user-attachments/assets/89ca6f7e-7c2d-4f91-9b19-5b2c95ea6620)
![image](https://github.com/user-attachments/assets/15ea3f09-0907-4040-ba3f-9fd81ceef9b1)
Features

Maxwell–Boltzmann Sampling: Particles initialized with velocities drawn from a 2D Maxwell–Boltzmann distribution at the chosen temperature.

Elastic Collisions: Perfectly elastic particle–particle and particle–wall collisions.

Entropy & Temperature: Real-time calculation using the 2D Sackur–Tetrode formula and kinetic-energy-based temperature measurement.

Temperature Clamping: Simulation temperature is clamped between 10 K and 10,000 K to prevent non-physical extremes.

Demon Control: Toggle between an ideal (zero-cost) demon and a realistic demon with Landauer-limit erasure accounting.

Units & Scaling: Pixel-based units mapped to real-world nanometers and meters per second; time magnification for clear visualization.

Getting Started

Prerequisites

A modern web browser (Chrome, Firefox, Safari, or Edge).

Usage

Clone or download the repository.

Open index.html in your browser.

Use the on-screen controls to adjust temperature (10 K–10,000 K), toggle demon mode, or reset the simulation.

Physics Details

Velocity Initialization: Uses Box–Muller to generate Gaussian components, normalized so σ = 1 at 300 K.

Temperature Computation: In 2D, ⟨KE⟩ = kT; code converts total kinetic energy to a temperature via T = (totalKE/N)/k_B, scaled back to Kelvin.

Entropy Calculation: Implements the 2D ideal-gas Sackur–Tetrode: S = N k_B[ln(A/(Nλ²)) + 1], with λ² ∝ 1/T.

Demon Information Cost: Erasure cost is tracked as bits × k_B ln 2; ideal measurements are reversible (zero cost).

Units & Scaling

Length: 1 px = 10 nm.

Velocity: 100 px/s ≃ 400 m/s at 300 K.

Time: 1 s of simulation ≃ 25 ns real time.
