# Pandemic Simulation

This is a project that I worked on during my senior year which simulates a pandemic in real time with an animation. I have various constraints that I have outlined in my report and try to predict/track mortality and infectious rates based on tuneable parameters of any given virus and the population that it infects.


## Features

- Simulates individuals moving and interacting in a confined space
- Models infection spread, recovery, immunity, and death
- Compares scenarios with and without mask usage
- Visualizes results with animated scatter plots and statistical graphs

## Requirements

- Python 3.7+
- `numpy`
- `matplotlib`

Install dependencies with:
```bash
pip install numpy matplotlib
```

## Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Pandemic-Simulation.git
   cd Pandemic-Simulation/Final\ Project
   ```

2. **Run the simulation:**
   ```bash
   python pandemic.py
   ```

   This will run the simulation with default parameters and display the animation and plots.

3. **Adjust parameters:**
   - You can change simulation parameters (population size, infection rate, etc.) at the top of `pandemic.py`:
     ```python
     g_size = 15  # Grid size (population = g_size^2)
     infectious_rate = 0.4
     mortality = 0.9
     recovery_time = 100
     total_time = 500
     mask_percent = 0.5
     ```

4. **Compare runtimes:**
   - To compare simulation runtimes for different population sizes, uncomment the following line at the end of `pandemic.py`:
     ```python
     # plot_runtime_vs_population([5, 8, 12, 16])
     ```

## How it Works

- Each individual is represented as a moving point in a grid.
- Infection spreads when an infected and a healthy individual come close.
- Mask usage reduces infection probability.
- Individuals recover or die based on probabilities and become immune if they recover.
- The simulation visualizes the number of healthy, infected, immune, and dead individuals over time.

## Output

- **Animated scatter plots** showing the state of each individual at each timestep.
- **Line plots** for infected and dead counts over time.
- **Bar charts** for state distribution at each timestep.

## Customization

- Modify parameters at the top of `pandemic.py` to explore different scenarios.
- You can add more features or tweak the logic for research or educational purposes.


---

*Created for educational purposes. For questions or contributions, please open an issue or pull request.*