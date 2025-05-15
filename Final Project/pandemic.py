#'self' allows us to establish a variable in one method and then use it in another 
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import time

#global constants (tuneable)
g_size = 7 #grid_size (if you suqare this number, that is the population contained in our simulation)
infectious_rate = 0.4 #chance of being infected
mortality = 0.9 #chance of death
recovery_time = 100 #timesteps required for recovery (leading, susbequently, to immunity)
total_time = 500 #timesteps for the duration of the pandemic
collision_dist = 0.3 #minimum distance between two people before we they are pushed apart to prevent overlapping, which would be unrealistic
mask_percent = 0.5 #population to be masked - 50%
mask_lowered_infect_chance = 0.4 #decrease in the likelihood of infection if a person wears a mask

class person:
    def __init__(self, x=0, y=0, infected=False, alive=True, immune=False, masked=False):
        #position & velocity parameters
        self.x = x
        self.y = y
        self.vel_x, self.vel_y = np.random.randn(2) / 8#Random values for x and y velocity components
        #virus parameters
        self.infected = infected 
        self.alive = alive
        self.immune = immune
        self.masked = masked
        #attibutes for modifier method - unique for each instance of Person
        self.infected_time = 0 #counter variable for how long a Person is infected for
        self.fate = False
        self.time_of_death = None #will be used to be assigned a random time in the revoery window if the Person is fated to die (self.fate = True)
        
        """
        For the last three variables above, it should be noted that it takes care of the issue of checking a random number
        against probability for each timestep. By using 'fate' and 'time of death', we are saying that the disease has an XY%
        chance to kill over the course of the infection, not at every timestep. In the method below named 'infection,' I check 
        for ONLY when a person is infected, whether or not they will die by comparing a random number to our mortality rate. 
        If True, I then assign a random time in their recovery window (rather than at the beginning or end of the
        window - that would be rather boring and unrealistic)
        """
        
    def infection(self):
        if not self.immune and self.alive:
            self.infected = True
            self.fate = np.random.random() < mortality #checking their fate ONLY IF the person is infected (not at every timestep)
            if self.fate: #if the person is fated to die...
                self.time_of_death = np.random.randint(1, recovery_time) #random death time between the recovery duration
            else: #if the person is NOT fated to die...
                self.time_of_death = None
            return True 
        return False
            
               
    def recovery(self):
        if self.alive and self.infected:
            self.immune = True
            self.infected = False
            self.infected_time = 0

    
    def die(self):
        if self.alive:
            self.alive = False
            self.infected = False
            self.infected_time = 0
            
    def contagious(self):
        if self.alive and self.infected:
            return True
        else:
            return False          

    #Distance between two poeple
    def proximity(self, num):
        return (np.sqrt((self.x - num.x)**2 + (self.y - num.y)**2))
    
        
    def mobility(self, grid_size): 
        if not self.alive:
            return 
        #set the boundaries for movement - confined in a square
        x_min = 0
        y_min = 0
        x_max = grid_size
        y_max = grid_size
        
        #update position one time step at a time
        self.x = self.x + self.vel_x * 1  #distance = velocity * time
        self.y = self.y + self.vel_y * 1
        
        if self.x >= x_max: #checking the right side boundary
            self.x = x_max - (self.x - x_max) #correct the posiiton of x if person went on or past the right boundary
            self.vel_x = -self.vel_x #bounce the velocity to the other direction
        elif self.x <= x_min: #checking the left side boundary
            self.x = (x_min - self.x) + x_min #correct the posiiton of x if person went on or past the left boundary
            self.vel_x = -self.vel_x
            
        if self.y >= y_max:
            self.y = (self.y - y_max) + y_max
            self.vel_y = -self.vel_y
        elif self.y <= y_min:
            self.y = y_min + (y_min - self.y)
            self.vel_y = -self.vel_y
                    
               
    def modifier(self,recover):
        if self.alive and self.infected:
            self.infected_time += 1
            if self.time_of_death == self.infected_time and self.fate:
                self.die()
            elif recover <= self.infected_time:
                self.recovery()
                
                
def grid(g_size, mask_percent): #Size of the grid is variable
    population = []
    for x in range(g_size):
        for y in range(g_size):
            population.append(person(x,y))
            
    num_mask = int(mask_percent * len(population))
    masked_index = random.sample(range(len(population)), num_mask) #this ensures that the masked people are randomly distributed across the grid
    for i in masked_index:
        population[i].masked = True        
            
    return population #the variable 'population' is an appended list with instances of the Person class for every point on the grid



#interaction between person 1 and person 2 
def interaction(population):
    for i in range(len(population)):
        p1 = population[i]
        for j in range(i + 1, len(population)):
            p2 = population[j]
            if p1.proximity(p2) < 0.35:
                if p1.contagious() and not p2.immune and p2.alive:
                    p2_infection_chance = infectious_rate * mask_lowered_infect_chance if p2.masked else infectious_rate
                    
                    if np.random.random() < p2_infection_chance:
                        p2.infection()
                if p2.contagious() and not p1.immune and p2.alive:
                    p1_infection_chance = infectious_rate * mask_lowered_infect_chance if p1.masked else infectious_rate
                    if np.random.random() < p1_infection_chance:
                        p1.infection()
                    
 
"""
Below, we are now going to attmept to implement the Persons bouncing off of each other. However, unlike the way in which
we implemented the bouncing of the Persons off of the boundaires of the grid (v = -v), we will introduce some form 
of unpredictability when it comes to a Person bounding off of another Person. We can do that by introducing some randomness
in the angle that they bounce of by
"""       


def collision_bw_persons(population): #implementation starts similar to the interaction method above 
    for i in range(len(population)):
        p1 = population[i]
        if not p1.alive:
            continue
        for j in range(i + 1, len(population)):
            p2 = population[j]
            if not p2.alive:
                continue
            dist = p1.proximity(p2)
            if dist < collision_dist:
                #taking in the current angles
                angle1 = np.arctan2(p1.vel_y, p1.vel_x)
                angle2 = np.arctan2(p2.vel_y, p2.vel_x)
                
                #magnitudes of velocity
                speed1 = np.sqrt(p1.vel_x**2 + p1.vel_y**2)
                speed2 = np.sqrt(p2.vel_x**2 + p2.vel_y**2)
                
                #random angular perturbation (+/- 30 degrees) - anything more than this angle might look un-human
                perturbation = np.random.uniform(-np.pi/6, np.pi/6)
                new_angle1 = angle1 + perturbation
                new_angle2 = angle2 - perturbation  #opposite perturbation for p2
                
                #update velocities with new angles, preserving speed
                p1.vel_x = speed1 * np.cos(new_angle1)
                p1.vel_y = speed1 * np.sin(new_angle1)
                p2.vel_x = speed2 * np.cos(new_angle2)
                p2.vel_y = speed2 * np.sin(new_angle2)
                
                #resolves overlap - this moves the people apart, working in conjuction with the angle perturbation
                nx = (p2.x - p1.x) / dist #here we take the x-component of the unit vector between the two people
                ny = (p2.y - p1.y) / dist #and the y-component as well
                overlap = (collision_dist - dist) / 2 #splits it equally
                p1.x -= nx * overlap
                p1.y -= ny * overlap
                p2.x += nx * overlap
                p2.y += ny * overlap
        
                   
def simulation(steps, g_size):
    #starting the timer
    start_time = time.time() #starts timer duration of simulation
    
    # Run two simulations: no masking and 50% masking
    populations = []
    positions = []
    states = []
    counts = []
    infected_counts = []
    dead_counts = []
    
    for mask_percentage in [0.0, mask_percent]:
        #initialize population
        pop = grid(g_size, mask_percentage)
        init_infected = random.choice(range(len(pop)))
        pop[init_infected].infection()

        pop_positions = []
        pop_states = []
        pop_counts = []
        pop_infected_counts = []
        pop_dead_counts = []

        #run simulation
        for step in range(steps + 1):
            infected = 0
            healthy = 0
            dead = 0
            immune = 0

            pos = [(p.x, p.y) for p in pop]
            state = []
            for p in pop:
                if not p.infected and p.alive and not p.immune:
                    healthy += 1
                    state.append('healthy')
                elif p.infected:
                    infected += 1
                    state.append('infected')
                elif p.immune:
                    immune += 1
                    state.append('immune')
                elif not p.alive:
                    dead += 1
                    state.append('dead')

            pop_positions.append(pos)
            pop_states.append(state)
            pop_counts.append((healthy, infected, immune, dead))
            pop_infected_counts.append(infected)
            pop_dead_counts.append(dead)

            for p in pop:
                p.mobility(g_size)
            
            collision_bw_persons(pop)
            interaction(pop)
            
            for p in pop:
                p.modifier(recovery_time)
        
        populations.append(pop)
        positions.append(pop_positions)
        states.append(pop_states)
        counts.append(pop_counts)
        infected_counts.append(pop_infected_counts)
        dead_counts.append(pop_dead_counts)

    end_time = time.time()
    run_time = end_time - start_time
    
    """ 
    starting from below this comment till the end of the 'simulation' function can be commented out IF you want 
    to call the function 'plot_runtime_vs_population' at the end of this file. This is just to prevent it 
    from running the animation and the two plots below 4 times (or however many population sizes you give it), and
    instead only want to see a plot between runtime of the computation and its population size.
    """
    
    #plot infected vs. timesteps for both simulations
    plt.figure()
    plt.plot(range(steps + 1), infected_counts[0], label='Infected (0% Masking)', color='red')
    plt.plot(range(steps + 1), infected_counts[1], label='Infected (50% Masking)', color='blue')
    plt.xlabel('Timestep')
    plt.ylabel('Number of Infected')
    plt.title('Infected People Over Time')
    plt.grid(True)
    plt.legend()
    plt.show()

    #plot dead vs. timesteps for both sims
    plt.figure()
    plt.plot(range(steps + 1), dead_counts[0], label='Dead (0% Masking)', color='red')
    plt.plot(range(steps + 1), dead_counts[1], label='Dead (50% Masking)', color='blue')
    plt.xlabel('Timestep')
    plt.ylabel('Number of Dead')
    plt.title('Dead People Over Time')
    plt.grid(True)
    plt.legend()
    plt.show()

    #animation with four subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'width_ratios': [3, 2]})
    
    #scatter plot (top left: no masking)
    ax1.set_xlim(-0.5, g_size + 0.5)
    ax1.set_ylim(-0.5, g_size + 0.5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    scatter1 = ax1.scatter([], [], s=50)
    #step counter and combined counts above scatter
    label_text1 = ax1.text(2.5, g_size + 1.5, 'Step:', ha='right', va='bottom', fontsize=12)
    number_text1 = ax1.text(2.6, g_size + 1.5, '000', ha='left', va='bottom', fontsize=12, bbox=dict(facecolor='white', edgecolor='none'))
    counts_text1 = ax1.text(4.2, g_size + 1.5, 'Healthy: 0, Infected: 0, Immune: 0, Dead: 0', ha='left', va='bottom', fontsize=10, bbox=dict(facecolor='white', edgecolor='none'))

    #histogram (top right: no masking)
    categories = ['Healthy', 'Infected', 'Immune', 'Dead']
    colors = ['blue', 'red', 'green', 'black']
    bar_width = 0.2
    x = np.arange(len(categories))
    bars1 = ax2.bar(x, [0, 0, 0, 0], width=bar_width * 4, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylim(0, g_size * g_size)
    ax2.set_ylabel('Count')
    ax2.set_title('State Distribution (0% Masking)')
    ax2.grid(True, axis='y')

    #scatter plot (bottom left: 50% masking)
    ax3.set_xlim(-0.5, g_size + 0.5)
    ax3.set_ylim(-0.5, g_size + 0.5)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    scatter2 = ax3.scatter([], [], s=50)
    #step counter and combined counts above scatter
    label_text2 = ax3.text(2.5, g_size + 1.5, 'Step:', ha='right', va='bottom', fontsize=12)
    number_text2 = ax3.text(2.6, g_size + 1.5, '000', ha='left', va='bottom', fontsize=12, bbox=dict(facecolor='white', edgecolor='none'))
    counts_text2 = ax3.text(4.2, g_size + 1.5, 'Healthy: 0, Infected: 0, Immune: 0, Dead: 0', ha='left', va='bottom', fontsize=10, bbox=dict(facecolor='white', edgecolor='none'))

    #histogram (bottom right: 50% masking)
    bars2 = ax4.bar(x, [0, 0, 0, 0], width=bar_width * 4, color=colors)
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.set_ylim(0, g_size * g_size)
    ax4.set_ylabel('Count')
    ax4.set_title('State Distribution (50% Masking)')
    ax4.grid(True, axis='y')

    def update(frame):
        #update scatter plot (no masking)
        pos1 = positions[0][frame]
        state1 = states[0][frame]
        scatter_colors1 = ['blue' if s == 'healthy' else 'red' if s == 'infected' else 'green' if s == 'immune' else 'black' for s in state1]
        scatter1.set_offsets(pos1)
        scatter1.set_color(scatter_colors1)
        
        #update histogram (no masking)
        for bar, count in zip(bars1, counts[0][frame]):
            bar.set_height(count)
        
        #update step and counts (no masking)
        number_text1.set_text(f"{frame:03d}")
        counts_text1.set_text(f"Healthy: {counts[0][frame][0]}, Infected: {counts[0][frame][1]}, Immune: {counts[0][frame][2]}, Dead: {counts[0][frame][3]}")

        #update scatter plot (50% masking)
        pos2 = positions[1][frame]
        state2 = states[1][frame]
        scatter_colors2 = ['blue' if s == 'healthy' else 'red' if s == 'infected' else 'green' if s == 'immune' else 'black' for s in state2]
        scatter2.set_offsets(pos2)
        scatter2.set_color(scatter_colors2)
        
        #update histogram (50% masking)
        for bar, count in zip(bars2, counts[1][frame]):
            bar.set_height(count)
        
        #update step and counts (50% masking)
        number_text2.set_text(f"{frame:03d}")
        counts_text2.set_text(f"Healthy: {counts[1][frame][0]}, Infected: {counts[1][frame][1]}, Immune: {counts[1][frame][2]}, Dead: {counts[1][frame][3]}")
        
        return [scatter1, scatter2] + list(bars1) + list(bars2) + [number_text1, counts_text1, number_text2, counts_text2]

    ani = FuncAnimation(fig, update, frames=steps + 1, interval=100, blit=True)
    plt.tight_layout()
    plt.show()
    
    #end the timer for the simulation runtime
    return run_time


def plot_runtime_vs_population(g_sizes):
    if len(g_sizes) != 4:
        raise ValueError("g_sizes must contain exactly 4 values")
    
    population_sizes = [g * g for g in g_sizes]
    run_times = []

    for g_size in g_sizes:
        run_time = simulation(total_time, g_size)
        run_times.append(run_time)
        print(f"Animation computation completed in {run_time:.2f} seconds")

    # Plot population size vs. run time
    plt.figure()
    plt.scatter(population_sizes, run_times, color='blue', label='Run Time')
    plt.plot(population_sizes, run_times, color='blue')
    plt.xlabel('Population Size')
    plt.ylabel('Run Time (sec)')
    plt.title('Population Size vs. Runtime')
    plt.grid(True)
    plt.legend()
    plt.show()


""" 
Now let's call the functions above
"""

simulation(total_time, g_size)
# plot_runtime_vs_population([5, 8, 12, 16])

