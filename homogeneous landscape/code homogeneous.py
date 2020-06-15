import random as rnd
import tkinter as tk
import numpy as np
import math as math
import matplotlib.pyplot as plt
from nlmpy import nlmpy
from PIL import Image
import csv

class Visual:
    '''This class arranges the visual output.'''
    def __init__(self, max_x, max_y):
        '''Initialize the visual class'''
        self.zoom = 10
        self.max_x = max_x
        self.max_y = max_y
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root,
                                width=self.max_x * self.zoom,
                                height=self.max_y * self.zoom)  
        self.canvas.pack()
        self.canvas.config(background='white')
        self.squares = np.empty((self.max_x, self.max_y), dtype=object)
        self.initialize_squares()

    def create_individual(self, x, y):
        '''Creates circle for individual'''
        color = "black"
        radius = 0.15
        return self.canvas.create_oval((x - radius) * self.zoom,
                                       (y - radius) * self.zoom,
                                       (x + radius) * self.zoom,
                                       (y + radius) * self.zoom,
                                       outline=color,
                                       fill=color)

    def move_drawing(self, drawing, x, y):
        radius = 0.15
        self.canvas.coords(drawing, (x - radius) * self.zoom,
                           (y - radius) * self.zoom,
                           (x + radius) * self.zoom,
                           (y + radius) * self.zoom)

    def color_square(self, resources, x, y):
        '''Changes the color of the square'''
        color = np.clip(resources / 100, 0, 1)
        green = int(255 * color)
        red = 255 - green
        blue = 0
        rgb = red, green, blue
        hex_code = '#%02x%02x%02x' % rgb
        self.canvas.itemconfigure(self.squares[x, y], fill=str(hex_code))

    def initialize_squares(self):
        '''returns a square (drawing object)'''
        for x in range(self.max_x):
            for y in range(self.max_y):
                self.squares[x, y] = self.canvas.create_rectangle(self.zoom * x,
                                                                  self.zoom * y,
                                                                  self.zoom * x + self.zoom,
                                                                  self.zoom * y + self.zoom,
                                                                  outline='black',
                                                                  fill='black')

class Individual:
    ''' class that regulates living individuals and their properties'''
    def __init__(self, x, y, mass, step_mean, diversion, angle, meta):
        '''initialisation'''
        self.x = x
        self.y = y
        self.age = 0
        self.mass = mass
        self.metabolic_cost = (0.14 * self.mass ** 0.751) * (3600 - 150)  
        self.time_move = 150  
        self.velocity = 0.30 * self.mass ** 0.29
        self.movement_cost = 0.17 * self.mass ** 0.75 + 3.4 * self.mass  
        self.transport_cost = cell_size * (0.56 * mass**0.46 + 11.3 * mass**0.72)  
        self.ingestion = (2 * self.mass ** 0.8) * (3600 - self.time_move)
        self.hourly_cost = self.metabolic_cost + (10 * self.transport_cost) 
        self.diversion = diversion  
        self.step_mean = step_mean  
        self.angle = rnd.uniform(0, 2 * math.pi)  
        self.resources = self.hourly_cost
        self.reproductive_age = rnd.randint(336, 504)
        self.cost = (0.158 * mass ** 0.92 * (7 * 10 ** 6)) / 15 + self.hourly_cost
        self.alive = True

        if meta.movie:
            self.drawing = meta.visual.create_individual(x, y)

    def move(self):
        '''relocates individual'''
        step = np.random.poisson(self.step_mean)  
        self.resources -= self.metabolic_cost
        
        if self.resources <= 0:
            self.die()
        else:
            self.angle += rnd.uniform(-self.diversion, self.diversion)
            future_pos_x = self.x + step * math.cos(self.angle)
            self.y += step * math.sin(self.angle)
            meta.list_x.remove(self.x)

            self.x = future_pos_x if 0 <= future_pos_x < meta.max_x else 0 - future_pos_x if future_pos_x < 0 \
                else max_x - (future_pos_x - meta.max_x)
            self.y %= max_y
            self.resources -= self.transport_cost * step
            meta.list_x.append(self.x)
            if self.resources <= 0:
                self.die()
            if meta.movie:
                meta.visual.move_drawing(self.drawing, self.x, self.y)

    def die(self):
        '''Removes individual from the population'''
        self.alive = False
        meta.list_diversion.remove(self.diversion)
        meta.list_step.remove(self.step_mean)
        meta.list_x.remove(self.x)
        if meta.movie:
            meta.visual.canvas.delete(self.drawing)

    def reproduction(self):
        ''' Introduces new individuals through reproduction'''
        if self.resources // self.cost >= 0:
            for young in range(np.random.poisson(self.resources // self.cost)):
                step_mean = rnd.uniform(step_min, step_max) if np.random.rand() < mutation_rate else self.step_mean
                diversion = rnd.uniform(div_min, div_max) if np.random.rand() < mutation_rate else self.diversion
                angle = rnd.uniform(0, 2 * math.pi)
                meta.population.append(Individual(self.x, self.y, self.mass, step_mean, diversion, angle, meta))
                meta.list_diversion.append(diversion)
                meta.list_step.append(step_mean)
                meta.list_x.append(self.x)
        
        self.die()

class Metapopulation:
    '''contains the whole population and regulates the daily life'''
    def __init__(self, max_x, max_y):
        '''initialisation'''
        self.population = []
        self.max_x = max_x
        self.max_y = max_y
        p = suitable_habitat 
        n0 = 1 - p
        continuous_env = nlmpy.mpd(nRow=self.max_y, nCol=self.max_x, h=autocorrelation)
        self.environment = carrying_capacity * nlmpy.classifyArray(continuous_env, [n0, p])
        self.pop_size = []
        self.list_step = []
        self.list_diversion = []
        self.list_resources = []
        self.mean_step = []
        self.mean_diversion = []
        self.mean_resources = []
        self.mean_regrowth = []
        self.list_regrowth = []
        self.list_x = []
        self.mean_x = []

        self.movie = False
        if self.movie:
            self.visual = Visual(self.max_x, self.max_y)

        for x_coord in range(self.max_x):
            for y_coord in range(self.max_y):
                if self.environment[x_coord][y_coord] > 0:
                    self.list_resources.append([x_coord, y_coord])

        self.initialize_pop()

    def initialize_pop(self):
        ''' initialize the population'''
        start_pop = start_population

        for n in range(start_pop):
            coordinates = rnd.choice(self.list_resources)
            x_start = coordinates[0]
            y_start = coordinates[1]
            step_mean = rnd.uniform(step_min, step_max)
            diversion = rnd.uniform(div_min, div_max)
            angle = rnd.uniform(0, 2 * math.pi)
            self.population.append(Individual(x_start, y_start, mass, step_mean, diversion, angle, self))
            self.list_step.append(step_mean)
            self.list_diversion.append(diversion)
            self.list_x.append(x_start)

    def a_day_in_the_life(self):
        ''' An hour in the life an individual'''
        rnd.shuffle(self.population)
        old_pop = self.population[:] 
        self.population.clear()
        for ind in old_pop:
            ind.move()
            if ind.alive:
                if 0 <= ind.x < max_x:
                    resources_position = self.environment[int(ind.x)][int(ind.y)] # the resources at the position x, y
                    ind.resources += min(resources_position, ind.ingestion)
                    self.environment[int(ind.x)][int(ind.y)] -= min(resources_position, ind.ingestion)
                # if the individual is of reproducible age it will produce offspring
                    if ind.reproductive_age <= ind.age:
                        ind.reproduction()
                    else:
                        ind.age += 1
                        self.population.append(ind)
                else:
                    ind.die()


        if self.movie:
            for x in range(self.max_x):
                for y in range(self.max_y):
                    self.visual.color_square(self.environment[x, y], x, y)

        # regrowth of resources
        for patch in self.list_resources:
            self.list_regrowth.append(regrowth * (1 - self.environment[patch[0]][patch[1]] / carrying_capacity))
            self.environment[patch[0]][patch[1]] += regrowth * (1 - self.environment[patch[0]][patch[1]] / carrying_capacity)


        np.clip(self.environment, 0, carrying_capacity, out=self.environment)
        self.pop_size.append(len(self.population))
        self.mean_resources.append(np.mean(self.environment))
        self.mean_step.append(np.mean(self.list_step))
        self.mean_diversion.append(np.mean(self.list_diversion))
        self.mean_regrowth.append(np.mean(self.list_regrowth))
        self.mean_x.append(np.mean(self.list_x))
        self.list_regrowth.clear()


        if self.movie:
            self.visual.canvas.update()


mutation_rate = 0.01
generations = 30000
cell_size = 0.25 #m
mass = 0.001
list_h = [0, 0.5, 1]
list_p = [0.05, 0.2, 0.5, 0.9]
regrowth = 15 * ((2 * mass ** 0.8) * (3600 - 150))
carrying_capacity = 20 * ((2 * mass ** 0.8) * (3600 - 150))
timer_list = range(generations)
color_list = ["yellow", "red", "green", "blue", "black", "cyan", "magenta", "brown", "orange"]
max_x = 10
max_y = 10
runs = 9
start_population = 10000
step_min, step_max = 0, 10
div_min, div_max = 0, math.pi


for autocorrelation in list_h:
    for suitable_habitat in list_p:

        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharex=True, figsize=(8, 8))
        fig1, axises1 = plt.subplots(3, 3, figsize=(8, 8))
        fig2, axises2 = plt.subplots(3, 3, figsize=(8, 8))
        fig6, axises6 = plt.subplots(3, 3, figsize=(8, 8))
        fig3, axises3 = plt.subplots(3, 3, figsize=(8, 8))
        fig4, axises4 = plt.subplots(3, 3, figsize=(8, 8))
        fig5, axises5 = plt.subplots(3, 3, figsize=(8, 8))
        ax1.set_title('Population size')
        ax2.set_title('Mean amount of resources (J)')
        ax3.set_title('Mean divergence (rad)')
        ax4.set_title('Mean step length (grid cells)')
        ax5.set_title('Mean regrowth')
        ax6.set_title('Mean x-coordinate')

        for simulation in range(runs):
            meta = Metapopulation(max_x, max_y)
            axises1[simulation // 3, simulation % 3].hist(meta.list_diversion)
            axises2[simulation // 3, simulation % 3].hist(meta.list_step)
            axises3[simulation // 3, simulation % 3].hist(meta.list_x)
            for timer in range(generations):
                meta.a_day_in_the_life()
                print(timer)
                print(len(meta.population))

                if timer == generations - 1:
                    if meta.pop_size[-1] > 100:
                        sample_pop = []
                        copy_pop = meta.population[:]
                        for i in range(100):
                            sample = rnd.choice(copy_pop)
                            sample_pop.append(sample)
                            copy_pop.remove(sample)
                    else:
                        sample_pop = meta.population[:]
                    ind_nr = 1
                    for i in sample_pop:

                        with open(f'sensitivity analysis landscape runs random walk 2 (p = {list_p[-1]}).csv', 'a', newline='') as ind:
                            iwriter = csv.writer(ind)
                            iwriter.writerow([f'{mass}', f'{max_x} x {max_y}', f'{autocorrelation}', f'{suitable_habitat}', f'{simulation}',
                                              f'{ind_nr}', f'{i.step_mean}', f'{i.diversion}', f'{i.x}'])
                        ind_nr += 1

            with open("sensitivity analysis landscape random walk.csv", 'a', newline='') as f:
                fwriter = csv.writer(f)
                fwriter.writerow([f'{mass}', f'{max_x} x {max_y}', f'{autocorrelation}', f'{suitable_habitat}',
                                  f'{simulation}', f'{meta.mean_step[-1]}', f'{meta.mean_diversion[-1]}',
                                  f'{meta.pop_size[-1]}', f'{meta.mean_resources[-1]}', f'{meta.mean_regrowth[-1]}',
                                  f'{meta.mean_x[-1]}'])

            ax1.plot(timer_list, meta.pop_size, color_list[simulation], animated=False)
            ax2.plot(timer_list, meta.mean_resources, color_list[simulation], animated=False)
            ax3.plot(timer_list, meta.mean_diversion, color_list[simulation], animated=False)
            ax4.plot(timer_list, meta.mean_step, color_list[simulation], animated=False)
            ax5.plot(timer_list, meta.mean_regrowth, color_list[simulation], animated=False)
            ax6.plot(timer_list, meta.mean_x, color_list[simulation], animated=False)
            axises4[simulation // 3, simulation % 3].hist(meta.list_diversion)
            axises5[simulation // 3, simulation % 3].hist(meta.list_step)
            axises6[simulation // 3, simulation % 3].hist(meta.list_x)


        fig.legend(labels=[f"Run {i}" for i in range(runs)], loc="lower center", ncol=5)
        fig.savefig(f'sensitivity random walk (dimensions = {max_x}, mass = 1 g, p = {suitable_habitat} and h = {autocorrelation}).png', dpi=200)
        fig1.suptitle('Distribution of divergence at time = 0', size=20)
        fig2.suptitle('Distribution of step length at time = 0', size=20)
        fig3.suptitle('Distribution of x-coordinates at time = 0', size=20)
        fig4.suptitle(f'Distribution of divergence at time = {timer}', size=20)
        fig5.suptitle(f'Distribution of step length at time = {timer}', size=20)
        fig6.suptitle(f'Distribution of x-coordinates at time = {timer}', size=20)

        fig1.savefig('fig1.png', dpi=200)
        fig2.savefig('fig2.png', dpi=200)
        fig3.savefig('fig3.png', dpi=200)
        fig4.savefig('fig4.png', dpi=200)
        fig5.savefig('fig5.png', dpi=200)
        fig6.savefig('fig6.png', dpi=200)
        files = [Image.open(pic) for pic in [f'fig{i}.png' for i in range(1, 7)]]
        w, h = 400, 400
        result = Image.new("RGB", (w * 2, h * 3))

        for index, file in enumerate(files):
            file.thumbnail((w, h), Image.ANTIALIAS)
            x = index // 3 * w
            y = index % 3 * h
            result.paste(file, (x, y, x + w, y + h))
        result.save(f'sensitivity allele distribution random walk (dimensions = {max_x}, mass = 1 g, p = {suitable_habitat} and h = {autocorrelation}).png', dpi=(300, 300))

if meta.movie:
    tk.mainloop()
