import numpy as np
import scipy.special as sp
import sys
import random
import pygame
import pygame.locals
from math import *
import matplotlib.pyplot as plt

# INIT variables
eps = sys.float_info.epsilon
timestep = 100
timestep_in_sec = timestep / 1000
windowSize = 300
line_thiccness = 5
iterations = 500
spawn_x = 40  # int(windowSize/2)
spawn_y = 40  # int(windowSize/2)

# 0: Robot Color, 1: Direction, 2: sensor On, 3: sensor off, 4: boundaries, 5 Text, 6 Background
technicolor = [(100, 100, 100), (255, 0, 0), (0, 255, 0), (0, 128, 0), (0, 0, 255), (0, 0, 0), (255, 255, 255)]

# Retro mode
retro = False
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
retrocolor = [GREEN, GREEN, GREEN, BLACK, GREEN, GREEN, BLACK]

colorpack = technicolor


# Tools for math
####################
# AUTHOR: Aleksander Michoński

# Function to get average of a list
def average(lst):
    return sum(lst) / len(lst)


# Find the perpendicular vector on a line, used to determine what directions the robot is allowed to move while colliding
def perpendicular(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


# Finds the direction of a line
def normalize(a):
    ar = np.array(a)
    return ar / np.linalg.norm(ar)


# Finds the slope of the line
def slope(p1, p2):
    return (p2[1] - p1[1]) * 1. / (p2[0] - p1[0] + eps)


# Y interceopt, where y = 0
def y_intercept(slop, p1):
    return p1[1] - 1. * slop * p1[0]


# Find the intersection between two lines
def intersect(line1, line2):
    # USED THE FOLLOWING WEBSITE FOR THIS CALCULATION
    # https://ericleong.me/research/circle-line/#line-line-intersection
    x1 = line1.p1[0]
    y1 = line1.p1[1]
    x2 = line1.p2[0]
    y2 = line1.p2[1]
    x3 = line2.p1[0]
    y3 = line2.p1[1]
    x4 = line2.p2[0]
    y4 = line2.p2[1]

    a1 = y2 - y1
    a2 = y4 - y3
    b1 = x1 - x2
    b2 = x3 - x4
    c1 = a1 * x1 + b1 * y1
    c2 = a2 * x3 + b2 * y3

    determinant = a1 * b2 - a2 * b1
    det_err = 0.1

    if -det_err <= determinant <= det_err:
        return None
    else:
        x = floor((c1 * b2 - b1 * c2) / determinant)
        y = floor((a1 * c2 - c1 * a2) / determinant)

    # X,Y is hypothetical point of intersection
    if min(x1, x2) <= x <= max(x1, x2):
        if min(x3, x4) <= x <= max(x3, x4):
            if min(y1, y2) <= y <= max(y1, y2):
                if min(y3, y4) <= y <= max(y3, y4):
                    return x, y
                else:
                    return None
            else:
                return None
        else:
            return None
    else:
        return None


# Finds intersection between a line and point
def line_point(x1, y1, x2, y2, px, py):
    d1 = sqrt(((px - x1) * (px - x1)) + ((py - y1) * (py - y1)))
    d2 = sqrt(((px - x2) * (px - x2)) + ((py - y2) * (py - y2)))
    line_len = sqrt(((x1 - x2) * (x1 - x2)) + ((y1 - y2) * (y1 - y2)))
    buffer = 0.2
    if line_len - buffer <= d1 + d2 <= line_len + buffer:
        return True
    return False


# Finds intersection between point and a circle
def point_circle(px, py, cx, cy, r):
    dist_x = px - cx
    dist_y = py - cy
    dist = sqrt((dist_x * dist_x) + (dist_y * dist_y))

    if dist <= r:
        return True
    else:
        return False


# FInd intersection between line and circle
def line_circle(rob, line_p1, line_p2):
    # http://www.jeffreythompson.org/collision-detection/line-circle.php
    buff = 5
    cx = rob.x
    cy = rob.y
    rad = rob.rad + buff
    x1 = line_p1[0]
    y1 = line_p1[1]
    x2 = line_p2[0]
    y2 = line_p2[1]

    inside1 = point_circle(x1, y1, cx, cy, rad)
    inside2 = point_circle(x2, y2, cx, cy, rad)
    if inside1 or inside2:
        return True

    distX = x1 - x2
    distY = y1 - y2
    leng = sqrt((distX * distX) + (distY * distY))

    dot = (((cx - x1) * (x2 - x1)) + ((cy - y1) * (y2 - y1))) / pow(leng, 2)

    closest_x = x1 + (dot * (x2 - x1))
    closest_y = y1 + (dot * (y2 - y1))

    on_segment = line_point(x1, y1, x2, y2, closest_x, closest_y)
    if not on_segment:
        return False

    dist_x = closest_x - cx
    dist_y = closest_y - cy
    dist_tot = sqrt((dist_x * dist_x) + (dist_y * dist_y))

    if dist_tot <= rad:
        return True

    return False


# Euclidean distance
def distance(p1, p2):
    return sqrt(((p2[0] - p1[0]) * (p2[0] - p1[0])) + ((p2[1] - p1[1]) * (p2[1] - p1[1])))


# Circle circle collision
def circle_circle(x1, y1, x2, y2, r1, r2):
    distSq = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
    radSumSq = (r1 + r2) * (r1 + r2)
    if (distSq == radSumSq):
        return True
    elif (distSq > radSumSq):
        return False
    else:
        return True


# END OF MATH

# A line made up of two points, the rest can be inferred with calculations
class Line:
    # AUTHOR: Aleksander Michoński & Lucas Vanparijs
    def __init__(self, p1, p2, ori):
        self.p1 = p1
        self.p2 = p2
        self.ori = ori

    def print(self):
        print("X: " + str(self.p1[0]) + ", Y: " + str(self.p1[1]) + "X: " + str(self.p2[0]) + ", Y: " + str(self.p2[1]))

    def draw(self, can, col):
        pygame.draw.line(can, col, self.p1, self.p2, 5)


# Rectangle class made up of 4 lines, and size measurements
class Rectangle:
    # AUTHOR: Aleksander Michoński & Lucas Vanparijs
    def __init__(self, pos, width, length):
        self.pos = pos  # top left corner
        self.width = width
        self.length = length
        buff = 0
        p1 = pos
        p2 = (pos[0] + width, pos[1])
        p3 = (pos[0], pos[1] + length)
        p4 = (pos[0] + width, pos[1] + length)
        self.lines = (
            Line((p1[0] - buff, p1[1]), (p2[0] + buff, p2[1]), -1),
            Line((p1[0], p1[1] - buff), (p3[0], p3[1] + buff), 1),
            Line((p2[0], p2[1] - buff), (p4[0], p4[1] + buff), -1),
            Line((p3[0] - buff, p3[1]), (p4[0] + buff, p4[1]), 1))
        # LINE 0
        # _______________
        # | L             |
        # | I             |   <- LINE 2
        # | N             |
        # | E             |
        # | 1             |
        # -----------------
        # LINE 3

    def draw(self, can, col):
        for l in self.lines:
            l.draw(can, col)


class Robot:
    # AUTHOR: Aleksander Michoński & Lucas Vanparijs
    def __init__(self, ann, dir, x, y, lwv, rwv, rad):
        # BASIC ROBOT VARIABLES
        self.dir = dir  # direction in radians
        self.x = x  # x coordinate
        self.y = y  # y coordinate
        self.last_x = x
        self.last_y = y
        self.trajectory = []
        self.lwv = lwv  # Left wheel velocity
        self.rwv = rwv  # Right wheel velocity
        self.rad = rad  # radius of the robot
        self.sensor_length = rad * 3  # Length that the distance sensors reaches
        self.num_sensors = 12
        self.sensors = [None] * self.num_sensors
        self.collided = False
        self.rotation_speed = 0
        self.vel = 3
        self.max_vel = 50
        self.color = (int(255 * random.random()), int(255 * random.random()), int(255 * random.random()))

        # Stats the robot keeps track of for fitness calculation
        self.avg_unsigned_rotation_speed = 0
        self.tot_unsigned_rotation_speed = 0
        self.n_unsigned_rotation_speed = 0
        self.speed_diff = 0
        self.hi_activation_val = 1.
        self.tot_fitness = 0

        # Generates dust
        self.num_dust = 500
        self.dust_rate = 1
        self.dust = [(int(windowSize * random.random()), int(windowSize * random.random())) for j in
                     range(self.num_dust)]

        # Initialises Neural Network
        self.ann = ann
        self.ann.set_robot(self)

        sens = list(range(0, self.num_sensors))
        for i in sens:
            self.sensors[i] = Sensor((i + 0.5) * pi / 6, self.sensor_length, self)

    def init_vars(self):
        self.avg_unsigned_rotation_speed = 0
        self.tot_unsigned_rotation_speed = 0
        self.n_unsigned_rotation_speed = 0
        self.speed_diff = 0
        self.hi_activation_val = 1.
        self.tot_fitness = 0
        self.num_dust = 500
        self.dust_rate = 1
        self.dust = [(int(windowSize * random.random()), int(windowSize * random.random())) for j in
                     range(self.num_dust)]

    # Updates everything in the robot
    def update(self):
        self.last_x = self.x
        self.last_y = self.y

        self.trajectory += [(self.x, self.y)]

        # Calculates directional components
        dirX = cos(self.dir)
        dirY = sin(self.dir)
        v = (self.lwv + self.rwv) / 2

        # Signed distance from ICC to the midpoint of the robot
        R = self.rad * ((self.lwv + self.rwv) / (self.rwv - self.lwv + eps))

        # Rate of rotation/angular velocity
        w = (self.rwv - self.lwv) / (2 * self.rad)

        # Instantaneous Center of curvature
        iccx = self.x - R * sin(self.dir)
        iccy = self.y + R * cos(self.dir)

        rotation = np.array(
            [[cos(w * timestep_in_sec), -sin(w * timestep_in_sec), 0],
             [sin(w * timestep_in_sec), cos(w * timestep_in_sec), 0], [0, 0, 1]])
        toOrigin = np.transpose(np.array([self.x - iccx, self.y - iccy, self.dir]))
        toICC = np.transpose(np.array([iccx, iccy, w * timestep_in_sec]))

        # @ does matrix multiplication
        newPos = (rotation @ toOrigin) + toICC

        # defaults at not collided
        self.collided = False

        # Checks for collisions
        collision_lines = [None] * 4
        for b in boundaries:
            for i in [0, 1, 2, 3]:
                l = b.lines[i]
                if line_circle(self, l.p1, l.p2):
                    self.collided = True
                    collision_lines[i] = l

        # Calculates the next step of the robot
        move_x = floor(self.x + dirX * v * timestep_in_sec) - self.x
        move_y = floor(self.y + dirY * v * timestep_in_sec) - self.y
        allowed_movement = np.array([0, 0])
        if self.collided:
            for l in collision_lines:
                if l is not None:
                    vec = [(l.p2[0] - l.p1[0]), (l.p2[1] - l.p1[1])]
                    norm = normalize(vec)
                    perp = np.array(perpendicular(norm))
                    perp = l.ori * perp
                    allowed_movement = np.add(allowed_movement, perp)

        # Based on the perpendicular vector and the direction of the line, the robot is allowed to move in each dimension when different condition have been met
        if allowed_movement[0] == 0:
            self.x = floor(self.x + dirX * v * timestep_in_sec)
        elif np.sign(move_x) == np.sign(allowed_movement[0]):
            self.x = floor(self.x + dirX * v * timestep_in_sec)
        else:
            self.x = self.x

        if allowed_movement[1] == 0:
            self.y = floor(self.y + dirY * v * timestep_in_sec)
        elif np.sign(move_y) == np.sign(allowed_movement[1]):
            self.y = floor(self.y + dirY * v * timestep_in_sec)
        else:
            self.y = self.y

        # Updates all the sensors
        for t in self.sensors:
            t.update()

        # Keeps the directions within 0 - 2*PI
        self.dir = newPos[2]
        if self.dir > 2 * pi:
            self.dir = 0
        if self.dir < 0:
            self.dir = 2 * pi

        self.n_unsigned_rotation_speed += 2
        self.tot_unsigned_rotation_speed += abs(self.rwv) + abs(self.lwv)
        self.avg_unsigned_rotation_speed = self.tot_unsigned_rotation_speed / self.n_unsigned_rotation_speed

        self.speed_diff = abs(self.lwv - self.rwv) / (2 * self.max_vel)

        # updates the dust
        self.dust = self.suck(self.dust)
        self.dust_rate = len(self.dust) / self.num_dust

        # Gets the value if the sensor that is most activated, (lowest value/distance)
        tmp_act = []  # np.zeros((1,input_layer_size))#np.empty((1,input_layer_size))
        for t in self.sensors:
            tmp_act += [t.value]
        tmp_act = np.array([tmp_act])
        self.hi_activation_val = (np.min(tmp_act))

        self.tot_fitness += self.ann.fitness()

    # add speed to left wheel
    def left_wheel(self):
        nxt_vel = self.lwv + self.vel
        if self.max_vel > nxt_vel > 0:
            self.lwv = nxt_vel
        else:
            self.lwv = self.max_vel

    # add speed to right wheel
    def right_wheel(self):
        nxt_vel = self.rwv + self.vel
        if self.max_vel > nxt_vel > 0:
            self.rwv = nxt_vel
        else:
            self.rwv = self.max_vel

    # subtract speed from left wheel
    def left_wheel_stop(self):
        nxt_vel = self.lwv - self.vel
        if self.max_vel > nxt_vel > 0:
            self.lwv = nxt_vel
        else:
            self.lwv = 0

    # subtract speed from right wheel
    def right_wheel_stop(self):
        nxt_vel = self.rwv - self.vel
        if self.max_vel > nxt_vel > 0:
            self.rwv = nxt_vel
        else:
            self.rwv = 0

    def stop(self):
        self.rwv = 0
        self.lwv = 0

    # Suck up dust that is under the robot, update dust list
    def suck(self, dust):
        for d in dust:
            if point_circle(d[0], d[1], self.x, self.y, self.rad):
                dust.remove(d)
        return dust

    # Returns sensor values
    def input_values(self):
        input_layer_size = self.num_sensors
        sensors_value = []
        for i in range(input_layer_size):
            sensors_value += [self.sensors[i].value / self.sensors[i].max_value]
        sensors_value = np.array([sensors_value])
        return sensors_value

    #Euclidean distance since last iteration
    def distance_since_last(self):
        return distance((self.last_x, self.last_y), (self.x, self.y))

    def draw(self, can):
        pygame.draw.circle(can, colorpack[0], (self.x, self.y), self.rad, 3)
        pygame.draw.line(can, colorpack[1], (self.x, self.y),
                         (self.x + self.rad * cos(self.dir), self.y + self.rad * sin(self.dir)), 3)


class Sensor:
    # AUTHOR: Aleksander Michoński & Lucas Vanparijs
    def __init__(self, rel_dir, length, rob):
        self.rel_dir = rel_dir
        self.length = length
        self.buffer = 0
        self.rob = rob
        self.start = (
            rob.x + (rob.rad - self.buffer) * cos(rob.dir + self.rel_dir),
            rob.y + (rob.rad - self.buffer) * sin(rob.dir + self.rel_dir))
        self.end = (
            rob.x + (rob.rad + self.length) * cos(rob.dir + self.rel_dir),
            rob.y + (rob.rad + self.length) * sin(rob.dir + self.rel_dir))
        self.line = Line(self.start, self.end, 0)
        self.triggered = False
        self.intersection_point = (None, None)
        self.max_value = 1.
        self.value = 1.

    def update(self):
        self.start = (
            floor(self.rob.x + (self.rob.rad - self.buffer) * cos(self.rob.dir + self.rel_dir)),
            floor(self.rob.y + (self.rob.rad - self.buffer) * sin(self.rob.dir + self.rel_dir)))
        self.end = (
            floor(self.rob.x + (self.rob.rad + self.length) * cos(self.rob.dir + self.rel_dir)),
            floor(self.rob.y + (self.rob.rad + self.length) * sin(self.rob.dir + self.rel_dir)))
        self.line = Line(self.start, self.end, 0)
        self.triggered = self.collision()
        if self.triggered:
            self.value = distance(self.start, self.intersection_point) / distance(self.start, self.end)
        else:
            self.value = self.max_value
            self.intersection_point = (None, None)

    def collision(self):
        for o in boundaries:
            for l in o.lines:
                ip = intersect(l, self.line)
                if ip is not None:
                    self.intersection_point = ip
                    return True
        return False

    def draw(self, can):

        if self.triggered:
            pygame.draw.line(can, colorpack[2], self.start, self.end, 3)
        else:
            pygame.draw.line(can, colorpack[3], self.start, self.end, 3)

        # label = myfont.render(str(self.value), 2, colorpack[5])
        # can.blit(label, (self.end[0], self.end[1]))


def random_weights(height, width):
    return np.random.uniform(-1, 1, (height, width))


def binarise(x):
    return np.where(x >= 0.5, 1, 0)


# https://dev.to/shamdasani/build-a-flexible-neural-network-with-backpropagation-in-python
class ANN(object):
    # AUTHOR: Lucas Vanparijs
    def __init__(self, w1, w2):
        input_layer_size = 12
        hidden_layer_size = 4
        output_layer_size = 2

        self.hidden_layer = None
        self.output_layer = None

        if w1 is not None:
            self.W1 = w1
        else:
            self.W1 = random_weights(input_layer_size, hidden_layer_size)

        if w2 is not None:
            self.W2 = w2
        else:
            self.W2 = random_weights(hidden_layer_size, output_layer_size)

        self.robot = None

    def set_robot(self, r):
        self.robot = r

    def learn(self, X):
        o = self.forward(X)
        # self.backward(X, o)# use this for back propagation
        return o

    #forward propagation
    def forward(self, X):
        bias = random.uniform(-1, 1) * 0.2
        self.hidden_layer = np.dot(X, self.W1) + bias
        self.hidden_layer = self.sigmoid(self.hidden_layer)
        self.output_layer = np.dot(self.hidden_layer, self.W2)
        o = self.sigmoid(self.output_layer)
        return o

    #backwards propagation, NOT USED
    def backward(self, X, o):
        fit = self.fitness()
        error = 0. - fit
        self.robot.tot_fitness += fit

        output_delta = error * self.sigmoid_(o)
        w2_err = output_delta.dot(self.W2.T)

        d_w2 = w2_err * self.sigmoid_(self.hidden_layer)

        self.W1 += X.T.dot(d_w2)
        self.W2 += self.hidden_layer.T.dot(output_delta)

    # FItness function
    def fitness(self):
        movement = 1 / (1 + exp(-self.robot.distance_since_last() + 4))
        fit = movement * (sqrt(self.robot.hi_activation_val)) * (1 - pow(self.robot.dust_rate, 4))
        return fit

    def sigmoid(self, x):
        return sp.expit(x)

    def sigmoid_(self, x):
        return x * (1 - x)


# Sorts population in descending order
def sort_population(pop):
    return sorted(pop, key=lambda pop: pop.tot_fitness, reverse=True)

# AUTHOR: Lucas Vanparijs
def mate(m, f):
    # different crossover method
    a = random.uniform(0, 1)
    w1 = m.ann.W1 * a + f.ann.W1 * (1. - a)
    a = random.uniform(0, 1)
    w2 = m.ann.W2 * a + f.ann.W2 * (1. - a)
    return mutate(Robot(ANN(w1, w2), random.uniform(0, 1) * pi * 2, spawn_x, spawn_y, 0, 0, 15))

# AUTHOR: Lucas Vanparijs
def mutate(indiv):
    mutation_rate = 0.05
    for ix in range(indiv.ann.W1.shape[0]):
        for iy in range(indiv.ann.W1.shape[1]):
            if random.uniform(0, 1) < mutation_rate:
                indiv.ann.W1[ix][iy] = random.uniform(-1, 1)

    for ix in range(indiv.ann.W2.shape[0]):
        for iy in range(indiv.ann.W2.shape[1]):
            if random.uniform(0, 1) < mutation_rate:
                indiv.ann.W2[ix][iy] = random.uniform(-1, 1)
    return indiv

# AUTHOR: Lucas Vanparijs
def evolve(individuals):
    tournament_size = floor(population_size / 10)
    sorted_pop = sort_population(individuals)
    selected = []

    for j in range(population_size):
        if j == 0:
            selected += [sorted_pop[0]]  # Elitism, best organism never gets worse per generation
        else:
            tournament = []
            for m in range(tournament_size):
                tournament += [individuals[random.randint(0, len(individuals) - 1)]]  # randomly select tournament
            tournament = sort_population(tournament)
            selected += [tournament[0]]
    childs = [sorted_pop[0]]
    for s in range(len(selected) - 1):
        childs += [mate(selected[random.randint(0, len(selected) - 1)], selected[random.randint(0, len(selected) - 1)])]
    return childs



#Start of Simulation Section
# AUTHOR: Aleksander Michoński & Lucas Vanparijs
generations = 50
population_size = 80

level1 = (
    Rectangle((-50, 0), 50, windowSize), Rectangle((0, -50), windowSize, 50),
    Rectangle((windowSize, 0), 50, windowSize),
    Rectangle((0, windowSize), windowSize, 50))

level2 = (
    Rectangle((-50, 0), 50, windowSize), Rectangle((0, -50), windowSize, 50),
    Rectangle((windowSize, 0), 50, windowSize),
    Rectangle((0, windowSize), windowSize, 50), Rectangle((75, 75), 150, 150))

level3 = (
    Rectangle((-50, 0), 50, windowSize), Rectangle((0, -50), windowSize, 50),
    Rectangle((windowSize, 0), 50, windowSize),
    Rectangle((0, windowSize), windowSize, 50), Rectangle((60, 0), 30, 230), Rectangle((165, 70), 30, 230))

level4 = (
    Rectangle((-50, 0), 50, windowSize), Rectangle((0, -50), windowSize, 50),
    Rectangle((windowSize, 0), 50, windowSize),
    Rectangle((0, windowSize), windowSize, 50), Rectangle((60, 60), 50, 50), Rectangle((155, 80), 30, 60), Rectangle((50, 200), 140, 40), Rectangle((225, 40), 25, 80))

level5 = (
    Rectangle((-50, 0), 50, windowSize), Rectangle((0, -50), windowSize, 50),
    Rectangle((windowSize, 0), 50, windowSize),
    Rectangle((0, windowSize), windowSize, 50), Rectangle((0, 70), 100, 100), Rectangle((165, 0), 30, 230))


pygame.init()
win = pygame.display.set_mode((windowSize, windowSize), 0, 32)
myfont = pygame.font.SysFont("monospace", 10)
win.fill(colorpack[6])
boundaries = level1

def switch(x):
    return {
        0: level4,
        1: level5,
        2: level3,
        3: level4,
        4: level5,
    }[x]

#Simulate and learn for each level
for lvl in range(2):
    population = [None] * population_size
    for p in range(population_size):
        population[p] = (Robot(ANN(None, None), random.uniform(0, 1) * pi * 2, spawn_x, spawn_y, 0, 0, 15))
    lst = []
    avg = []
    div = []
    mx = []
    for g in range(generations):
        level = switch(lvl)
        boundaries = level
        for i in range(iterations):
            pygame.event.pump()
            win.fill(colorpack[6])
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()  # sys.exit() if sys is imported
            draw_now = True
            lst = []
            for p in population:
                output = p.ann.learn(p.input_values())
                output_bin = binarise(output)
                if output_bin[0][0] == 1:
                    p.right_wheel()
                else:
                    p.right_wheel_stop()

                if output_bin[0][1] == 1:
                    p.left_wheel()
                else:
                    p.left_wheel_stop()

                p.update()
                if i == iterations - 1:
                    lst += [p.tot_fitness]

                if draw_now:
                    draw_now = True
                    p.draw(win)

            srt_pop = sort_population(population)
            for t in srt_pop[0].trajectory:
                pygame.draw.circle(win, srt_pop[0].color, t, srt_pop[0].rad, 0)
            for b in boundaries:
                b.draw(win, colorpack[4])
            pygame.display.update()
            pygame.time.wait(timestep)

        avg += [average(lst)]

        population = evolve(population)

        mx += [population[0].tot_fitness]

        traj = population[0].trajectory
        win.fill(colorpack[6])
        pygame.time.wait(5000)
        win.fill(colorpack[6])


        diversity = 0
        for p1 in population:
            for p2 in population:
                if not (p1 == p2):
                    dist_W1 = (p1.ann.W1 - p2.ann.W1) ** 2
                    dist_W2 = (p1.ann.W2 - p2.ann.W2) ** 2
                    dist_W1 = np.sum(dist_W1, axis=None)
                    dist_W2 = np.sum(dist_W2, axis=None)
                    diversity += np.sqrt(dist_W1) + np.sqrt(dist_W2)
        div += [diversity / 100]


        #Save data to file
        filename = 'LVL' + str(lvl+2) + '.txt'
        with open(filename, 'a') as outfile:
            print(
                "Best of Gen: " + str(g) + ", Ind: " + str(population[0]) + ", Fit: " + str(population[0].tot_fitness),
                file=outfile)
            print("W1", file=outfile)
            print(population[0].ann.W1, file=outfile)
            print("W2", file=outfile)
            print(population[0].ann.W2, file=outfile)
            print("AVG", file=outfile)
            print(avg, file=outfile)
            print("MX", file=outfile)
            print(mx, file=outfile)
            print("DIV", file=outfile)
            print(div, file=outfile)
            print("Trajectory", file=outfile)
            print(population[0].trajectory, file=outfile)


        fig, ax1 = plt.subplots()
        # plot of average and diversity
        ax2 = ax1.twinx()
        ax1.plot(avg, 'g-')
        ax2.plot(div, 'b-')
        ax1.plot(mx, 'y-')

        ax1.set_xlabel('generations')
        ax1.set_ylabel('fitness', color='g')
        ax2.set_ylabel('diversity', color='b')
        plt.grid(True)
        plt.show(block=False)
        plt.pause(5)
        plt.close()

        for s in population:
            s.tot_fitness = 0
            s.x = spawn_x
            s.y = spawn_y
            s.lwv = 0
            s.rwv = 0
            s.trajectory = []
            s.init_vars()

