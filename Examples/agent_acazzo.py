#!/usr/bin/env python
import retro
import numpy as np
import cv2
import neat
#import pickler


# dimensioni schermo: 200 256 3
imgarray = []
current_max_fitness = 0

env = retro.make('StreetFighterIISpecialChampionEdition-Genesis', 'Champion.Level1.RyuVsGuile')



def eval_genomes(genomes, config):

	for genome_id, genome in genomes:

		ob = env.reset()
		ac = env.action_space.sample()
		inx, iny, inc = env.observation_space.shape

		#print(inx, iny, inc)

		inx = int(inx/6)
		iny = int(iny/8)



		net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

		current_max_fitness = 0
		fitness_current = 0
		frame = 0
		counter = 0
		xpos = 0
		xpos_max = 0

		done = False

		#cv2.namedWindow("main", cv2.WINDOW_NORMAL)

		while not done:

			env.render()
			frame += 1

			#scaledimg = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
			#scaledimg = cv2.resize(scaledimg, (inx, iny))			
				
			ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
			ob = cv2.resize(ob, (inx, iny))
			ob = np.reshape(ob, (inx,iny))

			#cv2.imshow('main', scaledimg)
			#cv2.waitKey(1)

			for x in ob:
				for y in x:
					imgarray.append(y)


			nnOutput = net.activate(imgarray)

			ob, rew, done, info = env.step(nnOutput)
			imgarray.clear()


			health = info['health']
			enemy_health = info['enemy_health']
			matches_won = info['matches_won']
					    
			fitness_current = (health - enemy_health) / 176


			if fitness_current > current_max_fitness:
				current_max_fitness = fitness_current
				counter = 0
				print(genome_id, fitness_current)
			else:
				counter += 1

			if matches_won == 1:
				fitness_current += 10

			if done or counter == 700:
				if fitness_current < 0:
					genome.fitness = 0
				done = True
				print(genome_id, fitness_current)

			if fitness_current < 0:
				genome.fitness = 0


# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward')

p = neat.Population(config)
winner = p.run(eval_genomes)



