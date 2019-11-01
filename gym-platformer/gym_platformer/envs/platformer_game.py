import pygame
import random
import numpy as np
import os

class Platformer2D:
	def __init__(self, file_path, size=(5,5)):

		pygame.init()
		pygame.display.set_caption("Platformer")
		self.clock = pygame.time.Clock()
		self.game_over = False
		
		#Generate a new map
		if file_path is None:
			return
		else:
			file = np.load(file_path)
			


class Map:

	def __init__(self, map_cells=None, map_size=(5,5)):
		self.map_cells = map_cells
		self.map_size = map_size