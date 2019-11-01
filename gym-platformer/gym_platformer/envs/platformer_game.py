import pygame
import random
import numpy as np
import os


GAME_SPEED = 0.01
class Platformer2D:
	def __init__(self, file_path="test_map.npy", size=(5,5), render=True):

		pygame.init()
		pygame.display.set_caption("Platformer")
		self.clock = pygame.time.Clock()
		self.game_over = False
		self.render = render
		
		#Generate a new map
		if file_path is None:
			return
		else:
			if not os.path.exists(file_path):
				dir_path = os.path.dirname(os.path.abspath(__file__))
				rel_path = os.path.join(dir_path, "platformer_maps", file_path)
				if os.path.exists(rel_path):
					file_path = rel_path
				else:
					raise FileExistsError("Cannot find %s." % file_path)
			map_cells = Map.load_map(file_path)
			self.map = Map(map_cells=map_cells)
			file = np.load(file_path)

		


		self.start_point = self.map.get_locations_of_symbol(1)[0]
		self.goal = self.map.get_locations_of_symbol(2)[0]

		start_location = self.map.get_location_from_index(self.start_point)

		self.player = Player(start_location=start_location, game_map=self.map)

		if self.render is True:
			
			BLACK=(0,0,0,0)
			self.window_size = self.map.get_window_size()

			# to show the right and bottom border
			self.screen = pygame.display.set_mode(self.window_size)
			self.window_size = tuple(map(sum, zip(self.window_size, (-1, -1))))
			self.map_layer = pygame.Surface(self.screen.get_size()).convert_alpha()
		self.game_loop()


	def draw_obstacles(self):
		BLUE=(40,40,120)
		obstacle_locations = self.map.get_locations_of_symbol(3)

		cell_width = self.map.get_cell_size()
		for loc in obstacle_locations:
			loc = np.flip(loc)
			draw_location = loc * self.map.get_cell_size()
			rect = pygame.Rect(draw_location, self.map.get_cell_size())
			draw_rect = pygame.draw.rect(self.map_layer,BLUE, rect)

	def game_loop(self):
		BLACK=(0,0,0,0)
		dt = 0
		while True:
			ev = pygame.event.poll()    
			if ev.type == pygame.QUIT:  
			    break                  

			
			self.map_layer.fill(BLACK)
			self.draw_obstacles()
			self.player.perform_action("Right")
			self.player.update_position(dt)
			self.player.draw(self.map_layer)
			self.screen.blit(self.map_layer, (0,0))

			pygame.display.flip()
			pygame.display.update()
			dt = self.clock.tick(60)
		pygame.quit() 

def create_test_map():
	map_vals = np.zeros((10, 15))
	map_vals[:, 0] = 3
	map_vals[0,:] = 3
	map_vals[:,-1] = 3
	map_vals[-1, :] = 3
	map_vals[-2,1] = 1
	map_vals[-2,-2] = 2
	np.save("platformer_maps/test_map", map_vals)



class Player:

	ACTIONS = {
        "Left",
        "Right",
        "Jump"
    }
	def __init__(self, start_location, game_map, size=[32,64], color=(160,40,40), ):
		self.location = np.array(start_location)
		self.game_map = game_map
		self.location[0] += size[0] // 2
		self.size = size
		self.color = color
		self.velocity = np.array([0.0, 0.0])

	def get_location(self):
		return self.location

	def perform_action(self, action):
		if action is not None and action not in self.ACTIONS:
			raise ValueError("Action cannot be %s. Please choose one of the following actions: %s."
                             % (str(action), str(self.ACTIONS)))
		
		if action == "Left":
			self.velocity[0] = -1.0
		elif action == "Right":
			self.velocity[0] = 1.0


	def update_position(self, dt):
		new_location = self.location + self.velocity * (GAME_SPEED * dt)
		if self.check_bounds(new_location):
			self.location = new_location

	def check_bounds(self, location):
		if location[0] < 0 or location[0] > self.game_map.MAP_W or location[1] < 0 or location[1] > self.game_map.MAP_H:
			return False
		return True

	def draw(self, layer):
		pixel_loc = self.location.astype(int)
		rect = pygame.Rect(self.location, self.size)
		draw_rect = pygame.draw.rect(layer,self.color, rect)

class Map:

	def __init__(self, map_cells=None, map_size=(5,5), cell_size=[64, 64]):
		self.map_cells = map_cells
		self.cell_size = cell_size
		if self.map_cells is None:
			self.map_size = map_size
		else:
			self.map_size = map_cells.shape

	def load_map(file_path):
		return np.load(file_path)

	def get_locations_of_symbol(self, symbol):
		indices = np.argwhere(self.map_cells == symbol)
		return indices

	def get_location_from_index(self, index):
		x = int( index[1] * self.cell_size[1])
		y = int( index[0] * self.cell_size[0])
		return [x, y]

	def get_cell_size(self):
		return self.cell_size

	def get_window_size(self):
		width = int( self.map_size[1] * self.cell_size[1])
		height = int( self.map_size[0] * self.cell_size[0])
		return [width, height]

	@property
	def MAP_W(self):
		return self.map_size[1] * self.cell_size[1]

	@property
	def MAP_H(self):
		return self.map_size[0] * self.cell_size[0]

def main():
	create_test_map()
	a = Platformer2D()

if __name__ == "__main__":
	main()