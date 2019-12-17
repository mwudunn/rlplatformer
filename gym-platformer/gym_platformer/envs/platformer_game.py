import pygame
import random
import numpy as np
import os
from enum import Enum

GAME_SPEED = 0.1
GRAVITY_MODIFIER = 0.07
JUMP_SPEED = 5.0
MOVE_SPEED = 2.0
BLACK=(0,0,0,0)
#Game Parameters
params = {
	"PLAYER": 1,
	"GOAL": 2,
	"OBSTACLE": 3,
	"COIN": 4,
}

class Action(Enum):
	NONE = 0
	LEFT = 1
	RIGHT = 2
	JUMP = 3

class Platformer2D:
	def __init__(self, file_path="test_map.npy", enable_render=False, play_game=False, alloted_time=10000):

		pygame.init()
		pygame.display.set_caption("Platformer")
		self.clock = pygame.time.Clock()
		self.game_over = False
		self.enable_render = enable_render
		self.dt = 1
		self.timesteps = 0
		self.alloted_time = alloted_time
		
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
			self.game_map = Map(map_cells=map_cells)
			file = np.load(file_path)

		self.start_point = self.game_map.get_locations_of_symbol(1)[0]
		self.goal = self.game_map.get_locations_of_symbol(2)[0]
		start_location = self.game_map.get_location_from_index(self.start_point)
		self.player = Player(start_location=start_location, game_map=self.game_map)
		self.window_size = self.game_map.get_window_size()

		if self.enable_render:			
			# to show the right and bottom border

			self.screen = pygame.display.set_mode(self.window_size)
			self.window_size = tuple(map(sum, zip(self.window_size, (-1, -1))))
			self.map_layer = pygame.Surface(self.screen.get_size()).convert_alpha()

		if play_game:
			self.game_loop()

	def get_goal_location(self):
		return np.array(self.game_map.get_location_from_index(self.goal))

	def get_remaining_time(self):
		return self.alloted_time - self.timesteps

	def render(self, mode='human', close=False):
		if mode == "human":
			self.screen.fill(BLACK)
			self.map_layer.fill(BLACK)
			self.game_map.draw_objects(self.map_layer)
			
			# self.player.update_position(dt)
			self.player.draw(self.map_layer)

			self.screen.blit(self.map_layer, (0,0))

			pygame.display.flip()

	def get_player(self):
		return self.player

	def perform_action(self, ac, dt=20):
		self.timesteps += dt
		self.player.perform_action(ac, dt)

	def game_loop(self):
		
		dt = 0
		self.reset()
		while not self.game_over:
			ac = Action.NONE
			ev = pygame.event.poll()    
			if ev.type == pygame.QUIT:  
				break    

			keys = pygame.key.get_pressed()
			if keys[pygame.K_LEFT]:
				ac = Action.LEFT 
				
			if keys[pygame.K_RIGHT]:
				ac = Action.RIGHT   

			if keys[pygame.K_SPACE]:
				ac = Action.JUMP

			dt = self.clock.tick(60)
			self.perform_action(ac, dt)
			self.render()
			

			self.update_environment()

		pygame.quit() 

	def update_environment(self):
		if self.game_map.check_goal_collision(self.player):
			self.game_over = True
		elif self.timesteps > self.alloted_time:
			self.game_over = True
		return self.game_over

	def reset(self):
		start_location = self.game_map.get_location_from_index(self.start_point)
		self.game_map.coins_flag = [1 for i in self.game_map.coins]
		self.game_over = False
		self.timesteps = 0
		self.clock = pygame.time.Clock()
		self.start_time = pygame.time.get_ticks()
		self.player = Player(start_location=start_location, game_map=self.game_map)

	def get_state(self):
		state = self.player.get_state()
		#state.append(self.get_remaining_time())
		return np.array(state)

def create_test_map():
	"""
	map_vals = np.zeros((20, 30))
	map_vals[:, 0] = 3
	map_vals[0,:] = 3
	map_vals[:,-1] = 3
	map_vals[-1, :] = 3
	map_vals[-2,1] = 1
	map_vals[-2,-2] = 2
	map_vals[-5, 4:15] = 3
	map_vals[-8, 17:25] = 3
	map_vals[-12, 11:13] = 3
	map_vals[-16, 15] = 3
	for i in range(8):
		map_vals[-2, (i + 1)*3] = 4
	"""
	map_vals = np.zeros((20, 30))
	map_vals[:, 0] = 3
	map_vals[0,:] = 3
	map_vals[:,-1] = 3
	map_vals[-1, :] = 3
	map_vals[-2,1] = 1
	map_vals[-17, 15] = 2
	map_vals[-5, 4:15] = 3
	map_vals[-8, 17:25] = 3
	map_vals[-12, 11:13] = 3
	map_vals[-16, 15] = 3
	map_vals[-4, 2] = 4
	map_vals[-6, 8] = 4
	map_vals[-6, 5] = 4
	map_vals[-6, 11] = 4
	map_vals[-6, 14] = 4
	map_vals[-9, 18] = 4
	map_vals[-15, 15] = 4
	map_vals[-13, 12] = 4

	np.save("platformer_maps/test_map", map_vals)



class Player:

	def __init__(self, start_location, game_map, size=[32,32], color=(160,40,40)):
		self.location = np.array(start_location)
		self.game_map = game_map
		self.location[0] += size[0] // 2
		self.size = size
		self.color = color
		self.grounded = False
		self.velocity = np.array([0.0, 0.0])
		self.rect = pygame.Rect(self.location, self.size)

	def get_location(self):
		return self.location

	def get_coins(self):
		return self.game_map.coins_flag

	def get_rect(self):
		return self.rect

	def get_state(self):
		#return [self.get_location()[0], self.get_location()[1], self.velocity[0], self.velocity[1]]
		state = [self.get_location()[0], self.get_location()[1]]
		state.extend(self.get_coins())
		return state

	def perform_action(self, ac, dt):
		if ac is not None and ac not in Action:
			raise ValueError("Action cannot be %s. Please choose one of the following actions: %s."
							 % (str(ac), str(Action)))
		
		if ac == Action.LEFT:
			self.velocity[0] = -1.0 * MOVE_SPEED
		elif ac == Action.RIGHT:
			self.velocity[0] = 1.0 * MOVE_SPEED
		elif ac == Action.NONE:
			self.velocity[0] = 0.0

		if ac == Action.JUMP and self.grounded:
			self.velocity[1] = -JUMP_SPEED

		return self.update_state(dt)

	def update_state(self, dt):
		next_location = self.location + self.velocity * (GAME_SPEED * dt)
		next_location_x = np.array([self.location[0] + self.velocity[0] * (GAME_SPEED * dt), self.location[1]])
		next_location_y = np.array([self.location[0], self.location[1]  + self.velocity[1] * (GAME_SPEED * dt)])
		next_rect_x = pygame.Rect(next_location_x, self.size)
		next_rect_y = pygame.Rect(next_location_y, self.size)

		if self.game_map.check_obstacle_collision(next_rect_y):
			if self.velocity[1] > 0.0:
				self.velocity[1] = 0.0
				self.grounded = True
			next_location = np.array([next_location[0], self.location[1]])
		else:
			self.grounded = False

		if self.game_map.check_obstacle_collision(next_rect_x):
			next_location = np.array([self.location[0], next_location[1]])

		if not self.grounded:
			self.velocity[1] += GRAVITY_MODIFIER * dt * GAME_SPEED
			if self.velocity[1] > 5.0:
				self.velocity[1] = 5.0
			if self.velocity[1] < -5.0:
				self.velocity[1] = -5.0
		
		coin, index = self.game_map.check_coin_collision(next_rect_x)
		if index != -1:
			self.game_map.coins_flag[index] = 0

		next_rect = pygame.Rect(next_location, self.size)

		self.location = next_location
		self.rect = next_rect

	def collide(self, rect):
		return self.get_rect().colliderect(rect)

	def draw(self, layer):
		pixel_loc = self.location.astype(int)
		draw_rect = pygame.draw.rect(layer,self.color, self.rect)		


class Map:
	def __init__(self, map_cells=None, map_size=(5,5), cell_size=[32, 32]):
		self.map_cells = map_cells
		self.cell_size = cell_size
		if self.map_cells is None:
			self.map_size = map_size
		else:
			self.map_size = map_cells.shape
			obstacle_locations = self.get_locations_of_symbol(3)
			goal_locations = self.get_locations_of_symbol(2)
			coin_locations = self.get_locations_of_symbol(4)

			self.obstacles = []
			for obstacle_index_loc in obstacle_locations:
				obstacle_loc = self.get_location_from_index(obstacle_index_loc)
				obstacle = pygame.Rect(obstacle_loc, cell_size)
				self.obstacles.append(obstacle)

			self.coins = []
			self.coins_flag = []
			for coin_index_loc in coin_locations:
				coin_loc = self.get_location_from_index(coin_index_loc)
				coin = pygame.Rect(coin_loc, cell_size)
				self.coins.append(coin)
				self.coins_flag.append(1)

			self.goals = []
			for goal_index_loc in goal_locations:
				goal_loc = self.get_location_from_index(goal_index_loc)
				goal = pygame.Rect(goal_loc, cell_size)
				self.goals.append(goal)



	def load_map(file_path):
		return np.load(file_path)

	def get_locations_of_symbol(self, symbol):
		indices = np.argwhere(self.map_cells == symbol)
		return indices

	def get_location_from_index(self, index):
		x = int( index[1] * self.cell_size[1])
		y = int( index[0] * self.cell_size[0])
		return [x, y]

	def convert_position_to_tile(self, location):
		x_index = math.floor(location[0] / self.cell_size[0])
		y_index = math.floor(location[1] / self.cell_size[1])
		return [x_index, y_index]

	def get_cell_size(self):
		return self.cell_size

	def get_window_size(self):
		width = int( self.map_size[1] * self.cell_size[1])
		height = int( self.map_size[0] * self.cell_size[0])
		return [width, height]

	def check_goal_collision(self, player):
		for goal in self.goals:
			if player.collide(goal):
				return True
		return False
		
	def check_coin_collision(self, player_rect):
		for i in range(len(self.coins)):
			coin = self.coins[i]
			if self.coins_flag[i] and player_rect.colliderect(coin):
			   return (coin, i)
		return -1, -1

	def check_obstacle_collision(self, player_rect):
		for obstacle in self.obstacles:
			if player_rect.colliderect(obstacle):
				return True
		return False

	def draw_objects(self, layer):
		self.draw_obstacles(layer)
		self.draw_goal(layer)
		self.draw_coins(layer)

	def draw_goal(self, layer):
		COLOR=(200,80,80)
		for goal in self.goals:
			draw_rect = pygame.draw.rect(layer, COLOR, goal)

	def draw_obstacles(self, layer):
		BLUE=(40,40,120)
		for obstacle in self.obstacles:
			draw_rect = pygame.draw.rect(layer, BLUE, obstacle)

	def draw_coins(self, layer):
		YELLOW=(155, 155, 40)
		for i in range(len(self.coins)):
			coin = self.coins[i]
			if self.coins_flag[i]:
				draw_rect = pygame.draw.rect(layer, YELLOW, coin)


	@property
	def MAP_W(self):
		return self.map_size[1] * self.cell_size[1]

	@property
	def MAP_H(self):
		return self.map_size[0] * self.cell_size[0]

def main():
	create_test_map()
	a = Platformer2D(enable_render=True,play_game=True)

if __name__ == "__main__":
	main()
