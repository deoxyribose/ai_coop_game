from infer_goal import *
import utils
import numpy as np

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()
        self.obstacles = [
            Obstacle(300, 300, 100, 50),
            Obstacle(500, 150, 150, 60),
        ]
        goal_1 = Goal(700, 500, self.screen)
        goal_2 = Goal(100, 500, self.screen)
        self.goals = [goal_1, goal_2]

        self.player = Player(100, 100)   # Initialize player
        self.npc = NPC(200, 400)
        self.npc.perceive_obstacles(self.obstacles, self.screen)
        self.npc.perceive_player(self.player, self.screen)
        self.npc.perceive_goals(self.goals, self.screen)
        self.npc.make_path_planner(self.player)

        goal_1_reached = goal_1.check_reached(self.player) or goal_1.check_reached(self.npc)
        goal_2_reached = goal_2.check_reached(self.player) or goal_2.check_reached(self.npc)
        self.victory_condition = goal_1_reached and goal_2_reached

    def run(self):
        # Main game loop
        running = True
        while running:
            self.screen.fill("white")
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Update player position based on input
            keys = pygame.key.get_pressed()
            self.player.move(keys)

            # Check for collisions
            if self.player.check_collision(self.obstacles):
                # If collision, repel self.player position
                self.player.pos = self.player.prev_pos

            # Draw all elements
            for goal in self.goals:
                goal.draw(self.screen)
            for obstacle in self.obstacles:
                obstacle.draw(self.screen)
            self.player.draw(self.screen)
            self.npc.draw(self.screen)

            # Check if self.player reached the goal
            if self.victory_condition:
                font = pygame.font.Font(None, 74)
                text = font.render("Goal!", True, "purple")
                text_rect = text.get_rect(center=(self.screen.get_width() / 2, self.screen.get_height() * 0.2))
                self.screen.blit(text, text_rect)

            self.npc.replan(self.player, self.screen)
            
            # Update display and tick
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


class Player:
    def __init__(self, x, y, speed=5, size=20):
        self.pos = pygame.Vector2(x, y)
        self.prev_pos = pygame.Vector2(x, y)
        self.speed = speed
        self.size = size

    def move(self, keys):
        movement = pygame.Vector2(
            (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * self.speed,
            (keys[pygame.K_DOWN] - keys[pygame.K_UP]) * self.speed
        )
        self.prev_pos = self.pos.copy()
        self.pos += movement

    def check_collision(self, obstacles):
        player_rect = pygame.Rect(
            self.pos.x - self.size, self.pos.y - self.size,
            self.size * 2, self.size * 2
        )
        return any(player_rect.colliderect(obstacle.rect) for obstacle in obstacles)

    def draw(self, surface):
        pygame.draw.circle(surface, "blue", (int(self.pos.x), int(self.pos.y)), self.size)


class NPC:
    def __init__(self, x, y, speed=5, size=20):
        self.pos = pygame.Vector2(x, y)
        self.prev_pos = pygame.Vector2(x, y)
        self.speed = speed
        self.size = size
        self.player_movements = []
        self.path = None
        self.path_planner_res = 40

    def perceive_player(self, player, screen):
        player_pos = utils.screen2planner(player.pos, screen, self.path_planner_res)
        if not self.player_movements or player_pos != self.player_movements[-1]:
            self.player_movements.append(player_pos)
        self.player_size = utils.screen2planner(player.size, screen, self.path_planner_res)

    def perceive_goals(self, goals, screen):
        self.goals_pos = [utils.screen2planner(goal.pos, screen, self.path_planner_res) for goal in goals]

    def obstacle_repr(self, obstacle, screen):
        """
        Normalize parameters between 0 and 1 according to screen dimensions
        Generate tuple (x, y, radius) for circle representation
        """
        x_norm = obstacle.rect.x / screen.get_width()
        y_norm = obstacle.rect.y / screen.get_height()
        w_norm = obstacle.rect.width / screen.get_width()
        h_norm = obstacle.rect.height / screen.get_height()
        radius = max(1, utils.screen2planner(min(w_norm, h_norm) * 0.5, screen, self.path_planner_res))
        
        if w_norm > h_norm:
            n_tuples = int(3 * w_norm / h_norm)
            repr = [(x + radius, obstacle.rect.y, radius) for x in np.linspace(obstacle.rect.x, obstacle.rect.x + obstacle.rect.width, n_tuples)]
        else:
            n_tuples = int(3 * h_norm / w_norm)
            repr = [(x_norm, obstacle.rect.y + radius, radius) for y in np.linspace(obstacle.rect.y, obstacle.rect.y + obstacle.rect.height, n_tuples)]
        # return [(int(self.path_planner_res * tupl[0]), int(self.path_planner_res * tupl[1]), int(self.path_planner_res * tupl[2])) for tupl in repr]
        return [utils.screen2planner(tupl[:2], screen, self.path_planner_res) + (tupl[2],) for tupl in repr]

    def perceive_obstacles(self, obstacles, screen):
        self.obstacle_circles = []
        for obstacle in obstacles:
            obstacle_repr = self.obstacle_repr(obstacle, screen)
            self.obstacle_circles.extend(obstacle_repr)
    
    def make_path_planner(self, player):
        def plan_path(start_x, start_y, goal_x, goal_y):
            return RRT(start = [start_x, start_y],
                        goal = [goal_x, goal_y], 
                        obstacle_list = self.obstacle_circles, 
                        expand_dis=20.0,
                        rand_area = [0, self.path_planner_res],
                        play_area = (0, self.path_planner_res, 0, self.path_planner_res),
                        robot_radius = self.player_size)
        self.path_planner = plan_path

    def replan(self, player, screen):
        self.perceive_player(player, screen)
        if self.path is not None:
            goal_has_changed = False
            # goal_has_changed = self.path[-1] != self.goals_pos[0]
            # if goal_has_changed:
                # pass
                # print("Goal has changed")
            player_has_moved = list(self.player_movements[-1]) != self.path[0]
            if goal_has_changed or player_has_moved:
                self.plan(player, screen)
        else:
            self.plan(player, screen)
        self.draw_path(screen)
        
    def plan(self, player, screen):
        planner_player_pos = self.player_movements[-1]
        planner_goal_pos = self.goals_pos[0]
        self.path = self.path_planner(planner_player_pos[0], planner_player_pos[1], planner_goal_pos[0], planner_goal_pos[1]).planning(animation=True)
        if self.path:
            self.path = list(reversed(self.path)) # Reverse path to start from player position
    
    def draw_path(self, screen):
        if self.path:
            for i in range(len(self.path) - 1):
                segment_start = utils.planner2screen((self.path[i][0], self.path[i][1]), screen, self.path_planner_res)
                segment_end = utils.planner2screen((self.path[i + 1][0], self.path[i + 1][1]), screen, self.path_planner_res)
                pygame.draw.line(screen, "red", segment_start, segment_end, 2)

    def infer_goal_of_player(self, player):
        if len(self.player_movements) > 2:
            pass

    def move(self, keys):
        movement = pygame.Vector2(
            (keys[pygame.K_d] - keys[pygame.K_a]) * self.speed,
            (keys[pygame.K_s] - keys[pygame.K_w]) * self.speed
        )
        self.prev_pos = self.pos.copy()
        self.pos += movement

    def check_collision(self, obstacles):
        player_rect = pygame.Rect(
            self.pos.x - self.size, self.pos.y - self.size,
            self.size * 2, self.size * 2
        )
        return any(player_rect.colliderect(obstacle.rect) for obstacle in obstacles)

    def draw(self, surface):
        pygame.draw.circle(surface, "red", (int(self.pos.x), int(self.pos.y)), self.size)


class Obstacle:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)

    def draw(self, surface):
        pygame.draw.rect(surface, "black", self.rect)

class Goal:
    def __init__(self, x, y, screen, radius=30):
        self.pos = pygame.Vector2(x, y)
        self.radius = radius

    def check_reached(self, player):
        return player.pos.distance_to(self.pos) < self.radius

    def draw(self, surface):
        pygame.draw.circle(surface, "green", (int(self.pos.x), int(self.pos.y)), self.radius)

# Run the game
if __name__ == "__main__":
    game = Game()
    game.run()