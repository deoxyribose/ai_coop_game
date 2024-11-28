import pygame

def screen2planner(coords, screen, path_planner_res):
    if isinstance(coords, pygame.Vector2):
        return int(path_planner_res * coords.x / screen.get_width()), int(path_planner_res * (1 - coords.y / screen.get_height()))
    elif isinstance(coords, tuple):
        return int(path_planner_res * coords[0] / screen.get_width()), int(path_planner_res * (1 - coords[1] / screen.get_height()))
    elif isinstance(coords, int) or isinstance(coords, float):
        return path_planner_res * coords / ((screen.get_width() + screen.get_height()) / 2)
    
def planner2screen(coords, screen, path_planner_res):
    if isinstance(coords, pygame.Vector2):
        return coords.x * screen.get_width() / path_planner_res, (1 - coords.y / path_planner_res) * screen.get_height()
    elif isinstance(coords, tuple):
        return coords[0] * screen.get_width() / path_planner_res, (1 - coords[1] / path_planner_res) * screen.get_height()
