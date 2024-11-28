from PythonRobotics.PathPlanning.RRT.rrt import RRT
import pygame
import numpyro



def user_model(goals):
    start_x = numpyro.sample("start_x", numpyro.distributions.Uniform(0, 1))
    start_y = numpyro.sample("start_y", numpyro.distributions.Uniform(0, 1))

    n_goals = len(goals)
    goal = numpyro.sample("goal", numpyro.distributions.Categorical(probs=[1/n_goals]*n_goals))
    goal_x, goal_y = goals[goal].planner_pos

    path = rrt(start_x, start_y, goal_x, goal_y)