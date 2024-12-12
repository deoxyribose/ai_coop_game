import pygame
import numpyro
import jax
import jax.numpy as jnp
from jax.lax import while_loop
import jax.random as random
from jax import vmap
import matplotlib.pyplot as plt


def screen2planner(obstacle, screen):
    return jnp.cumsum(obstacle, axis=0) / jnp.array([screen.get_width(), screen.get_height()])

def collides(point, obstacles):
    if point.ndim == 1 and obstacles.ndim == 2:
        # assert jnp.all(obstacles[0] < obstacles[1]), f"First corner must be below and to the left of the second corner, but are {obstacles}"

        # check if point is inside the current obstacle
        x, y = point
        x_inside = (x >= obstacles[0, 0]) & (x <= obstacles[1, 0])
        y_inside = (y >= obstacles[0, 1]) & (y <= obstacles[1, 1])
        return x_inside & y_inside
    elif point.ndim == 1 and obstacles.ndim > 2:
        # check if point is inside any of the obstacles
        return jnp.any(vmap(collides, in_axes=(None, 0))(point, obstacles))
    else:
        print("Wrong dims")

def other_two_corners(obstacle):
    return jnp.array([[obstacle[0, 0], obstacle[1, 1]], [obstacle[1, 0], obstacle[0, 1]]])

def line_collides(point_1, point_2, obstacles):
    if obstacles.ndim == 2:
        # find intersection of the line between point_1 and point_2
        # with the line between the two corners of the obstacle
        # by solving a system of linear equations
        # if the intersection is within the line segment between the two points
        # then the line collides with the obstacle

        # line 1: y = m1 * x + c1
        # line 2: y = m2 * x + c2
        # m1 = (y2 - y1) / (x2 - x1)
        # m2 = (y4 - y3) / (x4 - x3)
        # c1 = y1 - m1 * x1
        # c2 = y3 - m2 * x3
        # x = (c2 - c1) / (m1 - m2)
        # y = m1 * x + c1
        # intersection = (x, y)

        # set up matrix and solve Ax = b
        # A = [[m1, -1], [m2, -1]]
        # b = [-c1, -c2]
        m1 = (point_2[1] - point_1[1]) / (point_2[0] - point_1[0])
        m2 = (obstacles[1, 1] - obstacles[0, 1]) / (obstacles[1, 0] - obstacles[0, 0])
        c1 = point_1[1] - m1 * point_1[0]
        c2 = obstacles[0, 1] - m2 * obstacles[0, 0]
        A = jnp.array([[m1, -1], [m2, -1]])
        b = jnp.array([-c1, -c2])
        x = jnp.linalg.solve(A, b)

        # check if intersection is within the line segment
        # and within the obstacle
        return jnp.logical_and(collides(x, obstacles), jnp.all(jnp.logical_and(x >= jnp.min(jnp.array([point_1, point_2]), axis=0), x <= jnp.max(jnp.array([point_1, point_2]), axis=0))))
    elif obstacles.ndim == 3:
        obstacles = jnp.concatenate([obstacles, vmap(other_two_corners)(obstacles)], axis=0)
        return jnp.any(vmap(lambda obstacle: line_collides(point_1, point_2, obstacle))(obstacles))
@jax.jit
def find_nearest(tree_nodes, point):
    distances = jnp.linalg.norm(tree_nodes - point, axis=-1)
    return jnp.argmin(distances)

@jax.jit
def rejection_sample(obstacles, rng_key):
    """
    Sample a point uniformly at random in the unit square, and reject it if it collides with any of
    the obstacles.
    """
    def loop_body(state):
        point, key = state
        new_key, subkey = random.split(key)
        new_point = random.uniform(subkey, (2,))
        return new_point, new_key
    
    def condition(state):
        point, _ = state
        return collides(point, obstacles)

    init_point = random.uniform(rng_key, (2,))
    init_key = random.split(rng_key)[1]
    
    final_state = while_loop(condition, loop_body, (init_point, init_key))
    return final_state[0]

@jax.jit
def expand_tree(carry, i):
    """
    Sample a point uniformly in non-obstacle space
    Find the nearest point in the tree
    If the edge between the two points is collision-free
    Add the point to the tree
    """
    tree_nodes, tree_edges, obstacles, goal, rng_key = carry
    
    # Sample a point not in obstacles
    next_point = rejection_sample(obstacles, rng_key)
    rng_key = random.split(rng_key)[1]
    
    # Find nearest node
    nearest_idx = find_nearest(tree_nodes, next_point)
    nearest_point = tree_nodes[nearest_idx]

    # Check collision once
    is_collision = line_collides(nearest_point, next_point, obstacles)
    
    # Use where to avoid branch divergence
    new_tree_nodes = jnp.where(is_collision, tree_nodes, tree_nodes.at[i].set(next_point))
    new_tree_edges = jnp.where(
        is_collision, 
        tree_edges, 
        tree_edges.at[i].set(jnp.array([nearest_idx, i], dtype=jnp.int32))
    )

    return (new_tree_nodes, new_tree_edges, obstacles, goal, rng_key), None


@jax.jit
def go_to_goal(carry, i):

    tree_nodes, tree_edges, obstacles, goal, rng_key = carry

    # Find nearest node
    nearest_idx = find_nearest(tree_nodes, goal)

    return (tree_nodes.at[i].set(goal), 
            tree_edges.at[i].set(jnp.array([nearest_idx, i], dtype=jnp.int32)), 
            obstacles, 
            goal, 
            rng_key), None

@jax.jit
def go_to_goal_or_expand_tree(carry, i):

    tree_nodes, tree_edges, obstacles, goal, rng_key = carry

    # Find nearest node
    nearest_idx = find_nearest(tree_nodes, goal)
    nearest_point = tree_nodes[nearest_idx]

    # Check collision once
    is_collision = line_collides(nearest_point, goal, obstacles)
    print(is_collision)

    return jax.lax.cond(
        is_collision,
        lambda carry_i: expand_tree(carry_i[0], carry_i[1]),
        lambda carry_i: go_to_goal(carry_i[0], carry_i[1]),
        ((tree_nodes, tree_edges, obstacles, goal, rng_key), i),
    )

@jax.jit
def goal_is_not_reached(carry_i):
    carry, i = carry_i
    tree_nodes, tree_edges, obstacles, goal, rng_key = carry
    
    # Find nearest node
    nearest_idx = find_nearest(tree_nodes, goal)
    nearest_point = tree_nodes[nearest_idx]

    # Check collision once
    return line_collides(nearest_point, goal, obstacles)


def path(start, goal, obstacles, screen, rng_key):
    start = jnp.array(start) / jnp.array([screen.get_width(), screen.get_height()])
    start = numpyro.sample("p_0", numpyro.distributions.Uniform(0, 1).expand_by((2,)), obs = start)
    
    goal = jnp.array(goal) / jnp.array([screen.get_width(), screen.get_height()])
    goal = numpyro.sample("p_goal", numpyro.distributions.Uniform(0, 1).expand_by((2,)), obs = goal)

    obstacles = vmap(screen2planner, in_axes=(0, None))(jnp.array(obstacles), screen)

    n_nodes = 40

    tree_nodes = jnp.zeros((n_nodes, 2))
    tree_nodes = tree_nodes.at[0].set(start)
    tree_edges = jnp.zeros((n_nodes, 2), dtype=jnp.int32)

    # (final_tree_nodes, final_tree_edges, obstacles, goal, rng_keys),_ = jax.lax.scan(
    #     expand_tree, (tree_nodes, tree_edges, obstacles, goal, rng_key), jnp.arange(1, n_nodes + 1)
    # )

    (final_tree_nodes, final_tree_edges, obstacles, goal, rng_keys),_ = jax.lax.scan(
        go_to_goal_or_expand_tree, (tree_nodes, tree_edges, obstacles, goal, rng_key), jnp.arange(1, n_nodes + 1)
    )

    # (final_tree_nodes, final_tree_edges, obstacles, goal, rng_keys),_ = jax.lax.while_loop(
    #     goal_is_not_reached,
    #     lambda carry_i: (expand_tree(carry_i[0], carry_i[1])[0], carry_i[1]+1),
    #     ((tree_nodes, tree_edges, obstacles, goal, rng_key), 0),
    # )


    numpyro.deterministic("nodes", final_tree_nodes)
    numpyro.deterministic("edges", final_tree_edges)

def sample_path(*args, rng_key = random.PRNGKey(0)):
    rng_key = random.split(rng_key)[0]
    
    # seed and trace
    # with jax.disable_jit():
    trace = numpyro.handlers.trace(numpyro.handlers.seed(path, rng_key)).get_trace(*args, rng_key = rng_key)
    
    # build a dict from the traced nodes and edges
    nodes = trace['nodes']['value'].tolist()
    print(nodes)
    edges = trace['edges']['value'].tolist()
    tree = {}
    for i, edge in enumerate(edges):
        if edge == [0, 0]:
            continue
        if i == 0:
            tree[tuple(nodes[i])] = [tuple(nodes[edge[1]])]
        else:
            if tuple(nodes[edge[0]]) in tree:
                tree[tuple(nodes[edge[0]])].append(tuple(nodes[edge[1]]))
            else:
                tree[tuple(nodes[edge[0]])] = [tuple(nodes[edge[1]])]
    return tree, rng_key