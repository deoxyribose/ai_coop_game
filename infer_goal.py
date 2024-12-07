import pygame
import numpyro
import jax
import jax.numpy as jnp
from jax.lax import while_loop
import jax.random as random
from jax import vmap

def screen2planner(obstacle, screen):
    return jnp.cumsum(obstacle, axis=0) / jnp.array([screen.get_width(), screen.get_height()])

def collides(point, obstacles):
    if point.ndim == 1 and obstacles.ndim == 2:
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

@jax.jit
def rejection_sample(obstacles, rng_key):
    """
    Sample a point uniformly at random in the unit square, and reject it if it collides with any of
    the obstacles.
    """
    init_point = random.uniform(rng_key, (2,))
    
    return while_loop(lambda point_rng_key: collides(point_rng_key[0], obstacles), 
                      lambda point_rng_key: (random.uniform(point_rng_key[1], (2,)), random.split(point_rng_key[1])[1]), 
                      (init_point, rng_key))[0]

def path(start, obstacles, screen):
    start = jnp.array(start) / jnp.array([screen.get_width(), screen.get_height()])
    start = numpyro.sample("p_0", numpyro.distributions.Uniform(0, 1).expand_by((2,)), obs = start)
    obstacles = vmap(screen2planner, in_axes=(0, None))(jnp.array(obstacles), screen)

    rng_key = random.PRNGKey(0)
    tree = {tuple(start.tolist()): []}
    for i in range(1, 20):
        rng_key = random.split(rng_key)[0]
        next = numpyro.deterministic(f"p_{i}", rejection_sample(obstacles, rng_key))
        all_points = jnp.array(list(tree.keys()))
        nearest = all_points[jnp.argmin(jnp.linalg.norm(all_points - next, axis=-1))]
        # check if every point on the line segment between nearest and next is collision-free
        
        tree[tuple(nearest.tolist())].append(next)
        tree[tuple(next.tolist())] = []
    numpyro.deterministic("tree", tree)

def sample_path(*args):
    # seed and trace
    trace = numpyro.handlers.trace(numpyro.handlers.seed(path, random.PRNGKey(0))).get_trace(*args)
    # return [trace[f'p_{i}']['value'] for i in range(20)]
    return trace['tree']['value']