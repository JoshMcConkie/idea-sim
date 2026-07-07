'''Greedy coverage solvers over precomputed path sets with known utility.

All solvers share the same sequential greedy assignment loop
(`greedy_assign_paths`); they differ in planning horizon and in how the
agent ordering is chosen:

- `full_horizon_greedy_solve`: plan full-length paths in one pass.
- `rolling_horizon_greedy_solve`: plan in chunks, carrying coverage state
  between rounds.
- `best_order_full_horizon_greedy_solve`: exhaustively try every agent
  ordering and keep the best (expensive baseline).
'''
import time
from itertools import permutations

import numpy as np

from coverage_planner.policies.tools import best_marginal_choice, build_path_model, commit_paths_to_agents
from coverage_planner.env import Model, Result


def greedy_assign_paths(model, agent_order, prior_coverage=None):
    '''Greedily assign one path per agent, in the given order, maximizing
    marginal coverage gain. Does not commit paths to the agents; callers
    decide which assignment (if any) to commit.

    returns:
        score: coverage score of the joint assignment.
        path_ids: chosen path indices, one per agent.
    '''
    if agent_order is None:
        raise ValueError("Sequential solver requires an agent_order; got None.")
    path_ids = []
    for agent in agent_order:
        score, path_ids = best_marginal_choice(model.agent_path_dict[agent],
                                                path_ids,
                                                model.util_mat, prior_coverage)
    return score, path_ids


def full_horizon_greedy_solve(model: Model) -> Result:
    '''Sequential submodular maximization solver. Alters in place the attributes of `model`,
    including the attached agents' paths. Returns maximized utility score.

    args:
        model: a Model instance containing the utility matrix (Nemhauser et al, 1978),
        available agent allocations, etc. It is transpose from Nemhauser et al.'s design.

    returns:
        result: a Result with the sequentially maximized objective/utility score.
    '''
    start_time = time.perf_counter()
    score, path_ids = greedy_assign_paths(model, model.agent_order)
    paths_by_agent = commit_paths_to_agents(model, path_ids)
    end_time = time.perf_counter()
    result = Result(model.grid.grid,paths_by_agent,
                  score,full_horizon_greedy_solve,
                  runtime=end_time-start_time,chosen_path_ids=path_ids,
                  metadata=None,agent_order=model.agent_order,steps=model.steps)
    model.grid.reset_grid()
    return result


def rolling_horizon_greedy_solve(model: Model, chunksize: int) -> Result:
    '''Rolling-horizon variant: plan in rounds of at most `chunksize` steps,
    carrying prior coverage state (and agent end positions) between rounds.
    '''
    start_time = time.perf_counter()
    full_rounds, remainder = divmod(model.steps,chunksize)
    rounds = [chunksize] * full_rounds
    if remainder:
        rounds.append(remainder)
    prior_coverage_state = None

    for chunk in rounds:
        round_model = build_path_model(model.grid, model.objective,chunk, model.agent_order)
        score, path_ids = greedy_assign_paths(round_model, round_model.agent_order,
                                              prior_coverage_state)
        paths_by_agent = commit_paths_to_agents(round_model, path_ids)

        if prior_coverage_state is None:
            prior_coverage_state = np.zeros(round_model.util_mat.shape[1])
        prior_coverage_state += round_model.util_mat[path_ids].max(axis=0)
        
    end_time = time.perf_counter()
    result = Result(model.grid.grid,paths_by_agent,
                  model.grid.get_score(),rolling_horizon_greedy_solve,
                  runtime=end_time-start_time,chosen_path_ids=path_ids,
                  metadata={"Round-steps":rounds},
                  agent_order=model.agent_order,steps=model.steps,)
    model.grid.reset_grid()
    return result

#TODO: Implement this function without the adjacency constraint (e.g. each step can be any viable coordinate that
# still satisfies the path length constraint)
def rolling_horizon_no_adjacency_constraint(model: Model, chunksize: int) -> Result:
    raise NotImplementedError


def best_order_full_horizon_greedy_solve(model: Model) -> Result:
    '''Try the full-horizon greedy assignment under every agent ordering and
    keep the best-scoring one. Factorial in the number of agents.
    '''
    start_time = time.perf_counter()
    solutions = []
    for agent_order in permutations(model.agent_path_dict.keys()):
        score, path_ids = greedy_assign_paths(model, agent_order)
        solutions.append((score, path_ids, agent_order))

    score,path_ids,agent_order = max(solutions)
    paths_by_agent = commit_paths_to_agents(model,path_ids)
    end_time = time.perf_counter()
    result = Result(model.grid.grid,paths_by_agent,
                  score,best_order_full_horizon_greedy_solve,
                  runtime=end_time-start_time,chosen_path_ids=path_ids,
                  metadata=None,agent_order=agent_order, steps=model.steps)
    model.grid.reset_grid()
    return result
