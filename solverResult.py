import pickle

import numpy as np
from scipy import stats
from ortools.linear_solver import pywraplp


def getGlobalReward(noOfTasks, noOfUsers, mcsp_reward_expectation, user_reward_expectation):

    # Solver
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')


    # Variables
    # x[i, j] is an array of 0-1 variables, which will be 1
    # if worker i is assigned to task j.
    x = {}
    for i in range(noOfUsers):
        for j in range(noOfTasks):
            x[i, j] = solver.IntVar(0, 1, '')

    # Constraints
    # Each worker is assigned to at most one task.
    for i in range(noOfUsers):
        solver.Add(solver.Sum([x[i, j] for j in range(noOfTasks)]) <= 1)

    # Each task is assigned to exactly one worker.
    for j in range(noOfTasks):
        solver.Add(solver.Sum([x[i, j] for i in range(noOfUsers)]) <= 1)

    # Objective
    objective_terms = []
    # users reward
    for i in range(noOfUsers):
        for j in range(noOfTasks):
            objective_terms.append(x[i, j]*user_reward_expectation[i][j])
    # mcsp reward
    for i in range(noOfUsers):
        for j in range(noOfTasks):
            objective_terms.append(x[i, j]*mcsp_reward_expectation[i][j])
    solver.Maximize(solver.Sum(objective_terms))

    # Solve
    status = solver.Solve()

    # Print solution.
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print(f'Total reward = {solver.Objective().Value()}\n')
        for i in range(noOfUsers):
            for j in range(noOfTasks):
                # Test if x[i,j] is 1 (with tolerance for floating point arithmetic).
                if x[i, j].solution_value() > 0.5:
                    print(f'Worker {i} assigned to task {j}.' +
                          f' mcsp reward: {mcsp_reward_expectation[i][j]} user reward: {user_reward_expectation[i][j]}')
    else:
        print('No solution found.')
        Exception("No solution found.")


    return solver.Objective().Value()


if __name__ == '__main__':
    getGlobalReward()