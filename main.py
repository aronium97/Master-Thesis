# This is a sample Python script.
import copy

import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy
from joblib import Parallel, delayed
#import ray
from scipy.stats import norm
from scipy import stats
from solverResult import getGlobalReward


def stableMatching(n, m, menPreferences, womenPreferences):
    # Initially, all n men are unmarried
    unmarriedMen = list(range(n))
    # None of the men has a spouse yet, we denote this by the value None
    manSpouse = [None] * n
    # None of the women has a spouse yet, we denote this by the value None
    womanSpouse = [None] * m
    # Each man made 0 proposals, which means that
    # his next proposal will be to the woman number 0 in his list
    nextManChoice = [0] * n

    # While there exists at least one unmarried man:
    while unmarriedMen :
        # Pick an arbitrary unmarried man
        he = unmarriedMen[0]
        if nextManChoice[he] == m:
            # this man will not find a woman
            manSpouse[he] = None
            unmarriedMen.pop(0)
        else:
            # Store his ranking in this variable for convenience
            hisPreferences = menPreferences[he]
            # Find a woman to propose to
            she = hisPreferences[nextManChoice[he]]
            # Store her ranking in this variable for convenience
            herPreferences = womenPreferences[she]
            # Find the present husband of the selected woman (it might be None)
            currentHusband = womanSpouse[she]

            # Now "he" proposes to "she".
            # Decide whether "she" accepts, and update the following fields
            # 1. manSpouse
            # 2. womanSpouse
            # 3. unmarriedMen
            # 4. nextManChoice
            if currentHusband == None:
                # No Husband case
                # "She" accepts any proposal
                womanSpouse[she] = he
                manSpouse[he] = she
                # "His" nextchoice is the next woman
                # in the hisPreferences list
                nextManChoice[he] = nextManChoice[he] + 1
                # Delete "him" from the
                # Unmarried list
                unmarriedMen.pop(0)
            else:
                # Husband exists
                # Check the preferences of the
                # current husband and that of the proposed man's
                currentIndex = herPreferences.index(currentHusband)
                hisIndex = herPreferences.index(he)
                # Accept the proposal if
                # "he" has higher preference in the herPreference list
                if currentIndex > hisIndex:
                    # New stable match is found for "her"
                    womanSpouse[she] = he
                    manSpouse[he] = she
                    nextManChoice[he] = nextManChoice[he] + 1
                    # Pop the newly wed husband
                    unmarriedMen.pop(0)
                    # Now the previous husband is unmarried add
                    # him to the unmarried list
                    unmarriedMen.insert(0, currentHusband)
                else:
                    nextManChoice[he] = nextManChoice[he] + 1

    return manSpouse

# Complexity Upper Bound : O(n^2)


def test_on_preference(task_duration, estimated_task_duration, noOfTimesVisited, t):
    assigned_preference = 0#np.sum(task_duration >= np.choose(np.argmax(estimated_task_duration, 1), task_duration.T).reshape(-1, 1), 1)
    return np.mean(assigned_preference)


def ucb_ca(noOfUsers, noOfTasks, T, task_duration, mcsp_utility, marge, lambda_var, deadline, explore_var, sampling_noise, enableNoAccessCount, countDegradeRate, countStartTime, countInfluence_var, countUntilAdjustment, noAccessMode, countEndTime, considerPreferenceMCSP_var, maxAllowedMarge, activateBurst, forgettingDuration_var, epsilon_greedy, takeNothingIfNegativeReward, rewardSensing, costPerSecond):
    # init
    noOfTimesChoosen = np.zeros([noOfUsers, noOfTasks]).astype(int)
    noOfTimesVisited = np.zeros([noOfUsers, noOfTasks]).astype(int)
    choosenTasks = -1 * np.ones(noOfUsers).astype(int)
    performedTasks = -1 * np.ones(noOfTasks).astype(int)
    lastBidOnTask = -1 * np.ones(noOfTasks)
    usersCurrentBidOnTask = -1 * np.ones(noOfUsers)
    rewardMeasurements = np.zeros([noOfUsers, T])
    globalRewardMeasurement = np.zeros(T)
    taskMeasurements = -1 * np.ones([noOfUsers, T])
    taskPreferenceMeasurements = np.zeros(T)
    choosenTasksMeasurements = np.zeros([noOfUsers, T])
    estimated_task_reward = np.zeros(([noOfUsers,noOfTasks]))
    estimated_task_duration = np.zeros(([noOfUsers,noOfTasks]))
    noAccessCounter = np.zeros([noOfUsers, noOfTasks])
    overDeadlineCounter = np.zeros([noOfUsers, noOfTasks])
    lastBidAccepted = np.zeros([noOfUsers, noOfTasks])
    lastWasAccepted = np.zeros([noOfUsers,noOfTasks])
    freeSensingTried = np.zeros([noOfUsers, noOfTasks])
    freeSensingDone = np.zeros(T)
    # algorithm
    for t in range(1, T):
        for i in range(0, noOfUsers):
            if t == 1:
                j = np.random.randint(0, noOfTasks)
                choosenTasks[i] = j
                usersCurrentBidOnTask[i] = estimated_task_duration[i][choosenTasks[i]] * (marge + 1)
            else:
                D = (np.random.uniform(0, 1) <= lambda_var) * 1
                if D == 1:
                    True == True
                    # do nothing
                else:
                    # estimate cost
                    plausibleTasks = np.zeros(noOfTasks)
                    testedBids = -1*np.ones(noOfTasks)
                    for j in range(0, noOfTasks):
                        # ----- caluclate bid
                        freeSensingTried[i, j] = 0 # reset
                        if enableNoAccessCount==False:
                            # off
                            task_duration_user = estimated_task_duration[i][j]
                        else:
                            # check if user should explore
                            userShouldSenseFree= (np.random.uniform(0, 1) <= (noAccessCounter[i][j])/countUntilAdjustment)#*(countEndTime>t) #noAccessCounter[i][j] > countUntilAdjustment # (np.random.uniform(0, 1) <= (noAccessCounter[i][j])/countUntilAdjustment) # stochastic
                            if userShouldSenseFree:
                                task_duration_user = estimated_task_duration[i][j]*0.7
                                freeSensingTried[i, j] = 1
                            else:
                                task_duration_user = estimated_task_duration[i][j]
                        testBid = task_duration_user * (marge + 1) * costPerSecond

                        # ------ test if task is plausible
                        taskIsPlausible = True
                        considerPreferenceMCSP = np.random.uniform(0, 1) <= considerPreferenceMCSP_var
                        if considerPreferenceMCSP:
                            taskIsPlausible = taskIsPlausible and ((testBid - mcsp_utility[i][j]) <= lastBidOnTask[j] or performedTasks[j] == i)
                        if taskIsPlausible:
                            plausibleTasks[j] = 1
                            testedBids[j] = testBid - mcsp_utility[i][j]
                            #noAccessCounter[i][j] = 0#np.max([0, noAccessCounter[i][j] - countDegradeRate])
                            #lastBidAccepted[i][j] = np.max([lastBidAccepted[i][j], testBid])
                            lastWasAccepted[i][j] = True
                        else:
                            noAccessCounter[i][j] += not(np.random.uniform(0, 1) <= t/(countEndTime))*t/(countInfluence_var) # determmnistic: (countEndTime>t)*t/(countInfluence_var)#1*(countEndTime-t)/t
                            lastWasAccepted[i][j] = False
                    # ---- decide for task and pull
                    ucbBound = np.copy(estimated_task_reward[i])
                    ucbBound[plausibleTasks == 0] = np.nan
                    # epsilon greedy
                    if epsilon_greedy == True:
                        explore = (np.random.uniform(0, 1) <= 1 / t * explore_var) * 1
                    else:
                        # dont consider best arm
                        explore = 1
                    doNotChooseTask = np.sum(plausibleTasks) == 0 or (takeNothingIfNegativeReward == True)*(((np.logical_or(ucbBound<0, plausibleTasks==0) ==True).all() and explore == 0))
                    if doNotChooseTask:
                        # no task can be choosen
                        choosenTasks[i] = -1
                        usersCurrentBidOnTask[i] = np.nan
                    else:
                        if explore == 1:
                            choosenTasks[i] = np.random.choice((np.argwhere(plausibleTasks > 0)).flatten())
                        else:
                            choosenTasks[i] = np.random.choice((np.argwhere(ucbBound == np.nanmax(ucbBound))).flatten())
                        usersCurrentBidOnTask[i] = testedBids[choosenTasks[i]]#np.random.normal(loc=estimated_task_duration[i][choosenTasks[i]], scale=0)* (marge + 1)#
                        if testedBids[choosenTasks[i]] == -1:
                            raise RuntimeError("non plausible task was choosen")

        # decide for winners
        tasksHaveUser = np.array([False] * noOfTasks)
        userIterate = np.arange(noOfUsers)
        np.random.shuffle(userIterate)
        for i in userIterate:
            choosenTasksMeasurements[i][t] = choosenTasks[i]
            if ~(choosenTasks[i] == -1):
                noOfTimesChoosen[i][choosenTasks[i]] += 1
                # check if bid is the lowest & is less than max allowed bid
                if (np.min(usersCurrentBidOnTask[choosenTasks == choosenTasks[i]]) >= usersCurrentBidOnTask[i]):
                    performedTasks[choosenTasks[i]] = i
                    lastBidOnTask[choosenTasks[i]] = usersCurrentBidOnTask[i]
                    tasksHaveUser[choosenTasks[i]] = True
        # reset tasks without users
        lastBidOnTask[~tasksHaveUser] = np.inf
        performedTasks[~tasksHaveUser] = -1
        # update no of visits
        for i in range(0, noOfUsers):
            #if i == 4:
            #    estimated_task_reward[i] = 1 + deadline
            if i in performedTasks:
                j = np.where(performedTasks == i)[0][0]
                if freeSensingTried[i,j]:
                    freeSensingDone[t] += 1
                    if t > 550:
                        print("sc")
                noOfTimesVisited[i][j] += 1
                forgettingDuration_var_i = forgettingDuration_var[1]**noOfTimesVisited[i][j]*forgettingDuration_var[0]
                burst = 1.2*(i==4 and t<30)*(activateBurst == True)
                sampleTaskDuration = np.max([np.random.normal(loc=task_duration[i][j] + burst, scale=sampling_noise),0])
                estimated_task_duration[i][j] = sampleTaskDuration*forgettingDuration_var_i + estimated_task_duration[i][j]*(1-forgettingDuration_var_i)
                if rewardSensing == 0:
                    # original
                    reward = np.min([mcsp_utility[i][j], usersCurrentBidOnTask[i] + mcsp_utility[i][j],(1+maxAllowedMarge)*task_duration[i][j]])*(sampleTaskDuration <= deadline) - sampleTaskDuration
                    reward *= costPerSecond
                    estimated_task_reward[i][j] = reward*forgettingDuration_var_i + estimated_task_reward[i][j]*(1-forgettingDuration_var_i)
                elif rewardSensing == 1:
                    # ignore reward
                    if usersCurrentBidOnTask[i] > 0:
                        reward = np.min([mcsp_utility[i][j], usersCurrentBidOnTask[i] + mcsp_utility[i][j], (1 + maxAllowedMarge) * task_duration[i][j]]) * (sampleTaskDuration <= deadline) - sampleTaskDuration
                        reward *= costPerSecond
                        estimated_task_reward[i][j] = reward * forgettingDuration_var_i + estimated_task_reward[i][j] * (1 - forgettingDuration_var_i)
                elif rewardSensing == 2:
                    # pseudo reward with deadlines
                    if usersCurrentBidOnTask[i] > 0:
                        reward = np.min([mcsp_utility[i][j], usersCurrentBidOnTask[i] + mcsp_utility[i][j], (1 + maxAllowedMarge) * task_duration[i][j]]) * (sampleTaskDuration <= deadline) - sampleTaskDuration
                        reward *= costPerSecond
                        estimated_task_reward[i][j] = reward * forgettingDuration_var_i + estimated_task_reward[i][j] * (1 - forgettingDuration_var_i)
                    else:
                        reward = (1 + marge) * sampleTaskDuration * (sampleTaskDuration <= deadline) - sampleTaskDuration# (1 + marge) * sampleTaskDuration # pseudo reward
                        reward *= costPerSecond
                        estimated_task_reward[i][j] = reward * forgettingDuration_var_i + estimated_task_reward[i][j] * (1 - forgettingDuration_var_i)
                elif rewardSensing == 3:
                    # pseudo reward
                    if usersCurrentBidOnTask[i] > 0:
                        reward = np.min([mcsp_utility[i][j], usersCurrentBidOnTask[i] + mcsp_utility[i][j], (1 + maxAllowedMarge) * task_duration[i][j]]) * (sampleTaskDuration <= deadline) - sampleTaskDuration
                        reward *= costPerSecond
                        estimated_task_reward[i][j] = reward * forgettingDuration_var_i + estimated_task_reward[i][j] * (1 - forgettingDuration_var_i)
                    else:
                        reward = (1 + marge) * sampleTaskDuration # pseudo reward
                        reward *= costPerSecond
                        estimated_task_reward[i][j] = reward * forgettingDuration_var_i + estimated_task_reward[i][j] * (1 - forgettingDuration_var_i)
                rewardMeasurements[i][t] = np.min([mcsp_utility[i][j], usersCurrentBidOnTask[i] + mcsp_utility[i][j], costPerSecond*(1+maxAllowedMarge)*task_duration[i][j]])*(sampleTaskDuration <= deadline) - costPerSecond*sampleTaskDuration # reward # todo: reward hier rein oder wo anders?
                globalRewardMeasurement[t] +=  mcsp_utility[i][j]*(sampleTaskDuration <= deadline) - sampleTaskDuration*costPerSecond
                taskMeasurements[i][t] = j
                noAccessCounter[i][j] = 0
                overDeadlineCounter[i][j] = overDeadlineCounter[i][j] + (sampleTaskDuration > deadline)*1


        taskPreferenceMeasurements[t] = test_on_preference(task_duration, estimated_task_duration, noOfTimesVisited, t)

    return rewardMeasurements, taskPreferenceMeasurements, estimated_task_duration, choosenTasksMeasurements, taskMeasurements, estimated_task_reward, noOfTimesVisited, overDeadlineCounter, noOfTimesChoosen, globalRewardMeasurement, freeSensingDone


#@ray.remote
def doMonteCarrloIterations(noOfMonteCarloIterations, noOfUsers, noOfTasks, T, task_duration, mcsp_utility, marge, lambda_var, deadline, explore_var, sampling_noise, enableNoAccessCount, countDegradeRate, countStartTime, countInfluence_var, countUntilAdjustment, noAccessMode, countEndTime, considerPreferenceMCSP_var, maxAllowedMarge, activateBurst, forgettingDuration_var, epsilon_greedy, takeNothingIfNegativeReward, rewardSensing, costPerSecond):
    rewardMeasurements = {}
    taskPreferenceMeasurements = {}
    estimated_task_duration = {}
    choosenTasksMeasurements = {}
    taskMeasurements = {}
    estimated_task_reward = {}
    noOfTimesVisited = {}
    overDeadlineCounter = {}
    noOfTimesChoosen = {}
    globalRewardMeasurement = {}
    freeSensingDoneMeasurement = {}
    for m in range(noOfMonteCarloIterations):
        print(m)
        rewardMeasurements_i, taskPreferenceMeasurements_i, estimated_task_duration_i, choosenTasksMeasurements_i, taskMeasurements_i, estimated_task_reward_i, noOfTimesVisited_i, overDeadlineCounter_i, noOfTimesChoosen_i, globalRewardMeasurement_i, freeSensingDoneMeasurement_i = ucb_ca(noOfUsers, noOfTasks, T, task_duration, mcsp_utility, marge, lambda_var, deadline, explore_var, sampling_noise, enableNoAccessCount, countDegradeRate, countStartTime, countInfluence_var, countUntilAdjustment, noAccessMode, countEndTime, considerPreferenceMCSP_var, maxAllowedMarge, activateBurst, forgettingDuration_var, epsilon_greedy, takeNothingIfNegativeReward, rewardSensing, costPerSecond)
        rewardMeasurements[m] = rewardMeasurements_i
        taskPreferenceMeasurements[m] = taskPreferenceMeasurements_i
        estimated_task_duration[m] = estimated_task_duration_i
        choosenTasksMeasurements[m] = choosenTasksMeasurements_i
        taskMeasurements[m] = taskMeasurements_i
        estimated_task_reward[m] = estimated_task_reward_i
        noOfTimesVisited[m] = noOfTimesVisited_i
        overDeadlineCounter[m] = overDeadlineCounter_i
        noOfTimesChoosen[m] = noOfTimesChoosen_i
        globalRewardMeasurement[m] = globalRewardMeasurement_i
        freeSensingDoneMeasurement[m] = freeSensingDoneMeasurement_i
    return rewardMeasurements, taskPreferenceMeasurements, estimated_task_duration, choosenTasksMeasurements, taskMeasurements, estimated_task_reward, noOfTimesVisited, overDeadlineCounter, noOfTimesChoosen, globalRewardMeasurement, freeSensingDoneMeasurement





def print_hi(name):
    use_ray = False
    customName = "random"
    resultsFileName = "blabla1"

    noOfTasks = 70
    noOfUsers = 100

    deadline = 0.8

    revenuePerMbit = 0.09  # revenue fopr mcsp: €/Mbit for mcsp
    costPerSecond = 0.06 # cost for users:   €/sec for users

    T = 1200
    lambda_var = 0.1
    forgettingDuration_var = (1,0.99) # start, decay with time
    marge = 0.1
    maxAllowedMarge = 0.1
    explore_var = 0.7 # 100: start decrasing from t=100
    sampling_noise = 0.01#0.01
    activateBurst_e = [0, 0, 0,0,0,0,0,0]


    takeNothingIfNegativeReward_e = [True, True, True, True, True] # todo: true after a delay, to prevent task=-1 assignment if bids were to low at beginning
    enableNoAccessCount_e = [True, True, True, True, False]
    rewardSensing_i = [2, 2, 2, 2, 2] # 0: original, 1: ignore, 2: pseudo reward
    noAccessMode_e = [0, 0, 0, 0, 0]
    countDegradeRate = 8
    countStartTime = 0
    countEndTime_e = [400, 400,400, 400, 400] # used
    countInfluence_var = 100  # used    hint: 100 for 10m20 1000 for 20m100 etc
    countUntilAdjustment_e = [100, 15, 30, 60, 500] # used
    considerPreferenceMCSP_var_e = [1 , 1, 1, 1, 1] # prob. for considering mcsp prefernce list for plausible list (1= consider it always)

    epsilon_greedy_e = [True, True, True, True, True]

    noOfMonteCarloIterations = 200
    showMatrices = False
    checkForStability = False

    noOfExperiments = 5

    print("iterations: " + str(T) + " seconds: " + str(deadline*T))

    pickelFileName = "data/" + str(noOfTasks) + str(noOfUsers) + str(customName)
    with open(pickelFileName + ".pkl", 'rb') as f:
        data = pickle.load(f)
    t_processing = data[0]
    t_upload = data[1]
    mcsp_utility = data[2]*revenuePerMbit

    task_duration = t_processing+t_upload # todo: aufsplitten

    # check for order
    for i in range(0, noOfTasks):
        _, counts = np.unique(task_duration[:, i]+mcsp_utility[:,i], return_counts=True)
        if not (counts == 1).all():
            raise ValueError("no strict ordering of users" + str(task_duration))

    # Compute results: ----------------------------------------------------
    if use_ray:
        ray.init(local_mode=False)
        results = ray.get([doMonteCarrloIterations.remote(noOfMonteCarloIterations, noOfUsers, noOfTasks, T, task_duration,mcsp_utility, marge, lambda_var,
                                deadline, explore_var, sampling_noise, enableNoAccessCount_e[iExperiment],
                                countDegradeRate, countStartTime, countInfluence_var, countUntilAdjustment_e[iExperiment],
                                noAccessMode_e[iExperiment], countEndTime_e[iExperiment], considerPreferenceMCSP_var_e[iExperiment], maxAllowedMarge, activateBurst_e[iExperiment], forgettingDuration_var, epsilon_greedy_e[iExperiment], takeNothingIfNegativeReward_e[iExperiment], rewardSensing_e[iExperiment], costPerSecond) for iExperiment in range(noOfExperiments)])
        ray.shutdown()

    else:
        results = Parallel(n_jobs=noOfExperiments)(delayed(doMonteCarrloIterations)(noOfMonteCarloIterations, noOfUsers, noOfTasks, T, task_duration,mcsp_utility, marge, lambda_var,
                            deadline, explore_var, sampling_noise, enableNoAccessCount_e[iExperiment],
                            countDegradeRate, countStartTime, countInfluence_var, countUntilAdjustment_e[iExperiment],
                           noAccessMode_e[iExperiment], countEndTime_e[iExperiment], considerPreferenceMCSP_var_e[iExperiment], maxAllowedMarge, activateBurst_e[iExperiment], forgettingDuration_var, epsilon_greedy_e[iExperiment], takeNothingIfNegativeReward_e[iExperiment], rewardSensing_i[iExperiment], costPerSecond) for iExperiment in range(noOfExperiments))


    # Plot results: ----------------------------------------------------

    fig, axs = plt.subplots(3, 4)
    fig2, axs2 = plt.subplots(1, 4)
    if showMatrices: figMatrizes, axsMatrizes = plt.subplots(5, noOfExperiments + 1)

    data = []
    for idd in range(58):
        data.append([])
    for iExperiment in range(noOfExperiments):

        print("//////////// experiment " + str(iExperiment))

        # get from ray results
        rewardMeasurements, taskPreferenceMeasurements, estimated_task_durationMeasurements, choosenTasksMeasurements, taskMeasurements, estimated_task_rewardMeasurements, noOfTimesVisitedMeasurements, overDeadlineCounterMeasurements, noOfTimesChoosenMeasurements, globalRewardMeasurement, freeSensingDoneMeasurement = results[iExperiment]

        prefer_tasks = []
        prefer_users = []
        # get task duration with deadline incroprorated + mcsp utility
        if sampling_noise > 0:
            p_max_allowed = scipy.stats.norm.cdf(np.minimum(deadline*np.ones([noOfUsers,noOfTasks]), task_duration*(1+maxAllowedMarge)), loc=task_duration, scale=sampling_noise)
            p_0 = scipy.stats.norm.cdf(0, loc=task_duration, scale=sampling_noise)
            task_duration_real = np.multiply(p_max_allowed-p_0, task_duration)
            #task_duration_real, var, skew, kurt = truncnorm.stats(a, b, moments='mvsk', scale=sampling_noise, loc=task_duration)
        else:
            task_duration_real = np.multiply(task_duration, task_duration<=deadline)
        # more:
        optimum_deadlines = "hard" # todo: consider to use soft
        if optimum_deadlines == "soft":
            task_duration_with_deadlines_userview = task_duration_real#np.multiply(task_duration_real, (task_duration <= deadline) * 1)
            task_duration_with_deadlines_taskview = task_duration_real#np.multiply(task_duration_real, (task_duration <= deadline)) * 1 + (task_duration > deadline)*10000
        else:
            task_duration_with_deadlines_userview = np.multiply(task_duration_real, (task_duration <= deadline) * 1)
            task_duration_with_deadlines_taskview = np.multiply(task_duration_real, (task_duration <= deadline)) * 1 + (task_duration > deadline)*10000
        # incorporate utilities
        mcsp_expected_utility = np.multiply(mcsp_utility, p_max_allowed)
        task_duration_with_deadlines_taskview *= costPerSecond
        task_duration_with_deadlines_taskview -= mcsp_expected_utility
        task_duration_with_deadlines_userview *= costPerSecond
        # tasks preferences (tasks prefer shorter durations)
        for i in range(noOfTasks):
            prefer_tasks.append((np.argsort(task_duration_with_deadlines_taskview[:,i])).tolist())
        # users pferences
        for i in range(noOfUsers):
            prefer_users.append((np.flip(np.argsort(task_duration_with_deadlines_userview[i][:]))).tolist())

        # advantage: tasks
        users_of_tasks_pessimal_stablematch = stableMatching(noOfTasks, noOfUsers, prefer_tasks, prefer_users)
        # advantage: users
        tasks_of_users_optimal_stablematch = stableMatching(noOfUsers, noOfTasks, prefer_users, prefer_tasks)

        # calculate stable optimal reward of users
        meanOptimalReward = np.zeros(noOfUsers)
        meanOptimalGlobalReward = 0
        optimalAssignment = -1*np.ones(noOfUsers)
        for i in range(noOfUsers):
            if not(tasks_of_users_optimal_stablematch[i] == None):
                meanOptimalReward[i] = costPerSecond*marge*task_duration_real[i][tasks_of_users_optimal_stablematch[i]]
                meanOptimalGlobalReward += mcsp_expected_utility[i][tasks_of_users_optimal_stablematch[i]] - costPerSecond*task_duration[i][tasks_of_users_optimal_stablematch[i]]
                optimalAssignment[i] = tasks_of_users_optimal_stablematch[i]

        # calculate stable pessimal reward of users
        meanPessimalReward = np.zeros(noOfUsers)
        meanPessimalGlobalReward = 0
        pessimalAssignment = -1*np.ones(noOfUsers)
        for i in range(noOfTasks):
            meanPessimalReward[users_of_tasks_pessimal_stablematch[i]] = costPerSecond*marge * task_duration_real[users_of_tasks_pessimal_stablematch[i]][i]
            meanPessimalGlobalReward += mcsp_expected_utility[users_of_tasks_pessimal_stablematch[i]][i] - costPerSecond*task_duration[users_of_tasks_pessimal_stablematch[i]][i]
            pessimalAssignment[users_of_tasks_pessimal_stablematch[i]] = i

        if np.sum(meanPessimalReward) == 0:
            raise Exception("pessimal reward=0! pessimal match would be no task assignment. increase minimum task duration")

        print("task_cost_with_deadlines_userview: " + str(task_duration_with_deadlines_userview))
        print("task_cost_with_deadlines_taskview: " + str(task_duration_with_deadlines_taskview))
        print("mcsp expected utility: " + str(mcsp_expected_utility))
        print("mean optimal match reward: " + str(meanOptimalReward))
        print("mean pessimal match reward: " + str(meanPessimalReward))
        print("mean optimal match global reward: " + str(meanOptimalGlobalReward))
        print("mean pessimal match global reward: " + str(meanPessimalGlobalReward))
        print("optimal match: " + str(optimalAssignment))
        print("pessimal match: " + str(pessimalAssignment))
        print("task duration:")
        print(task_duration)
        # ----------------------------------------------------

        # caluclate metrics
        stableRegret = np.zeros([noOfUsers, T])
        taskPreference = np.zeros([1, T])
        stability = np.zeros(T)
        noOfUnstableMatches = np.zeros(T)
        estimated_task_duration = np.zeros([noOfUsers,noOfTasks])
        estimated_task_reward = np.zeros([noOfUsers,noOfTasks])
        noOfTimesVisited = np.zeros([noOfUsers,noOfTasks])
        overDeadlineCounter = np.zeros([noOfUsers, noOfTasks])
        noOfTimesChoosen = np.zeros([noOfUsers, noOfTasks])
        globalReward = np.zeros(T)
        freeSensingDone = np.zeros(T)
        for m in range(noOfMonteCarloIterations):
            for t in range(0, T):
                # calculate average pessimal regret
                for i in range(0, noOfUsers):
                    stableRegret[i][t] = stableRegret[i][t]*m/(1+m) + 1/(m+1)*(meanPessimalReward[i] - rewardMeasurements[m][i][t])

                # calculate stability: stability no user
                stable = True
                unstableMatchesCount = 0
                if checkForStability:
                    for i in range(0, noOfUsers):
                        stableUser = True
                        stableTask = True
                        # check if user would find a better task that would prefer him
                        j = int(taskMeasurements[m][i][t])
                        if not (prefer_users[i][0] == j):
                            if j == -1:
                                betterTasks = np.array(prefer_users[i])
                            else:
                                matchIndex = np.where(np.array(prefer_users[i])==j)[0][0]
                                betterTasks = prefer_users[i][0:matchIndex]
                            # would task prefer user over its current user
                            for jbetter in betterTasks:
                                matchIndexTask = np.where(np.array(prefer_tasks[jbetter]) == i)[0][0]
                                # get current user of task
                                if jbetter in list(taskMeasurements[m][:,t]):
                                    iCurrent = list(taskMeasurements[m][:,t]).index(jbetter)
                                    iCurrentIndex = np.where(np.array(prefer_tasks[jbetter]) == iCurrent)[0][0]
                                    if iCurrentIndex > matchIndexTask:
                                        stableUser = False
                                        break
                                else:
                                    stableUser = False
                                    break

                        # check if task would find a better user that would prefer him
                        if j == -1:
                            stableTask == False
                        elif stableUser == False and not(j==-1):
                            # check if corresponding task would find a better suited user that also prefers him
                            matchIndex = np.where(np.array(prefer_tasks[j]) == i)[0][0]
                            betterUsers = prefer_tasks[j][0:matchIndex]
                            # would user prefer task over its current task
                            for ibetter in betterUsers:
                                matchIndexUser = np.where(np.array(prefer_users[ibetter]) == j)[0][0]
                                # get current task of user
                                jCurrent = taskMeasurements[m][ibetter, t]
                                if jCurrent == -1:
                                    stableTask = False
                                    break
                                else:
                                    jCurrentIndex = np.where(np.array(prefer_users[ibetter]) == jCurrent)[0][0]
                                    if jCurrentIndex > matchIndexUser:
                                        stableTask = False
                                        break

                        if stableUser==False and stableTask==False:
                            stable = False
                            unstableMatchesCount += 1
                            #break
                            # reset:
                            stableUser = True
                            stableTask = True

                stability[t] = stability[t]*m/(1+m) + 1/(m+1)*stable
                noOfUnstableMatches[t] = noOfUnstableMatches[t] * m / (1 + m) + 1 / (m + 1) * unstableMatchesCount

            taskPreference = taskPreference*m/(1+m) + taskPreferenceMeasurements[m]*1/(1+m)

            globalReward = globalReward*m/(1+m) + globalRewardMeasurement[m]*1/(1+m)

            estimated_task_duration = estimated_task_duration*m/(1+m) + estimated_task_durationMeasurements[m]*1/(1+m)
            estimated_task_reward = estimated_task_reward*m/(1+m) + estimated_task_rewardMeasurements[m]*1/(1+m)
            noOfTimesVisited = noOfTimesVisited*m/(1+m) + noOfTimesVisitedMeasurements[m]*1/(1+m)
            overDeadlineCounter = overDeadlineCounter * m / (1 + m) + overDeadlineCounterMeasurements[m] * 1 / (1 + m)
            noOfTimesChoosen = noOfTimesChoosen * m / (1 + m) + noOfTimesChoosenMeasurements[m] * 1 / (1 + m)
            freeSensingDone = freeSensingDone*m/(1+m) + freeSensingDoneMeasurement[m] * 1/(1+m)



        # plot regret
        axs[0, 0].plot(np.arange(1, T+1)*deadline, stableRegret.transpose())
        pessimalOptimalGap = np.array((meanPessimalReward - meanOptimalReward))
        #for i in range(noOfUsers):
        #    axs[0, 0].axhline(y=pessimalOptimalGap[i], color='r', linestyle='--')
        axs[0, 0].set_title('average stable pessimal regret over time steps')
        axs[0, 0].set_xlabel("seconds s ")
        axs[0, 0].legend(["user " + str(i) for i in range(noOfUsers)])

        # plot cum regret
        axs[2, 0].plot(np.arange(1, T + 1)*deadline, np.cumsum(np.array(stableRegret),1).transpose())
        axs[2, 0].set_xlabel("seconds s")
        axs[2, 0].set_title('average cumulative pessimal regret over time steps')
        for i in range(noOfUsers):
            axs[2, 0].plot(np.arange(1, T + 1)*deadline, [pessimalOptimalGap[i]*t*deadline for t in range(1,(T+1))], color='r', linestyle='--')
        axs[2, 0].legend(["user " + str(i) for i in np.arange(noOfUsers)])

        # plot max regret
        axs[1, 0].plot(np.arange(1, T + 1)*deadline, np.max(stableRegret, 0))
        if iExperiment==noOfExperiments-1:
            axs[1, 0].axhline(y=np.max(pessimalOptimalGap,0), color='r', linestyle='--')
        axs[1, 0].set_xlabel("seconds")
        axs[1, 0].set_title('average maximum pessimal regret over time steps')
        # todo: achutng, noise!!! wird immer leicht nach oben vershcoben sein ??

        # plot max cum regret
        axs2[0].plot(np.arange(1, T + 1)*deadline, np.max(np.cumsum(np.array(stableRegret),1).transpose(), 1))
        if iExperiment==noOfExperiments-1:
            axs2[0].plot(np.arange(1, T + 1)*deadline, np.max([pessimalOptimalGap*t*deadline for t in range(1,(T+1))], 1), color='r', linestyle='--')
        axs2[0].set_xlabel("seconds s")
        axs2[0].set_title('average maximum cumulative pessimal regret over time steps')

        # plot estimated preferences over time
        axs[1, 1].plot([i*deadline for i in range(0, T)], taskPreference.T)
        axs[1, 1].set_xlabel("seconds")
        axs[1, 1].set_title('mean estimated user-preference over time')

        if noOfUsers <=10:
            # plot choosen arms over time
            axs[0, 2].plot([i*deadline for i in range(0, T)], choosenTasksMeasurements[0].T)
            axs[0, 2].set_xlabel("seconds")
            axs[0, 2].set_title('choosen arms over time (m=0) \n(legend: optimal ass. from users persp.)')
            axs[0, 2].legend([i for i in tasks_of_users_optimal_stablematch])


            # plot taken arms over time
            axs[0, 1].plot([i*deadline for i in range(0, T)], taskMeasurements[0].T)
            axs[0, 1].set_xlabel("seconds")
            axs[0, 1].set_title('taken arms over time (m=0) \n(legend: optimal ass. from users persp.)')
            axs[0, 1].legend([i for i in tasks_of_users_optimal_stablematch])

        # plot stability over time
        if checkForStability:
            axs2[1].plot([i*deadline for i in range(1, T)], stability[1:])
            axs2[1].set_title('P(stability(t)) over t (based on true values)')
            axs2[1].set_xlabel("seconds")

            axs2[3].plot([i * deadline for i in range(1, T)], noOfUnstableMatches[1:])
            axs2[3].set_title('no. of unstable matches over t (based on true values)')
            axs2[3].set_xlabel("seconds")

        # plot exploration var over time
        axs[1, 3].plot([i*deadline for i in range(1, T)], [((1 / t * explore_var)*((1 / t * explore_var)<=1) + ((1 / t * explore_var)>1)*1) for t in range(1, T)])
        axs[1, 3].set_xlabel("seconds")
        axs[1, 3].set_title('exploration probability over time')

        # plot freesensingDone over time
        axs[2, 3].plot([i * deadline for i in range(0, T)], np.cumsum(freeSensingDone))
        axs[2, 3].set_xlabel("seconds")
        axs[2, 3].set_title('cumulative no. of free sensing bids')


        # plot global reward over time
        axs2[2].plot([i*deadline for i in range(0, T)], globalReward)
        axs2[2].set_title('global reward over time')
        axs2[2].set_xlabel("seconds")
        globRew = getGlobalReward(noOfTasks, noOfUsers, deadline, sampling_noise, task_duration, mcsp_utility, costPerSecond)
        if iExperiment == noOfExperiments - 1:
            axs2[2].axhline(y=np.max(globRew, 0), color='r', linestyle='--')

        #print("measured rewards: " + str(rewardMeasurements))
        print("estimated durations: " + str(estimated_task_duration))
        print("estimated rewards: " + str(estimated_task_reward))
        print("noOfTimesVisited: " + str(noOfTimesVisited))
        print("noOfTimesChoosen: " + str(noOfTimesChoosen))

        if showMatrices:

            axsMatrizes[0,iExperiment].matshow(noOfTimesVisited, cmap=plt.cm.cool, aspect='auto')
            axsMatrizes[0, iExperiment].set_title("no Of Times Visited")
            for i in range(noOfUsers):
                for j in range(noOfTasks):
                    c = noOfTimesVisited[i, j]
                    axsMatrizes[0,iExperiment].text(i, j, str("{:.2e}".format(c)), va='center', ha='center')
            axsMatrizes[1, iExperiment].matshow(estimated_task_reward, cmap=plt.cm.cool, aspect='auto')
            axsMatrizes[1, iExperiment].set_title("estimated_task_reward")
            for i in range(noOfUsers):
                for j in range(noOfTasks):
                    c = estimated_task_reward[i, j]
                    axsMatrizes[1, iExperiment].text(i, j, str("{:.2e}".format(c)), va='center', ha='center')
            axsMatrizes[2, iExperiment].matshow(estimated_task_duration, cmap=plt.cm.cool,  aspect='auto')
            axsMatrizes[2, iExperiment].set_title("estimated task duration")
            for i in range(noOfUsers):
                for j in range(noOfTasks):
                    c = estimated_task_duration[i, j]
                    axsMatrizes[2, iExperiment].text(i, j, str("{:.2e}".format(c)), va='center', ha='center')
            axsMatrizes[3, iExperiment].matshow(overDeadlineCounter, cmap=plt.cm.cool, aspect='auto')
            axsMatrizes[3, iExperiment].set_title("over Deadline Counter")
            for i in range(noOfUsers):
                for j in range(noOfTasks):
                    c = overDeadlineCounter[i, j]
                    axsMatrizes[3, iExperiment].text(i, j, str("{:.2e}".format(c)), va='center', ha='center')
            axsMatrizes[4, iExperiment].matshow(noOfTimesChoosen, cmap=plt.cm.cool, aspect='auto')
            axsMatrizes[4, iExperiment].set_title("no of times choosen")
            for i in range(noOfUsers):
                for j in range(noOfTasks):
                    c = noOfTimesChoosen[i, j]
                    axsMatrizes[4, iExperiment].text(i, j, str("{:.2e}".format(c)), va='center', ha='center')


        # theorem 6
        #delta = 1
        #epsilon = (1-lambda_var)*lambda_var**(noOfUsers-1)
        #rightSide = noOfUsers**5*noOfTasks**2/(epsilon**(noOfUsers**4+1))*np.log(T)*(1/delta**2*np.log(T) + 3)
        #rewardG = np.zeros(noOfUsers)
        #for i in range(0, noOfUsers):
        #    rewardG[i] = rightSide*24*np.max([np.max(meanPessimalReward[i] - task_duration[i][:]),meanPessimalReward[i]])

        # save variables
        data[0].append( meanPessimalReward)
        data[1].append( task_duration_with_deadlines_userview)
        data[2].append( task_duration_with_deadlines_taskview)
        data[3].append( mcsp_expected_utility)
        data[4].append( meanOptimalReward)
        data[5].append( meanOptimalGlobalReward)
        data[6].append( meanPessimalGlobalReward)
        data[7].append( optimalAssignment)
        data[8].append( pessimalAssignment)
        data[9].append( task_duration)
        data[10].append( stability)
        data[11].append( noOfUnstableMatches)
        data[12].append( taskPreference)
        data[13].append( globalReward)
        data[14].append( estimated_task_duration)
        data[15].append( estimated_task_reward)
        data[16].append( noOfTimesVisited)
        data[17].append( overDeadlineCounter)
        data[18].append( noOfTimesChoosen)
        data[19].append( freeSensingDone)
        data[20].append( T)
        data[21].append( deadline)
        data[22].append( stableRegret)
        data[23].append( noOfUsers)
        data[24].append( choosenTasksMeasurements)
        data[25].append( taskMeasurements)
        data[26].append( tasks_of_users_optimal_stablematch)
        data[27].append( noOfTasks)
        data[28].append( explore_var)
        data[29].append( noOfExperiments)
        data[30].append( globRew)
        data[31].append(prefer_users)
        data[32].append(prefer_tasks)
        #siminfo:
        data[33].append(noOfTasks)
        data[34].append(noOfUsers)
        data[35].append(deadline)
        data[36].append(revenuePerMbit)
        data[37].append(costPerSecond)
        data[38].append(T)
        data[39].append(lambda_var)
        data[40].append(forgettingDuration_var)
        data[41].append(marge)
        data[42].append(maxAllowedMarge)
        data[43].append(explore_var)
        data[44].append(sampling_noise)
        data[45].append(activateBurst_e[iExperiment])
        data[46].append(takeNothingIfNegativeReward_e[iExperiment])
        data[47].append(enableNoAccessCount_e[iExperiment])
        data[48].append(rewardSensing_i[iExperiment])
        data[49].append(noAccessMode_e[iExperiment])
        data[50].append(countDegradeRate)
        data[51].append(countStartTime)
        data[52].append(countEndTime_e[iExperiment])
        data[53].append(countInfluence_var)
        data[54].append(countUntilAdjustment_e[iExperiment])
        data[55].append(considerPreferenceMCSP_var_e[iExperiment])
        data[56].append(epsilon_greedy_e[iExperiment])
        data[57].append(noOfMonteCarloIterations)

    axs[1, 0].legend(["exp " + str(i) for i in range(noOfExperiments)])
    axs[2, 1].legend(["exp " + str(i) for i in range(noOfExperiments)])
    axs[2, 0].legend(["exp " + str(i) for i in range(noOfExperiments)])
    axs2[2].legend(["exp " + str(i) for i in range(noOfExperiments)])
    axs2[1].legend(["exp " + str(i) for i in range(noOfExperiments)])
    axs2[0].legend(["exp " + str(i) for i in range(noOfExperiments)])
    plt.tight_layout()
    plt.show()

    with open("autoresults/" + resultsFileName + ".pkl", 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(data, f)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/