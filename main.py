# This is a sample Python script.
import copy
import os

import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy
from joblib import Parallel, delayed
#import ray
from tqdm import tqdm
from solverResult import getGlobalReward


def calculateExpectationValues(costPerSecond, marge, deadline, sampling_noise, task_duration, mcsp_utility, noOfSamples):
    noOfUsers = np.size(task_duration, 0)
    noOfTasks = np.size(task_duration, 1)

    user_reward_sum = np.zeros([noOfUsers, noOfTasks])
    mcsp_reward_sum = np.zeros([noOfUsers, noOfTasks])

    for i in tqdm(range(noOfUsers)):
        for j in range(noOfTasks):
            for iSample in range(noOfSamples):
                sample = np.max([0,np.random.normal(loc=task_duration[i][j], scale=sampling_noise)])
                if sample>deadline:
                    user_reward_sum[i][j] += -sample * costPerSecond[i]
                    # if mcsp_Learning = off:
                    #   mcsp_reward_sum = mcsp_utility[i][j] - np.min([mcsp_utility[i][j], sample*costPerSecond[i]])
                elif sample*(1+marge)*costPerSecond[i] >= mcsp_utility[i][j]:
                    user_reward_sum[i][j] +=  mcsp_utility[i][j] - sample*costPerSecond[i]
                elif sample < 0:
                    mcsp_reward_sum[i][j] += mcsp_utility[i][j]
                else:
                    user_reward_sum[i][j] += sample*marge*costPerSecond[i]
                    mcsp_reward_sum[i][j] += mcsp_utility[i][j] - sample*(1+marge)*costPerSecond[i]

    user_reward_expectation = np.divide(user_reward_sum, noOfSamples)
    mcsp_reward_expectation = np.divide(mcsp_reward_sum, noOfSamples)

    return user_reward_expectation, mcsp_reward_expectation




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


def get_state(task_availabilities_t, lastPlausibleSet,  best_tasks_param, context, t):
    # kontext 1: just return the integer number
    if context == 0:
        return 0
    elif context == 1:
        return int("".join(str(x) for x in task_availabilities_t), 2)
    elif context == 2:
        return int("".join(str(x) for x in lastPlausibleSet), 2)
    elif context == 3:
        #noOfBestTasks = best_tasks_param[0]
        noOfBestTasks = int(np.floor(t/900))
        #noOfBestTasks = 2
        ucbBound = -1*best_tasks_param[1]
        sortedUCBBound = np.argsort(ucbBound)
        task_availabilities_tt = np.copy(task_availabilities_t)
        task_availabilities_tt[sortedUCBBound[noOfBestTasks:]] = 1
        state = int("".join(str(x) for x in task_availabilities_tt), 2)
        return state
    elif context == 4:
        #noOfBestTasks = best_tasks_param[0]
        #noOfBestTasks = int(np.floor(t/3000))
        noOfBestTasks = 1
        ucbBound = -1*best_tasks_param[1]
        sortedUCBBound = np.argsort(ucbBound)
        task_availabilities_tt = np.copy(task_availabilities_t)
        task_availabilities_tt[sortedUCBBound[noOfBestTasks:]] = 1
        state = int("".join(str(x) for x in task_availabilities_tt), 2)
        return state
    elif context == 5:
        #noOfBestTasks = best_tasks_param[0]
        #noOfBestTasks = int(np.floor(t/3000))
        noOfBestTasks = 2
        ucbBound = -1*best_tasks_param[1]
        sortedUCBBound = np.argsort(ucbBound)
        task_availabilities_tt = np.copy(task_availabilities_t)
        task_availabilities_tt[sortedUCBBound[noOfBestTasks:]] = 1
        state = int("".join(str(x) for x in task_availabilities_tt), 2)
        return state


def ucb_ca(noOfUsers, noOfTasks, T, task_duration, mcsp_utility, marge, lambda_var, deadline, explore_var, sampling_noise, enableNoAccessCount, countDegradeRate, countStartTime, countInfluence_var, countUntilAdjustment, noAccessMode, countEndTime, considerPreferenceMCSP_var, maxAllowedMarge, activateBurst, forgettingDuration_var, epsilon_greedy, takeNothingIfNegativeReward, rewardSensing, costPerSecond, newGreedy, isAvailable, task_availabilities, context, doPlausibleSetPrediction):
    # init
    noOfTimesChoosen = np.zeros([noOfUsers, noOfTasks]).astype(int)
    noOfTimesVisited = np.zeros([noOfUsers, noOfTasks]).astype(int)
    choosenTasks = -1 * np.ones(noOfUsers).astype(int)
    performedTasks = -1 * np.ones(noOfTasks).astype(int)
    lastBidOnTask = -1 * np.ones(noOfTasks)
    usersCurrentBidOnTask = -1 * np.ones(noOfUsers)
    rewardMeasurements = np.zeros([noOfUsers, T])
    rewardMeasurements_MCSP = np.zeros([noOfTasks, T])
    globalRewardMeasurement = np.zeros(T)
    taskMeasurements = -1 * np.ones([noOfUsers, T])
    taskPreferenceMeasurements = np.zeros(T)
    choosenTasksMeasurements = np.zeros([noOfUsers, T])
    estimated_task_reward = np.zeros(([noOfUsers, noOfTasks, 2**noOfTasks]))
    estimated_task_reward_without_state = np.zeros(([noOfUsers, noOfTasks]))
    estimated_task_duration = np.zeros(([noOfUsers,noOfTasks]))
    noAccessCounter = np.zeros([noOfUsers, noOfTasks])
    overDeadlineCounter = np.zeros([noOfUsers, noOfTasks])
    lastBidAccepted = np.zeros([noOfUsers, noOfTasks])
    lastWasAccepted = np.zeros([noOfUsers,noOfTasks])
    freeSensingTried = np.zeros([noOfUsers, noOfTasks])
    freeSensingDone = np.zeros(T)
    sameTasksInARow = np.zeros(noOfUsers)
    lastPlausibleSets = np.zeros([noOfUsers, noOfTasks])
    lastTask = -1*np.ones(noOfUsers)
    predictedPlausibleSet = (np.ones(([noOfUsers, noOfTasks, 2**noOfTasks])), np.zeros(([noOfUsers, noOfTasks, 2**noOfTasks])), np.zeros(([noOfUsers, noOfTasks, 2**noOfTasks])))



    # algorithm
    for t in range(1, T):
        for i in range(0, noOfUsers):
            if t == 1:
                j = np.random.randint(0, noOfTasks)
                choosenTasks[i] = j
                usersCurrentBidOnTask[i] = estimated_task_duration[i][choosenTasks[i]] * (marge + 1) - mcsp_utility[i][j]
            else:
                if considerPreferenceMCSP_var == True:
                    D = (np.random.uniform(0, 1) <= lambda_var) * 1
                else:
                    D = 0
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
                            userShouldSenseFree =  (np.random.uniform(0, 1) <= (noAccessCounter[i][j])/countUntilAdjustment)#(countEndTime>t)*(noAccessCounter[i][j] > countUntilAdjustment)#*(countEndTime>t) # # (np.random.uniform(0, 1) <= (noAccessCounter[i][j])/countUntilAdjustment) # stochastic
                            if userShouldSenseFree:
                                task_duration_user = estimated_task_duration[i][j]*0.7
                                freeSensingTried[i, j] = 1
                            else:
                                task_duration_user = estimated_task_duration[i][j]
                        testBid = task_duration_user * (marge + 1) * costPerSecond[i] #+ sameTasksInARow[i]/100

                        # ------ test if task is plausible
                        taskIsPlausible = True
                        considerPreferenceMCSP = np.random.uniform(0, 1) <= considerPreferenceMCSP_var
                        if considerPreferenceMCSP:
                            taskIsPlausible = taskIsPlausible and ((testBid - mcsp_utility[i][j]) <= lastBidOnTask[j] or performedTasks[j] == i)
                        if doPlausibleSetPrediction:
                            if t>1400:#200<predictedPlausibleSet[1][i, j, get_state(task_availabilities[:, t - 1], [], [], 1, [])]:#(np.random.uniform(0, 1) > 1/(t*explore_var+1)) * 1:#t>500:# 0.9 < predictedPlausibleSet[2][i, j, get_state(task_availabilities[:, t - 1], [], [], 1, [])]: #t > 0:
                                predictionAccuracy = 1#1/t*predictedPlausibleSet[1][i,j,get_state(task_availabilities[:,t-1], [], [], 1, [])]
                            else:
                                predictionAccuracy = 0
                            predictedPlausibility = predictedPlausibleSet[0][i, j, get_state(task_availabilities[:, t - 1], [], [], 1, [])]
                            taskIsPlausible = (taskIsPlausible * (1 - predictionAccuracy) + (predictedPlausibility > 0.5) * predictionAccuracy) > 0.5
                        if isAvailable:
                            taskIsPlausible = taskIsPlausible and task_availabilities[j, t-1]#(np.random.uniform(0, 1) <= 0.3)
                        if taskIsPlausible:
                            plausibleTasks[j] = 1
                            testedBids[j] = testBid - mcsp_utility[i][j]
                            #noAccessCounter[i][j] = 0#np.max([0, noAccessCounter[i][j] - countDegradeRate])
                            #lastBidAccepted[i][j] = np.max([lastBidAccepted[i][j], testöBid])
                            lastWasAccepted[i][j] = True
                        else:
                            noAccessCounter[i][j] += np.min([1,(t/countInfluence_var)*(t<countEndTime)]) # determmnistic: (countEndTime>t)*t/(countInfluence_var)#1*(countEndTime-t)/t
                            lastWasAccepted[i][j] = False #np.random.uniform(0, 1) <=
                    lastPlausibleSets[i] = np.copy(plausibleTasks)
                    # ---- decide for task and pull
                    ucbBound = np.copy(estimated_task_reward[i,:,get_state(task_availabilities[:, t - 1], lastPlausibleSets[i], [3, estimated_task_reward_without_state[i]], context, t)])

                    ucbBound[plausibleTasks == 0] = np.nan
                    # epsilon greedy
                    if epsilon_greedy == True:
                        explore = (np.random.uniform(0, 1) <= 1/(t*explore_var+1)) * 1
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
                        usersCurrentBidOnTask[i] = testedBids[choosenTasks[i]] #np.random.normal(loc=estimated_task_duration[i][choosenTasks[i]], scale=0)* (marge + 1)#
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
                taskAcceptedUser = (np.min(usersCurrentBidOnTask[choosenTasks == choosenTasks[i]]) >= usersCurrentBidOnTask[i])
                if taskAcceptedUser:
                    performedTasks[choosenTasks[i]] = i
                    lastBidOnTask[choosenTasks[i]] = usersCurrentBidOnTask[i]
                    tasksHaveUser[choosenTasks[i]] = True
                elif newGreedy:# or (doPlausibleSetPrediction and t<1000):
                    estimated_task_reward[i,choosenTasks[i],get_state(task_availabilities[:,t-1], lastPlausibleSets[i], [3, estimated_task_reward_without_state[i]], context, t)] = 0 * forgettingDuration_var_i + estimated_task_reward[i,
                        choosenTasks[i],get_state(task_availabilities[:,t-1], lastPlausibleSets[i], [3, estimated_task_reward_without_state[i]], context, t)] * (1 - forgettingDuration_var_i)

                if doPlausibleSetPrediction:
                    # update trustability
                    predictedPlausibility = 0.5 < predictedPlausibleSet[2][i, choosenTasks[i], get_state(task_availabilities[:, t - 1], [], [], 1, [])]
                    if taskAcceptedUser == predictedPlausibility:
                        predictedPlausibleSet[2][i, choosenTasks[i], get_state(task_availabilities[:, t - 1], [], [], 1, [])] = 0.5 + predictedPlausibleSet[2][i, choosenTasks[i], get_state(task_availabilities[:, t - 1], [], [], 1, [])]*0.5
                    else:
                        predictedPlausibleSet[2][i, choosenTasks[i], get_state(task_availabilities[:, t - 1], [], [], 1, [])] *= 0.5
                    noOfTimesUpdated = predictedPlausibleSet[1][i,choosenTasks[i],get_state(task_availabilities[:,t-1], [], [], 1, [])]
                    # update mean value
                    predictedPlausibleSet[0][i,choosenTasks[i],get_state(task_availabilities[:,t-1], [], [], 1, [])] = noOfTimesUpdated/(1+noOfTimesUpdated)*predictedPlausibleSet[0][i,choosenTasks[i],get_state(task_availabilities[:,t-1], [], [], 1, [])] + 1/(1+noOfTimesUpdated)*taskAcceptedUser
                    # update no of visits
                    predictedPlausibleSet[1][i, choosenTasks[i], get_state(task_availabilities[:,t-1], [], [], 1, [])] += 1


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
                if lastTask[i] == j:
                    sameTasksInARow[i] +=1
                else:
                    lastTask[i] = j
                    sameTasksInARow[i] = 0
                noOfTimesVisited[i][j] += 1
                forgettingDuration_var_i = forgettingDuration_var[1]**noOfTimesVisited[i][j]*forgettingDuration_var[0]
                burst = 1.2*(i==4 and t<30)*(activateBurst == True)
                sampleTaskDuration = np.max([0,np.random.normal(loc=task_duration[i][j] + burst, scale=sampling_noise)])

                if t == 1:
                    estimated_task_duration[i][j] = sampleTaskDuration
                else:
                    forgettingDuration_var_i = 0.5
                    estimated_task_duration[i][j] = sampleTaskDuration*forgettingDuration_var_i + estimated_task_duration[i][j]*(1-forgettingDuration_var_i)
                if rewardSensing == 0:
                    # original
                    reward = np.min([mcsp_utility[i][j], usersCurrentBidOnTask[i] + mcsp_utility[i][j], costPerSecond[i] * (1 + maxAllowedMarge) * sampleTaskDuration]) * (sampleTaskDuration <= deadline) - costPerSecond[i] * sampleTaskDuration
                    estimated_task_reward[i][j] = reward*forgettingDuration_var_i + estimated_task_reward[i][j]*(1-forgettingDuration_var_i)
                elif rewardSensing == 1:
                    # ignore reward
                    if not(freeSensingTried[i, j]):
                        reward = np.min([mcsp_utility[i][j], usersCurrentBidOnTask[i] + mcsp_utility[i][j], costPerSecond[i] * (1 + maxAllowedMarge) * sampleTaskDuration]) * (sampleTaskDuration <= deadline) - costPerSecond[i] * sampleTaskDuration
                        estimated_task_reward[i][j] = reward * forgettingDuration_var_i + estimated_task_reward[i][j] * (1 - forgettingDuration_var_i)
                elif rewardSensing == 2:
                    # pseudo reward with deadlines
                    if True:#not(freeSensingTried[i, j]):#
                        reward = np.min([mcsp_utility[i][j], usersCurrentBidOnTask[i] + mcsp_utility[i][j]]) * (sampleTaskDuration <= deadline) - costPerSecond[i] * sampleTaskDuration
                        estimated_task_reward[i, j, get_state(task_availabilities[:,t-1], lastPlausibleSets[i], [3, estimated_task_reward_without_state[i]], context, t)] = reward * forgettingDuration_var_i + estimated_task_reward[i,j,get_state(task_availabilities[:,t-1], lastPlausibleSets[i], [3, estimated_task_reward_without_state[i]], context, t)] * (1 - forgettingDuration_var_i) #, costPerSecond[i] * (1 + maxAllowedMarge) * sampleTaskDuration
                        estimated_task_reward_without_state[i,j] = reward * forgettingDuration_var_i + estimated_task_reward_without_state[i,j] * (1 - forgettingDuration_var_i)
                    else:
                        reward = np.min([mcsp_utility[i][j], sampleTaskDuration*costPerSecond[i]*(1+marge)]) * (
                                    sampleTaskDuration <= deadline) - costPerSecond[i] * sampleTaskDuration
                        estimated_task_reward[i,j,get_state(task_availabilities[:,t-1], lastPlausibleSets[i], [3, estimated_task_reward_without_state], context, t)] = reward * forgettingDuration_var_i + estimated_task_reward[i,j,get_state(task_availabilities[:,t-1], lastPlausibleSets[i], [3, estimated_task_reward_without_state], context, t)] * (1 - forgettingDuration_var_i)
                elif rewardSensing == 3:
                    # pseudo reward
                    if not(freeSensingTried[i, j]):
                        reward = np.min([mcsp_utility[i][j], usersCurrentBidOnTask[i] + mcsp_utility[i][j], costPerSecond[i] * (1 + maxAllowedMarge) * sampleTaskDuration]) * (sampleTaskDuration <= deadline) - costPerSecond[i] * sampleTaskDuration
                        estimated_task_reward[i][j] = reward * forgettingDuration_var_i + estimated_task_reward[i][j] * (1 - forgettingDuration_var_i)
                    else:
                        reward = (1 + marge) * sampleTaskDuration # pseudo reward
                        reward *= costPerSecond[i]
                        estimated_task_reward[i][j] = reward * forgettingDuration_var_i + estimated_task_reward[i][j] * (1 - forgettingDuration_var_i) # , costPerSecond[i]*(1+maxAllowedMarge)*sampleTaskDuration
                rewardMeasurements[i][t] = np.min([mcsp_utility[i][j], usersCurrentBidOnTask[i] + mcsp_utility[i][j]])*(sampleTaskDuration <= deadline) - costPerSecond[i]*sampleTaskDuration # reward # todo: reward hier rein oder wo anders?
                globalRewardMeasurement[t] +=  mcsp_utility[i][j]*(sampleTaskDuration <= deadline) - sampleTaskDuration*costPerSecond[i]
                taskMeasurements[i][t] = j
                noAccessCounter[i][j] = 0
                overDeadlineCounter[i][j] = overDeadlineCounter[i][j] + (sampleTaskDuration > deadline)*1
                rewardMeasurements_MCSP[j][t] = (mcsp_utility[i][j]- np.min([mcsp_utility[i][j], usersCurrentBidOnTask[i] + mcsp_utility[i][j]])) * (
                                                       sampleTaskDuration <= deadline)  # reward # todo: reward hier rein oder wo anders?

            #if t == 900 and context==3:
            #   # do transfer learning
            #   for iState in range(2 ** noOfTasks):
            #       estimated_task_reward[i, :, iState] = estimated_task_reward_without_state[i, :]

        taskPreferenceMeasurements[t] = test_on_preference(task_duration, estimated_task_duration, noOfTimesVisited, t)

    return rewardMeasurements, taskPreferenceMeasurements, estimated_task_duration, choosenTasksMeasurements, taskMeasurements, estimated_task_reward, noOfTimesVisited, overDeadlineCounter, noOfTimesChoosen, globalRewardMeasurement, freeSensingDone, rewardMeasurements_MCSP


#@ray.remote
def doMonteCarrloIterations(noOfMonteCarloIterations, noOfUsers, noOfTasks, T, task_duration, mcsp_utility, marge, lambda_var, deadline, explore_var, sampling_noise, enableNoAccessCount, countDegradeRate, countStartTime, countInfluence_var, countUntilAdjustment, noAccessMode, countEndTime, considerPreferenceMCSP_var, maxAllowedMarge, activateBurst, forgettingDuration_var, epsilon_greedy, takeNothingIfNegativeReward, rewardSensing, costPerSecond, newGreedy, isAvailable, task_availabilities, context, doPlausibleSetPrediction):
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
    rewardMeasurements_MCSP = {}
    for m in tqdm(range(noOfMonteCarloIterations)):
        rewardMeasurements_i, taskPreferenceMeasurements_i, estimated_task_duration_i, choosenTasksMeasurements_i, taskMeasurements_i, estimated_task_reward_i, noOfTimesVisited_i, overDeadlineCounter_i, noOfTimesChoosen_i, globalRewardMeasurement_i, freeSensingDoneMeasurement_i, rewardMeasurements_MCSP_i = ucb_ca(noOfUsers, noOfTasks, T, task_duration, mcsp_utility, marge, lambda_var, deadline, explore_var, sampling_noise, enableNoAccessCount, countDegradeRate, countStartTime, countInfluence_var, countUntilAdjustment, noAccessMode, countEndTime, considerPreferenceMCSP_var, maxAllowedMarge, activateBurst, forgettingDuration_var, epsilon_greedy, takeNothingIfNegativeReward, rewardSensing, costPerSecond, newGreedy, isAvailable, task_availabilities, context, doPlausibleSetPrediction)
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
        rewardMeasurements_MCSP[m] = rewardMeasurements_MCSP_i
    return rewardMeasurements, taskPreferenceMeasurements, estimated_task_duration, choosenTasksMeasurements, taskMeasurements, estimated_task_reward, noOfTimesVisited, overDeadlineCounter, noOfTimesChoosen, globalRewardMeasurement, freeSensingDoneMeasurement, rewardMeasurements_MCSP





def print_hi(name):
    use_ray = False
    customName = "random"
    resultsFileName = "zzz"

    noOfTasks = 3
    noOfUsers = 3

    deadline = 7

    revenuePerMbit = 0.09  # revenue fopr mcsp: €/Mbit for mcsp
    costPerSecond = noOfUsers * [0.02]
    T = 3000
    lambda_var = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    forgettingDuration_var = (1, 0.99) # start, decay with time
    marge = 0.1
    maxAllowedMarge = 100.1
    explore_var = [0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.1]
    sampling_noise = 0.01  # 0.01
    activateBurst_e = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    takeNothingIfNegativeReward_e = [False] * 20  # [True, True, True, True, True, True, True, True, True, True, True, True, True, True,
    # True]  # todo: true after a delay, to prevent task=-1 assignment if bids were to low at beginning
    enableNoAccessCount_e = [False, False, False, False, False, False, False]
    rewardSensing_i = [2, 2, 2, 2, 2, 2, 2] # 0: original, 1: ignore, 2: pseudo reward
    noAccessMode_e = [0, 0, 0, 0, 0, 0, 0]
    countDegradeRate = 8
    countStartTime = 0
    countEndTime_e = [500] * 15  # used
    countInfluence_var_e = [20000, 20000, 20000, 20000, 20000, 20000, 20000]
    countUntilAdjustment_e = [500, 500, 500, 500, 500, 500, 500]
    considerPreferenceMCSP_var_e = [1, 1, 1, 1, 1, 1, 0] # prob. for considering mcsp prefernce list for plausible list (1= consider it always)

    epsilon_greedy_e = [True, True, True, True, True, True, True]
    newGreedy_e = [False, False, False, False, False, False, True]

    isAvailable_e = [True, True, True, True, True, True, True]

    context_e = [0, 1, 0, 0, 1, 0, 0]
    doPlausibleSetPrediction_e = [1,0,0,0,0,0,0]

    noOfMonteCarloIterations = 30
    showMatrices = False
    checkForStability = False

    expList = list([0,2,6]) #range(len(epsilon_greedy_e)) #

    print("iterations: " + str(T) + " seconds: " + str(deadline*T))

    pickelFileName = "data/" + str(noOfTasks) + str(noOfUsers) + str(customName)
    with open(pickelFileName + ".pkl", 'rb') as f:
        data = pickle.load(f)
    t_processing = data[0]
    t_upload = data[1]
    mcsp_utility = data[2]*revenuePerMbit

    task_duration = t_processing+t_upload # todo: aufsplitten

    task_availabilitiesFileName = "data/" + str(noOfTasks) + "_auto_" + str(customName) + "availabilities" + ".pkl"
    if os.path.isfile(task_availabilitiesFileName):
        print("loading task_availabilities")
        with open(task_availabilitiesFileName, 'rb') as f:
            task_availabilities = pickle.load(f)

    # check for plausible set predicitonm
    if sum((doPlausibleSetPrediction_e[i] and np.array(context_e[i])>0) for i in expList):
        raise ValueError("wrong context setting for plausible set prediction")

    # check for order
    for i in range(0, noOfTasks):
        _, counts = np.unique(task_duration[:, i]+mcsp_utility[:,i], return_counts=True)
        if not (counts == 1).all():
            raise ValueError("no strict ordering of users" + str(task_duration))

    prefer_tasks = []
    prefer_users = []

    # get expected values
    # attention: if cost, price, deadline or margin are changed, sampling_noise the pkl has to be deleted.
    expectedValspickelFileName = pickelFileName+"expectedVals" + ".pkl"
    if os.path.isfile(expectedValspickelFileName):
        print("expectedVals exists")
        with open(expectedValspickelFileName, 'rb') as f:
            dataExpected = pickle.load(f)
        user_reward_expectation = dataExpected[0]
        mcsp_reward_expectation = dataExpected[1]
    else:
        print("recalculate expectedVals")
        user_reward_expectation, mcsp_reward_expectation = calculateExpectationValues(costPerSecond, marge, deadline, sampling_noise, task_duration, mcsp_utility, 100000)
        dataExpected = [user_reward_expectation, mcsp_reward_expectation]
        with open(expectedValspickelFileName, 'wb') as f:
            pickle.dump(dataExpected, f)



    # tasks preferences             # todo: mcsp_expected_utility darf kein p_max_allowed verwenden, dieses ist nur für nutzer reserviert. muss eigenes machen nur mit deadline.
    for i in range(noOfTasks):
        prefer_tasks.append((np.flip(np.argsort(mcsp_reward_expectation[:, i]))).tolist())
    # users pferences
    for i in range(noOfUsers):
        prefer_users.append((np.flip(np.argsort(user_reward_expectation[i][:]))).tolist())

    # advantage: tasks
    users_of_tasks_pessimal_stablematch = stableMatching(noOfTasks, noOfUsers, prefer_tasks, prefer_users)
    # advantage: users
    tasks_of_users_optimal_stablematch = stableMatching(noOfUsers, noOfTasks, prefer_users, prefer_tasks)

    # calculate stable optimal reward of users
    meanOptimalReward = np.zeros(noOfUsers)
    meanOptimalGlobalReward = 0
    optimalAssignment = -1 * np.ones(noOfUsers)
    for i in range(noOfUsers):
        if not (tasks_of_users_optimal_stablematch[i] == None):
            meanOptimalReward[i] = user_reward_expectation[i][tasks_of_users_optimal_stablematch[i]]
            meanOptimalGlobalReward += mcsp_reward_expectation[i][tasks_of_users_optimal_stablematch[i]] + \
                                       user_reward_expectation[i][tasks_of_users_optimal_stablematch[i]]
            optimalAssignment[i] = tasks_of_users_optimal_stablematch[i]

    # calculate stable pessimal reward of users
    meanPessimalReward = np.zeros(noOfUsers)
    meanOptimalReward_MCSP = np.zeros(noOfTasks)
    meanPessimalGlobalReward = 0
    pessimalAssignment = -1 * np.ones(noOfUsers)
    for i in range(noOfTasks):
        if not (users_of_tasks_pessimal_stablematch[i] == None):
            meanPessimalReward[users_of_tasks_pessimal_stablematch[i]] = \
            user_reward_expectation[users_of_tasks_pessimal_stablematch[i]][i]
            meanPessimalGlobalReward += mcsp_reward_expectation[users_of_tasks_pessimal_stablematch[i]][i] + \
                                        user_reward_expectation[users_of_tasks_pessimal_stablematch[i]][i]
            pessimalAssignment[users_of_tasks_pessimal_stablematch[i]] = i
            meanOptimalReward_MCSP[i] += mcsp_reward_expectation[users_of_tasks_pessimal_stablematch[i]][i]

    if np.sum(meanPessimalReward) == 0:
        raise Exception("pessimal reward=0! pessimal match would be no task assignment. increase minimum task duration")
    print("task_duration: " + str(task_duration))
    print("user_reward_expectation: " + str(user_reward_expectation))
    print("mcsp_reward_expectation: " + str(mcsp_reward_expectation))
    print("mean optimal match reward: " + str(meanOptimalReward))
    print("mean pessimal match reward: " + str(meanPessimalReward))
    print("mean optimal (MCSP) match reward: " + str(meanOptimalReward_MCSP))
    print("mean optimal match global reward: " + str(meanOptimalGlobalReward))
    print("mean pessimal match global reward: " + str(meanPessimalGlobalReward))
    print("optimal match: " + str(optimalAssignment))
    print("pessimal match: " + str(pessimalAssignment))

    # ----------------------------------------------------


    # Compute results: ----------------------------------------------------
    if use_ray:
        ray.init(local_mode=False)
        results = ray.get([doMonteCarrloIterations.remote(noOfMonteCarloIterations, noOfUsers, noOfTasks, T, task_duration,mcsp_utility, marge, lambda_var,
                                deadline, explore_var, sampling_noise, enableNoAccessCount_e[iExperiment],
                                countDegradeRate, countStartTime, countInfluence_var_e[iExperiment], countUntilAdjustment_e[iExperiment],
                                noAccessMode_e[iExperiment], countEndTime_e[iExperiment], considerPreferenceMCSP_var_e[iExperiment], maxAllowedMarge, activateBurst_e[iExperiment], forgettingDuration_var, epsilon_greedy_e[iExperiment], takeNothingIfNegativeReward_e[iExperiment], rewardSensing_e[iExperiment], costPerSecond, isAvailable_e[iExperiment]) for iExperiment in expList])
        ray.shutdown()

    else:
        results = Parallel(n_jobs=len(expList))(delayed(doMonteCarrloIterations)(noOfMonteCarloIterations, noOfUsers, noOfTasks, T, task_duration,mcsp_utility, marge, lambda_var[iExperiment],
                            deadline, explore_var[iExperiment], sampling_noise, enableNoAccessCount_e[iExperiment],
                            countDegradeRate, countStartTime, countInfluence_var_e[iExperiment], countUntilAdjustment_e[iExperiment],
                           noAccessMode_e[iExperiment], countEndTime_e[iExperiment], considerPreferenceMCSP_var_e[iExperiment], maxAllowedMarge, activateBurst_e[iExperiment], forgettingDuration_var, epsilon_greedy_e[iExperiment], takeNothingIfNegativeReward_e[iExperiment], rewardSensing_i[iExperiment], costPerSecond, newGreedy_e[iExperiment], isAvailable_e[iExperiment], task_availabilities, context_e[iExperiment], doPlausibleSetPrediction_e[iExperiment]) for iExperiment in expList)


    # Plot results: ----------------------------------------------------

    fig, axs = plt.subplots(3, 4)
    fig2, axs2 = plt.subplots(1, 4)
    fig3, axs3 = plt.subplots(2, max(2,len(expList)))
    fig.canvas.manager.set_window_title('MAIN 1')
    fig2.canvas.manager.set_window_title('MAIN 1')

    data = []
    for idd in range(60):
        data.append([])
    for iExperiment in expList:

        print("//////////// experiment " + str(iExperiment))

        # get from ray results
        rewardMeasurements, taskPreferenceMeasurements, estimated_task_durationMeasurements, choosenTasksMeasurements, taskMeasurements, estimated_task_rewardMeasurements, noOfTimesVisitedMeasurements, overDeadlineCounterMeasurements, noOfTimesChoosenMeasurements, globalRewardMeasurement, freeSensingDoneMeasurement, rewardMeasurements_MCSPMeasurements = results[expList.index(iExperiment)]

        # caluclate metrics
        stableRegret = np.zeros([noOfUsers, T])
        taskPreference = np.zeros([1, T])
        stability = np.zeros(T)
        noOfUnstableMatches = np.zeros(T)
        estimated_task_duration = np.zeros([noOfUsers,noOfTasks])
        estimated_task_reward = np.zeros([noOfUsers,noOfTasks,2**noOfTasks])
        noOfTimesVisited = np.zeros([noOfUsers,noOfTasks])
        overDeadlineCounter = np.zeros([noOfUsers, noOfTasks])
        noOfTimesChoosen = np.zeros([noOfUsers, noOfTasks])
        globalReward = np.zeros(T)
        freeSensingDone = np.zeros(T)
        rewMes = np.zeros([noOfUsers, T])
        rewardMeasurements_MCSP = np.zeros([noOfTasks,T])
        for m in range(noOfMonteCarloIterations):
            for t in range(0, T):
                # calculate average pessimal regret
                for i in range(noOfUsers):
                    stableRegret[i][t] = stableRegret[i][t]*m/(1+m) + 1/(m+1)*(meanPessimalReward[i] - rewardMeasurements[m][i,t])
                # calculate stability: stability no user
                stable = True
                unstableMatchesCount = 0
                if checkForStability:
                    for i in range(0, noOfUsers):
                        stableUser = True
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

                        if stableUser==False:
                            stable = False
                            unstableMatchesCount += 1

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
            rewMes = rewMes*m/(1+m) + rewardMeasurements[m] * 1/(1+m)
            rewardMeasurements_MCSP = rewardMeasurements_MCSP*m/(1+m) + rewardMeasurements_MCSPMeasurements[m]*1/(1+m)


        # plot regret
        axs[0, 0].plot(np.arange(1, T+1)*deadline, stableRegret.transpose())
        pessimalOptimalGap = np.array(meanPessimalReward - meanOptimalReward)
        #for i in range(noOfUsers):
        #    axs[0, 0].axhline(y=pessimalOptimalGap[i], color='r', linestyle='--')
        axs[0, 0].set_title('average stable pessimal regret over time steps')
        axs[0, 0].set_xlabel("seconds s ")
        axs[0, 0].legend(["user " + str(i) for i in range(noOfUsers)])

        # plot reward
        axs[2, 1].plot(np.arange(1, T + 1) * deadline, rewMes.transpose())
        for i in range(noOfUsers):
            axs[2, 1].plot(np.arange(1, T) ,
                           [meanPessimalReward[i] for t in range(1, T)], color='r', linestyle='--')

        # plot mcsp reward
        axs[2, 2].plot(np.arange(1, T + 1) * deadline,
                       np.cumsum(np.sum(np.array(rewardMeasurements_MCSP.transpose()), 1).transpose()))
        axs[2, 2].set_title('average stable mcsp regret over time steps')
        axs[2, 2].set_xlabel("seconds s ")

        # plot cum regret
        axs[2, 0].plot(np.arange(1, T + 1)*deadline, np.cumsum(np.array(stableRegret),1).transpose())
        axs[2, 0].set_xlabel("seconds s")
        axs[2, 0].set_title('average cumulative pessimal regret over time steps')
        for i in range(noOfUsers):
            axs[2, 0].plot(np.arange(1, T + 1)*deadline, [pessimalOptimalGap[i]*t*deadline for t in range(1,(T+1))], color='r', linestyle='--')
        axs[2, 0].legend(["user " + str(i) for i in np.arange(noOfUsers)])

        # plot max regret
        axs[1, 0].plot(np.arange(1, T + 1)*deadline, np.max(stableRegret, 0))
        if iExperiment==len(expList)-1:
            axs[1, 0].axhline(y=np.max(pessimalOptimalGap,0), color='r', linestyle='--')
        axs[1, 0].set_xlabel("seconds")
        axs[1, 0].set_title('average maximum pessimal regret over time steps')
        # todo: achutng, noise!!! wird immer leicht nach oben vershcoben sein ??

        # plot max cum regret
        axs2[0].plot(np.arange(1, T + 1)*deadline, np.max(np.cumsum(np.array(stableRegret),1).transpose(), 1))
        if iExperiment==len(expList)-1:
            axs2[0].plot(np.arange(1, T + 1)*deadline, np.max([pessimalOptimalGap*t*deadline for t in range(1,(T+1))], 1), color='r', linestyle='--')
        axs2[0].set_xlabel("seconds s")
        axs2[0].set_title('average maximum cumulative pessimal regret over time steps')

        # plot estimated preferences over time
        axs[1, 1].plot([i*deadline for i in range(0, T)], taskPreference.T)
        axs[1, 1].set_xlabel("seconds")
        axs[1, 1].set_title('mean estimated user-preference over time')

        if noOfUsers <=15:
            # plot choosen arms over time

            # get average assignment
            choosenTasksMeasurementsAVG = np.zeros([np.size(choosenTasksMeasurements[0], 0), np.size(choosenTasksMeasurements[0], 1)])
            for iIndex in range(len(choosenTasksMeasurements)):
                choosenTasksMeasurementsAVG += choosenTasksMeasurements[iIndex]
            choosenTasksMeasurementsAVG /= len(choosenTasksMeasurements)

            axs3[0, expList.index(iExperiment)].plot([i*deadline for i in range(0, T)], choosenTasksMeasurementsAVG.T)
            axs3[0, expList.index(iExperiment)].set_xlabel("seconds")
            axs3[0, expList.index(iExperiment)].set_title('choosen arms over time (m=0) \n(legend: optimal ass. from users persp.)')
            axs3[0, expList.index(iExperiment)].legend([i for i in tasks_of_users_optimal_stablematch])


            # plot taken arms over time

            # get average assignment
            takenTasksMeasurementsAVG = np.zeros([np.size(taskMeasurements[0], 0), np.size(taskMeasurements[0], 1)])
            for iIndex in range(len(taskMeasurements)):
                takenTasksMeasurementsAVG += taskMeasurements[iIndex]
            takenTasksMeasurementsAVG /= len(taskMeasurements)

            axs3[1, expList.index(iExperiment)].plot([i*deadline for i in range(0, T)], takenTasksMeasurementsAVG.T)
            axs3[1, expList.index(iExperiment)].set_xlabel("seconds")
            axs3[1, expList.index(iExperiment)].set_title('taken arms over time (m=0) \n(legend: optimal ass. from users persp.)')
            axs3[1, expList.index(iExperiment)].legend([i for i in tasks_of_users_optimal_stablematch])


        # plot stability over time
        if checkForStability:
            axs2[1].plot([i*deadline for i in range(1, T)], stability[1:])
            axs2[1].set_title('P(stability(t)) over t (based on true values)')
            axs2[1].set_xlabel("seconds")

            axs2[3].plot([i * deadline for i in range(1, T)], noOfUnstableMatches[1:])
            axs2[3].set_title('no. of unstable matches over t (based on true values)')
            axs2[3].set_xlabel("seconds")

        # plot exploration var over time
        axs[1, 3].plot([i*deadline for i in range(1, T)], [1/(1+min(explore_var)*t) for t in range(1, T)])
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
        globRew = getGlobalReward(noOfTasks, noOfUsers, mcsp_reward_expectation, user_reward_expectation)
        if iExperiment == len(expList) - 1:
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
        data[1].append( user_reward_expectation)
        data[2].append( mcsp_reward_expectation)
        data[3].append( 0)
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
        data[29].append( len(expList))
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
        data[53].append(countInfluence_var_e[iExperiment])
        data[54].append(countUntilAdjustment_e[iExperiment])
        data[55].append(considerPreferenceMCSP_var_e[iExperiment])
        data[56].append(epsilon_greedy_e[iExperiment])
        data[57].append(noOfMonteCarloIterations)
        data[58].append(meanOptimalReward_MCSP)
        data[59].append(rewardMeasurements_MCSP)

    axs[1, 0].legend(["exp " + str(i) for i in expList])
    axs[2, 1].legend(["exp " + str(i) for i in expList])
    axs[2, 0].legend(["exp " + str(i) for i in expList])
    axs2[2].legend(["exp " + str(i) for i in expList])
    axs2[1].legend(["exp " + str(i) for i in expList])
    axs2[0].legend(["exp " + str(i) for i in expList])
    plt.tight_layout()
    plt.show()

    with open("autoresults/" + resultsFileName + ".pkl", 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(data, f)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/