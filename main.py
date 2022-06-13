# This is a sample Python script.
import copy

import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy
from joblib import Parallel, delayed
import ray
from scipy.stats import norm


def stableMatching(n, menPreferences, womenPreferences):
    # Initially, all n men are unmarried
    unmarriedMen = list(range(n))
    # None of the men has a spouse yet, we denote this by the value None
    manSpouse = [None] * n
    # None of the women has a spouse yet, we denote this by the value None
    womanSpouse = [None] * n
    # Each man made 0 proposals, which means that
    # his next proposal will be to the woman number 0 in his list
    nextManChoice = [0] * n

    # While there exists at least one unmarried man:
    while unmarriedMen:
        # Pick an arbitrary unmarried man
        he = unmarriedMen[0]
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
    assigned_preference = np.sum(task_duration >= np.choose(np.argmax(estimated_task_duration, 1), task_duration.T).reshape(-1, 1), 1)
    return np.mean(assigned_preference)


def ucb_ca(noOfUsers, noOfTasks, T, task_duration, marge, lambda_var, deadline, explore_var, sampling_noise, enableNoAccessCount, countDegradeRate, countStartTime, countInfluence_var, countUntilAdjustment, noAccessMode, countEndTime, considerPreferenceMCSP_var, maxAllowedMarge, activateBurst, forgettingDuration_var, epsilon_greedy, takeNothingIfNegativeReward):
    # init
    noOfTimesChoosen = np.zeros([noOfUsers, noOfTasks]).astype(int)
    noOfTimesVisited = np.zeros([noOfUsers, noOfTasks]).astype(int)
    choosenTasks = -1 * np.ones(noOfUsers).astype(int)
    performedTasks = -1 * np.ones(noOfTasks).astype(int)
    lastBidOnTask = -1 * np.ones(noOfTasks)
    usersCurrentBidOnTask = -1 * np.ones(noOfUsers)
    rewardMeasurements = np.zeros([noOfUsers, T])
    taskMeasurements = -1 * np.ones([noOfUsers, T])
    taskPreferenceMeasurements = np.zeros(T)
    choosenTasksMeasurements = np.zeros([noOfUsers, T])
    estimated_task_reward = np.zeros(([noOfUsers,noOfTasks]))
    estimated_task_duration = np.zeros(([noOfUsers,noOfTasks]))
    noAccessCounter = np.zeros([noOfUsers, noOfTasks])
    lastBidAccepted = np.zeros([noOfUsers, noOfTasks])
    lastWasAccepted = np.zeros([noOfUsers,noOfTasks])
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
                        # zwei teile: einmal obendrüber problem, eimal maxAcceptedBid Problem!
                        if enableNoAccessCount==False:
                            # off
                            task_duration_user = estimated_task_duration[i][j]
                        else:
                            # check if user should explore
                            userShouldExplore = noAccessCounter[i][j] > countUntilAdjustment
                            if userShouldExplore:
                                task_duration_user = 0
                            else:
                                task_duration_user = estimated_task_duration[i][j]
                        testBid = task_duration_user * (marge + 1)

                        # ------ test if task is plausible
                        taskIsPlausible = True
                        considerPreferenceMCSP = np.random.uniform(0, 1) <= considerPreferenceMCSP_var
                        if considerPreferenceMCSP:
                            taskIsPlausible = taskIsPlausible and (testBid <= lastBidOnTask[j] or performedTasks[j] == i)
                        if taskIsPlausible:
                            plausibleTasks[j] = 1
                            testedBids[j] = testBid
                            #noAccessCounter[i][j] = 0#np.max([0, noAccessCounter[i][j] - countDegradeRate])
                            #lastBidAccepted[i][j] = np.max([lastBidAccepted[i][j], testBid])
                            lastWasAccepted[i][j] = True
                        else:
                            noAccessCounter[i][j] += 1
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
            if i in performedTasks:
                j = np.where(performedTasks == i)[0][0]
                noOfTimesVisited[i][j] += 1
                burst = 10*(i==0 and j == 0 and t<20)*(activateBurst == True)#(np.random.uniform(0, 1) <= 0.4) * task_duration[i][j]
                sampleTaskDuration = np.max([np.random.normal(loc=task_duration[i][j] + burst, scale=sampling_noise),0])
                estimated_task_duration[i][j] = sampleTaskDuration*forgettingDuration_var + estimated_task_duration[i][j]*(1-forgettingDuration_var)#
                #estimated_task_duration[i][j] = (estimated_task_duration[i][j] * (noOfTimesVisited[i][j] - 1) + sampleTaskDuration) / noOfTimesVisited[i][j]
                reward = np.min([usersCurrentBidOnTask[i],(1+maxAllowedMarge)*task_duration[i][j]])*(sampleTaskDuration <= deadline) - sampleTaskDuration #-sampleTaskDuration])#usersCurrentBidOnTask[i] - sampleTaskDuration#usersCurrentBidOnTask[i]*(sampleTaskDuration <= deadline) - sampleTaskDuration
                estimated_task_reward[i][j] = reward*forgettingDuration_var + estimated_task_reward[i][j]*(1-forgettingDuration_var)#(estimated_task_reward[i][j] * (noOfTimesVisited[i][j] - 1) + reward) / noOfTimesVisited[i][j] #
                rewardMeasurements[i][t] = reward # todo: reward hier rein oder wo anders?
                taskMeasurements[i][t] = j
                noAccessCounter[i][j] = 0


        taskPreferenceMeasurements[t] = test_on_preference(task_duration, estimated_task_duration, noOfTimesVisited, t)

    print("measured rewards: " + str(rewardMeasurements))
    print("estimated durations: "+ str(estimated_task_duration))
    print("estimated rewards: " + str(estimated_task_reward))
    print("noOfTimesVisited: " + str(noOfTimesVisited))

    return rewardMeasurements, taskPreferenceMeasurements, estimated_task_duration, choosenTasksMeasurements, taskMeasurements


#@ray.remote
def doMonteCarrloIterations(noOfMonteCarloIterations, noOfUsers, noOfTasks, T, task_duration, marge, lambda_var, deadline, explore_var, sampling_noise, enableNoAccessCount, countDegradeRate, countStartTime, countInfluence_var, countUntilAdjustment, noAccessMode, countEndTime, considerPreferenceMCSP_var, maxAllowedMarge, activateBurst, forgettingDuration_var, epsilon_greedy, takeNothingIfNegativeReward):
    rewardMeasurements = {}
    taskPreferenceMeasurements = {}
    estimated_task_duration = {}
    choosenTasksMeasurements = {}
    taskMeasurements = {}
    for m in range(noOfMonteCarloIterations):
        print(m)
        rewardMeasurements_i, taskPreferenceMeasurements_i, estimated_task_duration_i, choosenTasksMeasurements_i, taskMeasurements_i = ucb_ca(noOfUsers, noOfTasks, T, task_duration, marge, lambda_var, deadline, explore_var, sampling_noise, enableNoAccessCount, countDegradeRate, countStartTime, countInfluence_var, countUntilAdjustment, noAccessMode, countEndTime, considerPreferenceMCSP_var, maxAllowedMarge, activateBurst, forgettingDuration_var, epsilon_greedy, takeNothingIfNegativeReward)
        rewardMeasurements[m] = rewardMeasurements_i
        taskPreferenceMeasurements[m] = taskPreferenceMeasurements_i
        estimated_task_duration[m] = estimated_task_duration_i
        choosenTasksMeasurements[m] = choosenTasksMeasurements_i
        taskMeasurements[m] = taskMeasurements_i
    return rewardMeasurements, taskPreferenceMeasurements, estimated_task_duration, choosenTasksMeasurements, taskMeasurements





def print_hi(name):
    use_ray = False
    customName = "random"

    noOfTasks = 2
    noOfUsers = 2

    deadline = 10#noOfTasks + 1000

    T = 10000
    lambda_var = 0.1
    forgettingDuration_var = 0.5
    marge = 0.1
    maxAllowedMarge = 0.11
    explore_var = 20 # 100: start decrasing from t=100
    sampling_noise = 0.1
    activateBurst_e = [1,1,1,1,1]

    takeNothingIfNegativeReward_e = [False, False, True, False, False] # todo: true after a delay, to prevent task=-1 assignment if bids were to low at beginning
    enableNoAccessCount_e = [True, False, True, True, True]
    noAccessMode_e = [0, 0, 0, 0, 0]
    countDegradeRate = 30 # for mode 0
    countStartTime = 0
    countEndTime_e = [5000, 5000, 5000, 5000, 5000]
    countInfluence_var = 1/1000 # for mode 0s
    countUntilAdjustment = 20 # for mode 1
    considerPreferenceMCSP_var_e = [1, 1 , 1, 0, 0] # prob. for considering mcsp prefernce list for plausible list (1= consider it always)
    epsilon_greedy_e = [True, True,True, True, False]

    noOfMonteCarloIterations = 10

    noOfExperiments = 5


    pickelFileName = "data/" + str(noOfTasks) + str(noOfUsers) + str(customName)
    with open(pickelFileName + ".pkl", 'rb') as f:
        task_duration = pickle.load(f)

    # check for order
    for i in range(0, noOfTasks):
        _, counts = np.unique(task_duration[:, i], return_counts=True)
        if not (counts == 1).all():
            raise ValueError("no strict ordering of users" + str(task_duration))

    # Compute results: ----------------------------------------------------
    if use_ray:
        ray.init(local_mode=False)
        results = ray.get([doMonteCarrloIterations.remote(noOfMonteCarloIterations, noOfUsers, noOfTasks, T, task_duration, marge, lambda_var,
                                deadline, explore_var, sampling_noise, enableNoAccessCount_e[iExperiment],
                                countDegradeRate, countStartTime, countInfluence_var, countUntilAdjustment,
                                noAccessMode_e[iExperiment], countEndTime_e[iExperiment], considerPreferenceMCSP_var_e[iExperiment], maxAllowedMarge, activateBurst_e[iExperiment], forgettingDuration_var, epsilon_greedy_e[iExperiment], takeNothingIfNegativeReward_e[iExperiment]) for iExperiment in range(noOfExperiments)])
        ray.shutdown()

    else:
        results = Parallel(n_jobs=noOfExperiments)(delayed(doMonteCarrloIterations)(noOfMonteCarloIterations, noOfUsers, noOfTasks, T, task_duration, marge, lambda_var,
                            deadline, explore_var, sampling_noise, enableNoAccessCount_e[iExperiment],
                            countDegradeRate, countStartTime, countInfluence_var, countUntilAdjustment,
                           noAccessMode_e[iExperiment], countEndTime_e[iExperiment], considerPreferenceMCSP_var_e[iExperiment], maxAllowedMarge, activateBurst_e[iExperiment], forgettingDuration_var, epsilon_greedy_e[iExperiment], takeNothingIfNegativeReward_e[iExperiment]) for iExperiment in range(noOfExperiments))


    # Plot results: ----------------------------------------------------

    fig, axs = plt.subplots(3, 4)

    for iExperiment in range(noOfExperiments):

        # get from ray results
        rewardMeasurements, taskPreferenceMeasurements, estimated_task_duration, choosenTasksMeasurements, taskMeasurements = results[iExperiment]

        prefer_tasks = []
        prefer_users = []
        # get task duration with deadline incroprorated
        if sampling_noise > 0:
            p_max_allowed = scipy.stats.norm.cdf(np.minimum(deadline*np.ones([noOfUsers,noOfTasks]), task_duration*(1+marge)), loc=task_duration, scale=sampling_noise)
            p_0 = scipy.stats.norm.cdf(0, loc=task_duration, scale=sampling_noise)
            task_duration_real = np.multiply(p_max_allowed-p_0, task_duration)
            #task_duration_real, var, skew, kurt = truncnorm.stats(a, b, moments='mvsk', scale=sampling_noise, loc=task_duration)
        else:
            task_duration_real = np.multiply(task_duration, task_duration<=deadline)
        # more:
        optimum_deadlines = "hard"
        if optimum_deadlines == "soft":
            task_duration_with_deadlines_userview = task_duration_real#np.multiply(task_duration_real, (task_duration <= deadline) * 1)
            task_duration_with_deadlines_taskview = task_duration_real#np.multiply(task_duration_real, (task_duration <= deadline)) * 1 + (task_duration > deadline)*10000
        else:
            task_duration_with_deadlines_userview = np.multiply(task_duration_real, (task_duration <= deadline) * 1)
            task_duration_with_deadlines_taskview = np.multiply(task_duration_real, (task_duration <= deadline)) * 1 + (task_duration > deadline)*10000
        # tasks preferences (tasks prefer shorter durations)
        for i in range(noOfTasks):
            prefer_tasks.append((np.argsort(task_duration_with_deadlines_taskview[:,i])).tolist())
        # users pferences
        for i in range(noOfUsers):
            prefer_users.append((np.flip(np.argsort(task_duration_with_deadlines_userview[i][:]))).tolist())

        # advantage: tasks
        users_of_tasks_pessimal_stablematch = stableMatching(noOfUsers, prefer_tasks, prefer_users)
        # advantage: users
        tasks_of_users_optimal_stablematch = stableMatching(noOfUsers, prefer_users, prefer_tasks)

        # calculate stable optimal reward of users
        meanOptimalReward = np.zeros(noOfUsers)
        optimalAssignment = np.zeros(noOfUsers)
        for i in range(noOfUsers):
            meanOptimalReward[i] = marge*task_duration_real[i][tasks_of_users_optimal_stablematch[i]]
            optimalAssignment[i] = tasks_of_users_optimal_stablematch[i]

        # calculate stable pessimal reward of users
        meanPessimalReward = np.zeros(noOfUsers)
        pessimalAssignment = np.zeros(noOfUsers)
        for i in range(noOfTasks):
            meanPessimalReward[users_of_tasks_pessimal_stablematch[i]] = marge * task_duration_real[users_of_tasks_pessimal_stablematch[i]][i]
            pessimalAssignment[users_of_tasks_pessimal_stablematch[i]] = i

        if np.sum(meanPessimalReward) == 0:
            raise Exception("pessimal reward=0! pessimal match would be no task assignment. increase minimum task duration")

        print("task_duration_with_deadlines_userview: " + str(task_duration_with_deadlines_userview))
        print("task_duration_with_deadlines_taskview: " + str(task_duration_with_deadlines_taskview))
        print("mean optimal match reward: " + str(meanOptimalReward))
        print("mean pessimal match reward: " + str(meanPessimalReward))
        #print("stable matching unique?: " + str((optimalAssignment==pessimalAssignment).all())) # wrong?
        print("optimal match: " + str(optimalAssignment))
        print("pessimal match: " + str(pessimalAssignment))
        print("task duration:")
        print(task_duration)
        # ----------------------------------------------------

        # caluclate metrics
        stableRegret = np.zeros([noOfUsers, T])
        taskPreference = np.zeros([1, T])
        stability = np.zeros(T)
        for m in range(noOfMonteCarloIterations):
            for t in range(0, T):
                # calculate average pessimal regret
                for i in range(0, noOfUsers):
                    stableRegret[i][t] = stableRegret[i][t]*m/(1+m) + 1/(m+1)*(meanPessimalReward[i] - rewardMeasurements[m][i][t])

                # calculate stability: stability no user
                stable = True
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
                        break

                stability[t] = stability[t]*m/(1+m) + 1/(m+1)*stable

            taskPreference = taskPreference*m/(1+m) + taskPreferenceMeasurements[m]*1/(1+m)


        # plot regret
        axs[0, 0].plot(np.arange(1, T+1), stableRegret.transpose())
        pessimalOptimalGap = np.array((meanPessimalReward - meanOptimalReward))
        #for i in range(noOfUsers):
        #    axs[0, 0].axhline(y=pessimalOptimalGap[i], color='r', linestyle='--')
        axs[0, 0].set_title('average stable pessimal regret over time steps')
        axs[0, 0].legend(["user " + str(i) for i in range(noOfUsers)])

        # plot cum regret
        axs[2, 0].plot(np.arange(1, T + 1), np.cumsum(np.array(stableRegret),1).transpose())
        axs[2, 0].set_title('average cumulative pessimal regret over time steps')
        axs[2, 0].legend(["user " + str(i) for i in range(noOfUsers)])
        for i in range(noOfUsers):
            axs[2, 0].plot([pessimalOptimalGap[i]*t for t in range(1,T+1)], color='r', linestyle='--')

        # plot max regret
        axs[1, 0].plot(np.arange(1, T + 1), np.max(stableRegret, 0))
        if iExperiment==noOfExperiments-1:
            axs[1, 0].axhline(y=np.max(pessimalOptimalGap,0), color='r', linestyle='--')
        axs[1, 0].set_title('average maximum pessimal regret over time steps')
        # todo: achutng, noise!!! wird immer leicht nach oben vershcoben sein ??

        # plot max cum regret
        axs[2, 1].plot(np.arange(1, T + 1), np.max(np.cumsum(np.array(stableRegret),1).transpose(), 1))
        if iExperiment==noOfExperiments-1:
            axs[2, 1].plot(np.max([pessimalOptimalGap*t for t in range(1,T+1)], 1), color='r', linestyle='--')
        axs[2, 1].set_title('average maximum cumulative pessimal regret over time steps')

        # plot estimated preferences over time
        axs[1, 1].plot([i for i in range(0, T)], taskPreference.T)
        axs[1, 1].set_title('mean estimated user-preference over time')

        # plot choosen arms over time
        axs[0, 2].plot([i for i in range(0, T)], choosenTasksMeasurements[0].T)
        axs[0, 2].set_title('choosen arms over time (m=0) \n(legend: optimal ass. from users persp.)')
        axs[0, 2].legend([i for i in tasks_of_users_optimal_stablematch])


        # plot taken arms over time
        axs[0, 1].plot([i for i in range(0, T)], taskMeasurements[0].T)
        axs[0, 1].set_title('taken arms over time (m=0) \n(legend: optimal ass. from users persp.)')
        axs[0, 1].legend([i for i in tasks_of_users_optimal_stablematch])

        # plot stability over time
        axs[1, 2].plot([i for i in range(1, T)], stability[1:])
        axs[1, 2].set_title('P(stability(t)) over t (based on true values)')

        # plot exploration var over time
        axs[1, 3].plot([((1 / t * explore_var)*((1 / t * explore_var)<=1) + ((1 / t * explore_var)>1)*1) for t in range(1, T)])
        axs[1, 3].set_title('exploration probability over time')



        # theorem 6
        #delta = 1
        #epsilon = (1-lambda_var)*lambda_var**(noOfUsers-1)
        #rightSide = noOfUsers**5*noOfTasks**2/(epsilon**(noOfUsers**4+1))*np.log(T)*(1/delta**2*np.log(T) + 3)
        #rewardG = np.zeros(noOfUsers)
        #for i in range(0, noOfUsers):
        #    rewardG[i] = rightSide*24*np.max([np.max(meanPessimalReward[i] - task_duration[i][:]),meanPessimalReward[i]])

    axs[1, 0].legend(["exp " + str(i) for i in range(noOfExperiments)])
    axs[2, 1].legend(["exp " + str(i) for i in range(noOfExperiments)])
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/