# This is a sample Python script.
import copy

import numpy as np
import pickle
import matplotlib.pyplot as plt

def wPrefersM1OverM(prefer, w, m, m1, N):

    for i in range(N):

        if (prefer[w][i] == m1):
            return True

        if (prefer[w][i] == m):
            return False

def stableMarriage(prefer, N):

    wPartner = [-1 for i in range(N)]

    mFree = [False for i in range(N)]

    freeCount = N

    # While there are free men
    while (freeCount > 0):

        m = 0
        while (m < N):
            if (mFree[m] == False):
                break
            m += 1

        i = 0
        while i < N and mFree[m] == False:
            w = prefer[m][i]

            if (wPartner[w - N] == -1):
                wPartner[w - N] = m
                mFree[m] = True
                freeCount -= 1

            else:

                m1 = wPartner[w - N]

                if (wPrefersM1OverM(prefer, w, m, m1, N) == False):
                    wPartner[w - N] = m
                    mFree[m] = True
                    mFree[m1] = False
            i += 1

    print("advantage player ", " disadvantage player")
    for i in range(N):
        print(i, "\t", wPartner[i])

    return wPartner

def test_on_preference(task_duration, estimated_task_duration):
    assigned_preference = np.sum(task_duration >= np.choose(np.argmax(estimated_task_duration, 1), task_duration.T).reshape(-1, 1), 1)
    return np.mean(assigned_preference)


def ucb_ca(noOfUsers, noOfTasks, T, estimated_task_durationn, task_duration, marge, lambda_var):
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

    # algorithm
    for t in range(1, T):
        print(t)
        if t == 1000:
            print("hh")
        for i in range(0, noOfUsers):
            if t == 1:
                j = np.random.randint(0, noOfTasks)
                choosenTasks[i] = j
                usersCurrentBidOnTask[i] = estimated_task_durationn[i][choosenTasks[i]] * (marge + 1)
            else:
                D = (np.random.uniform(0, 1) <= lambda_var) * 1
                if D == 1:
                    True == True
                    # do nothing
                else:
                    # estimate cost
                    # todo: durch estimation ersetzen
                    plausibleTasks = np.zeros(noOfTasks)
                    for j in range(0, noOfTasks):
                        task_duration_user = estimated_task_durationn[i][j]
                        plannedBid = task_duration_user * (marge + 1)
                        if plannedBid < lastBidOnTask[j] or performedTasks[j] == i:
                            plausibleTasks[j] = 1  # task is plausible
                    # calculate UCB
                    estimatedDuration = estimated_task_durationn[i]
                    ucbBound = (estimatedDuration + np.sqrt(np.log(t) / noOfTimesVisited[
                        i])) * plausibleTasks  # todo: nicht mit 0 machen sondern rausnehmen
                    # pull max ucb
                    if np.sum(plausibleTasks) > 0:
                        #print("now:")
                        #print(i)
                        #print(ucbBound)
                        #print(estimated_task_durationn[i])
                        choosenTasks[i] = np.random.choice(((np.argwhere(ucbBound == np.nanmax(ucbBound)))).flatten())
                        usersCurrentBidOnTask[i] = estimated_task_durationn[i][choosenTasks[i]] * (marge + 1)
                    else:
                        # no task can be choosen
                        choosenTasks[i] = -1
                        usersCurrentBidOnTask[i] = np.nan
        # decide for winners
        tasksHaveUser = np.array([False] * noOfTasks)
        userIterate = np.arange(noOfUsers)
        np.random.shuffle(userIterate)
        for i in userIterate:
            choosenTasksMeasurements[i][t] = choosenTasks[i]
            if ~(choosenTasks[i] == -1):
                noOfTimesChoosen[i][choosenTasks[i]] += 1
                if np.min(usersCurrentBidOnTask[choosenTasks == choosenTasks[i]]) >= usersCurrentBidOnTask[
                    i]:
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
                sampleTaskDuration = np.random.normal(loc=task_duration[i][j], scale=0.0)
                estimated_task_durationn[i][j] = (estimated_task_durationn[i][j] * (
                            noOfTimesVisited[i][j] - 1) + sampleTaskDuration) / noOfTimesVisited[i][j]
                rewardMeasurements[i][t] = usersCurrentBidOnTask[i] - sampleTaskDuration
                taskMeasurements[i][t] = j

        taskPreferenceMeasurements[t] = test_on_preference(task_duration, estimated_task_durationn)

    return rewardMeasurements, taskPreferenceMeasurements, estimated_task_durationn, choosenTasksMeasurements, taskMeasurements




def print_hi(name):
    customName = "random"

    noOfTasks = 5
    noOfUsers = 5
    N = noOfUsers  # todo: noch ändern

    T = 1500
    lambda_var = 0.1
    marge = 0.1

    noOfMonteCarloIterations = 1

    pickelFileName = "data/" + str(noOfTasks) + str(noOfUsers) + str(customName)
    with open(pickelFileName + ".pkl", 'rb') as f:
        task_duration, estimated_task_duration_init = pickle.load(f)

    rewardMeasurements = {}
    taskPreferenceMeasurements = {}
    estimated_task_duration = {}
    choosenTasksMeasurements = {}
    taskMeasurements = {}
    for m in range(noOfMonteCarloIterations):
        print(m)
        rewardMeasurements_i, taskPreferenceMeasurements_i, estimated_task_duration_i, choosenTasksMeasurements_i, taskMeasurements_i = ucb_ca(noOfUsers, noOfTasks, T, copy.deepcopy(estimated_task_duration_init), task_duration, marge, lambda_var)
        rewardMeasurements[m] = rewardMeasurements_i
        taskPreferenceMeasurements[m] = taskPreferenceMeasurements_i
        estimated_task_duration[m] = estimated_task_duration_i
        choosenTasksMeasurements[m] = choosenTasksMeasurements_i
        taskMeasurements[m] = taskMeasurements_i


    #----------------------------------------------------
    # Driver Code (first: advantage player gives preferences (all coded +N), second: disadvantage player gives preferencesy)

    # advantage: user
    prefer = []
    # users preferences (users prefer longer durations)
    for i in range(noOfUsers):
        prefer.append((noOfUsers + np.flip(np.argsort(task_duration[i][:]))).tolist())
    # tasks pferences (tasks prefer shorter durations)
    for i in range(noOfTasks):
        prefer.append(np.argsort(task_duration[:][i]).tolist())
    tasks_of_users_optimal_stablematch = stableMarriage(prefer, N)

    # advantage: tasks
    prefer = []
    # tasks preferences (tasks prefer shorter durations)
    for i in range(noOfTasks):
        prefer.append((noOfTasks + np.argsort(task_duration[:][i])).tolist())
    # users pferences
    for i in range(noOfUsers):
        prefer.append((np.flip(np.argsort(task_duration[i][:]))).tolist())
    users_of_tasks_pessimal_stablematch = stableMarriage(prefer, N)

    # calculate stable optimal reward of users
    meanOptimalReward = np.zeros(noOfUsers)
    optimalAssignment = np.zeros(noOfUsers)
    for i in range(noOfUsers):
        meanOptimalReward[i] = marge*task_duration[i][tasks_of_users_optimal_stablematch[i]]
        optimalAssignment[i] = tasks_of_users_optimal_stablematch[i]

    # calculate stable pessimal reward of users
    meanPessimalReward = np.zeros(noOfUsers)
    pessimalAssignment = np.zeros(noOfUsers)
    for i in range(noOfTasks):
        meanPessimalReward[users_of_tasks_pessimal_stablematch[i]] = marge * task_duration[users_of_tasks_pessimal_stablematch[i]][i]
        pessimalAssignment[users_of_tasks_pessimal_stablematch[i]] = i

    print("mean optimal match reward: " + str(meanOptimalReward))
    print("mean pessimal match reward: " + str(meanPessimalReward))
    print("stable matching unique?: " + str((optimalAssignment==pessimalAssignment).all()))
    print("optimal match: " + str(optimalAssignment))
    print("pessimal match: " + str(pessimalAssignment))
    print("task duration:")
    print(task_duration)
    # ----------------------------------------------------

    # caluclate metrics
    stableRegret = np.zeros([noOfUsers, T])
    taskPreference = np.zeros([1, T])
    for m in range(noOfMonteCarloIterations):
        # calculate regret (right now: stable optimal regret! optimal regret should be pessimal regret)
        a = np.zeros([noOfUsers, T])
        for t in range(0, T):
            for i in range(0, noOfUsers):
                a[i][t] = (meanPessimalReward[i] - rewardMeasurements[m][i][t])
                stableRegret[i][t] = stableRegret[i][t]*m/(1+m) + 1/(m+1)*a[i][t]#t*overallReward[i] - np.cumsum(rewardMeasurements[i])[t]  # t*userPessRegret[i] -

        taskPreference = taskPreference*m/(1+m) + taskPreferenceMeasurements[m]*1/(1+m)


    # plot regret
    fig, axs = plt.subplots(2, 3)
    axs[0, 0].plot(np.arange(1, T+1), stableRegret.transpose())
    for i in range(noOfUsers):
        cc = np.array((meanPessimalReward - meanOptimalReward))
        axs[0, 0].axhline(y=cc[:][i], color='r', linestyle='--')
    #axs[0, 0].axhline(y=np.min(meanPessimalReward - meanOptimalReward, 0), color='r', linestyle='--')
    axs[0, 0].set_title('stable regret over time steps')

    # plot max regret
    axs[1, 0].plot(np.arange(1, T + 1), np.max(stableRegret, 0))
    axs[1, 0].set_title('maximum regret over time steps')

    # plot estimated preferences over time
    axs[1, 1].plot([i for i in range(0, T)], taskPreference.T)
    axs[1, 1].set_title('mean estimated user-preference over time')

    # plot choosen arms over time
    axs[0, 2].plot([i for i in range(0, T)], choosenTasksMeasurements[0].T)
    axs[0, 2].set_title('choosen arms over time (m=0) \n(legend: optimal ass. from users persp.)')
    axs[0, 2].legend([i for i in np.argmax(task_duration,1)])


    # plot taken arms over time
    axs[0, 1].plot([i for i in range(0, T)], taskMeasurements[0].T)
    axs[0, 1].set_title('taken arms over time (m=0) \n(legend: optimal ass. from users persp.)')
    axs[0, 1].legend([i for i in np.argmax(task_duration, 1)])





    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
