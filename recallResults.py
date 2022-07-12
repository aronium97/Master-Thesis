# This is a sample Python script.
import copy

import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed


def checkTheStability(data, iExperiment):
    meanPessimalReward = data[0][iExperiment]
    user_reward_expectation = data[1][iExperiment]
    mcsp_reward_expectation = data[2][iExperiment]
    _ = data[3][iExperiment]
    meanOptimalReward = data[4][iExperiment]
    meanOptimalGlobalReward = data[5][iExperiment]
    meanPessimalGlobalReward = data[6][iExperiment]
    optimalAssignment = data[7][iExperiment]
    pessimalAssignment = data[8][iExperiment]
    task_duration = data[9][iExperiment]
    stability = data[10][iExperiment]
    noOfUnstableMatches = data[11][iExperiment]
    taskPreference = data[12][iExperiment]
    globalReward = data[13][iExperiment]
    estimated_task_duration = data[14][iExperiment]
    estimated_task_reward = data[15][iExperiment]
    noOfTimesVisited = data[16][iExperiment]
    overDeadlineCounter = data[17][iExperiment]
    noOfTimesChoosen = data[18][iExperiment]
    freeSensingDone = data[19][iExperiment]
    T = data[20][iExperiment]
    deadline = data[21][iExperiment]
    stableRegret = data[22][iExperiment]
    noOfUsers = data[23][iExperiment]
    choosenTasksMeasurements = data[24][iExperiment]
    taskMeasurements = data[25][iExperiment]
    tasks_of_users_optimal_stablematch = data[26][iExperiment]
    noOfTasks = data[27][iExperiment]
    explore_var = data[28][iExperiment]
    noOfExperiments = data[29][iExperiment]
    globRew = data[30][iExperiment]
    prefer_users = data[31][iExperiment]
    prefer_tasks = data[32][iExperiment]

    noOfMonteCarloIterations = len(taskMeasurements)
    stability = np.zeros(T)
    noOfUnstableMatches = np.zeros(T)
    for m in tqdm(range(noOfMonteCarloIterations)):
        for t in range(0, T):
            # calculate stability: stability no user
            stable = True
            unstableMatchesCount = 0
            if True:
                for i in range(0, noOfUsers):
                    stableUser = True
                    # check if user would find a better task that would prefer him
                    j = int(taskMeasurements[m][i][t])
                    if not (prefer_users[i][0] == j):
                        if j == -1:
                            betterTasks = np.array(prefer_users[i])
                        else:
                            matchIndex = np.where(np.array(prefer_users[i]) == j)[0][0]
                            betterTasks = prefer_users[i][0:matchIndex]
                        # would task prefer user over its current user
                        for jbetter in betterTasks:
                            matchIndexTask = np.where(np.array(prefer_tasks[jbetter]) == i)[0][0]
                            # get current user of task
                            if jbetter in list(taskMeasurements[m][:, t]):
                                iCurrent = list(taskMeasurements[m][:, t]).index(jbetter)
                                iCurrentIndex = np.where(np.array(prefer_tasks[jbetter]) == iCurrent)[0][0]
                                if iCurrentIndex > matchIndexTask:
                                    stableUser = False
                                    break
                            else:
                                stableUser = False
                                break

                    if stableUser == False:
                        stable = False
                        unstableMatchesCount += 1

            stability[t] = stability[t] * m / (1 + m) + 1 / (m + 1) * stable
            noOfUnstableMatches[t] = noOfUnstableMatches[t] * m / (1 + m) + 1 / (m + 1) * unstableMatchesCount
    return stability, noOfUnstableMatches

def print_hi(name):

    showMatrices = False
    checkForStability = True

    noOfExperiments = 7

    flname = "10m30 top alles"#"legit 10m20 soft braker"
    pickelFileName = "autoresults/" + flname
    with open(pickelFileName + ".pkl", 'rb') as f:
        data = pickle.load(f)

    # Plot results: ----------------------------------------------------

    fig, axs = plt.subplots(3, 4)
    fig2, axs2 = plt.subplots(1, 4)
    if showMatrices: figMatrizes, axsMatrizes = plt.subplots(5, noOfExperiments + 1)

    if checkForStability:
        results = Parallel(n_jobs=noOfExperiments)(
            delayed(checkTheStability)(data, iExperiment) for iExperiment in
            range(noOfExperiments))

    for iExperiment in range(noOfExperiments):

        meanPessimalReward = data[0][iExperiment]
        user_reward_expectation = data[1][iExperiment]
        mcsp_reward_expectation = data[2][iExperiment]
        _ = data[3][iExperiment]
        meanOptimalReward = data[4][iExperiment]
        meanOptimalGlobalReward = data[5][iExperiment]
        meanPessimalGlobalReward = data[6][iExperiment]
        optimalAssignment = data[7][iExperiment]
        pessimalAssignment = data[8][iExperiment]
        task_duration = data[9][iExperiment]
        stability = data[10][iExperiment]
        noOfUnstableMatches = data[11][iExperiment]
        taskPreference = data[12][iExperiment]
        globalReward = data[13][iExperiment]
        estimated_task_duration = data[14][iExperiment]
        estimated_task_reward = data[15][iExperiment]
        noOfTimesVisited = data[16][iExperiment]
        overDeadlineCounter = data[17][iExperiment]
        noOfTimesChoosen = data[18][iExperiment]
        freeSensingDone = data[19][iExperiment]
        T = data[20][iExperiment]
        deadline = data[21][iExperiment]
        stableRegret = data[22][iExperiment]
        noOfUsers = data[23][iExperiment]
        choosenTasksMeasurements = data[24][iExperiment]
        taskMeasurements = data[25][iExperiment]
        tasks_of_users_optimal_stablematch = data[26][iExperiment]
        noOfTasks = data[27][iExperiment]
        explore_var = data[28][iExperiment]
        noOfExperiments = data[29][iExperiment]
        globRew = data[30][iExperiment]
        prefer_users = data[31][iExperiment]
        prefer_tasks = data[32][iExperiment]

        if checkForStability:
            stability, noOfUnstableMatches = results[iExperiment]

        print("//////////// experiment " + str(iExperiment))


        if np.sum(meanPessimalReward) == 0:
            raise Exception("pessimal reward=0! pessimal match would be no task assignment. increase minimum task duration")

        print("task_duration: " + str(task_duration))
        print("user_reward_expectation: " + str(user_reward_expectation))
        print("mcsp_reward_expectation: " + str(mcsp_reward_expectation))
        print("mean optimal match reward: " + str(meanOptimalReward))
        print("mean pessimal match reward: " + str(meanPessimalReward))
        print("mean optimal match global reward: " + str(meanOptimalGlobalReward))
        print("mean pessimal match global reward: " + str(meanPessimalGlobalReward))
        print("optimal match: " + str(optimalAssignment))
        print("pessimal match: " + str(pessimalAssignment))
        # ----------------------------------------------------

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

    axs[1, 0].legend(["exp " + str(i) for i in range(noOfExperiments)])
    axs[2, 1].legend(["exp " + str(i) for i in range(noOfExperiments)])
    axs[2, 0].legend(["exp " + str(i) for i in range(noOfExperiments)])
    axs2[2].legend(["exp " + str(i) for i in range(noOfExperiments)])
    axs2[1].legend(["exp " + str(i) for i in range(noOfExperiments)])
    axs2[0].legend(["exp " + str(i) for i in range(noOfExperiments)])
    plt.tight_layout()
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/