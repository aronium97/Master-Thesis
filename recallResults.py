# This is a sample Python script.
import copy
import os

import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

#plt.style.reload_library()
#plt.style.use(['science', 'grid'])
#plt.rc('font', family='lmodern', serif='Times')
##plt.rc('text', usetex=True)
#plt.rc('xtick', labelsize=6)
#plt.rc('ytick', labelsize=6)
#plt.rc('axes', labelsize=6)
#plt.rc('legend',fontsize=6)

# bernd style:
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=6)
plt.rc('ytick', labelsize=6)
plt.rc('axes', labelsize=6)
plt.rc('legend', fontsize=6)
#Size calculation
# width as measured in inkscape
widthFig = 3.487 / 1.8
heightFig = widthFig
#Data for plotting
xAxisName='X axis'
yAxisName='Y axis'
label0 = []
label0.append('CA-MAB-FS')
label0.append('CA-MAB')
label0.append('Greedy')
label0.append('Random')
#Color palette and markers
mycolors = ['red', '#0F7173', '#D8A47F', '#41EAD4']
mymarkers = ['x', 'v', 'x', 'o']
#Markers on the x axis
xticks = (0,200,500,800,1100)
xlimits = (0,1200)
yticks = (1,1.5,2,2.5)
ylimits = (1,2.5)



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
    deadline = 1200/T#data[21][iExperiment]
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
            j_unstable = []
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
                                if iCurrentIndex > matchIndexTask and not(jbetter in j_unstable):
                                    stableUser = False
                                    j_unstable.append(jbetter)
                                    break
                            elif not(jbetter in j_unstable):
                                stableUser = False
                                j_unstable.append(jbetter)
                                break

                    if stableUser == False:
                        stable = False
                        unstableMatchesCount += 1

            stability[t] = stability[t] * m / (1 + m) + 1 / (m + 1) * stable
            noOfUnstableMatches[t] = noOfUnstableMatches[t] * m / (1 + m) + 1 / (m + 1) * unstableMatchesCount
    return stability, noOfUnstableMatches

def print_hi(name):

    cumRegretBernd = []
    stabilityBernd = []
    overallRewardBernd = []
    noOfFreeSensingOffersBernd = []




    showMatrices = False
    checkForStability = False

    noOfExperiments = 3

    showLatex = False

    flname = "zzz_10m30mitausfallen"#"legit 10m20 soft braker"
    pickelFileName = "autoresults/" + flname
    with open(pickelFileName + ".pkl", 'rb') as f:
        data = pickle.load(f)

    # Plot results: ----------------------------------------------------

    fig, axs = plt.subplots(3, 4)
    fig2, axs2 = plt.subplots(1, 4)

    if showLatex:
        figh, axsh = plt.subplots(1, 4)
        # width as measured in inkscape
        #width = 3.487*2.3
        #height = width / 1.618 / 2.5
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=.19, bottom=.19, right=.99, top=.97)

        # bernd:
        fig5, ax5 = plt.subplots()
        fig5.subplots_adjust(left=.20, bottom=.23, right=0.94, top=.72)
        fig6, ax6 = plt.subplots()
        fig6.subplots_adjust(left=.20, bottom=.23, right=.99, top=.76)
        fig7, ax7 = plt.subplots()
        fig7.subplots_adjust(left=.20, bottom=.23, right=0.99, top=.72)
        fig8, ax8 = plt.subplots()
        fig8.subplots_adjust(left=.20, bottom=.23, right=.99, top=.76)


    if showLatex:
        expList = list([8,12,13,14])#list([9,12,13,14])#list([4,9,10,11])#list([6,5,4,2])
    else:
        expList = range(noOfExperiments)#list([8,12,13,14])#range(noOfExperiments)#list([0,2,4,8,9,10])#
        #expList = list([5, 12, 13, 14])

    if showMatrices: figMatrizes, axsMatrizes = plt.subplots(5, noOfExperiments + 1)

    if checkForStability:
        stableValsFileName = pickelFileName + "stableVals" + ".pkl"
        if os.path.isfile(stableValsFileName):
            print("stableVals exists")
            with open(stableValsFileName, 'rb') as f:
                results = pickle.load(f)
        else:
            print("recalculate stableVals")
            results = Parallel(n_jobs=noOfExperiments)(
                delayed(checkTheStability)(data, iExperiment) for iExperiment in
                expList)  # range(noOfExperiments))
            with open(stableValsFileName, 'wb') as f:
                pickle.dump(results, f)


    maxCumSumRegretEnd = []
    maxCumSumRegretEnd_MCSP = []
    for iExperiment in expList:#range(noOfExperiments):#expList:#range(noOfExperiments):

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
        deadline = 1#1/1.1#data[21][iExperiment]
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
        meanOptimalReward_MCSP = data[58][iExperiment]
        rewardMeasurements_MCSP = data[59][iExperiment]

        if checkForStability:
            stability, noOfUnstableMatches = results[expList.index(iExperiment)]

        print("//////////// experiment " + str(iExperiment))


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

        subsampled_idx = np.arange(0, T,100)

        # plot regret
        axs[0, 0].plot(np.arange(1, T+1)*deadline, stableRegret.transpose())
        pessimalOptimalGap = np.array((meanPessimalReward - meanOptimalReward))
        #for i in range(noOfUsers):
        #    axs[0, 0].axhline(y=pessimalOptimalGap[i], color='r', linestyle='--')
        axs[0, 0].set_title('average stable pessimal regret over time steps')
        axs[0, 0].set_xlabel("seconds s ")
        axs[0, 0].legend(["user " + str(i) for i in range(noOfUsers)])

        # plot mcsp reward
        #axs[2, 2].plot(np.arange(1, T + 1) * deadline,
        #               np.cumsum(np.sum(meanOptimalReward_MCSP - rewardMeasurements_MCSP.transpose(), 1)))
        axs[2, 2].plot(np.arange(1, T + 1) * deadline,
                       np.cumsum(np.sum(np.array(rewardMeasurements_MCSP.transpose()), 1).transpose()))
        axs[2, 2].set_title('average stable mcsp regret over time steps')
        axs[2, 2].set_xlabel("seconds s ")
        maxCumSumRegretEnd_MCSP.append(
            np.cumsum(np.sum(np.array(rewardMeasurements_MCSP.transpose()), 1).transpose())[-1])



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
        axs2[0].plot(np.arange(1, T + 1)*deadline, [np.max(np.cumsum(np.array(stableRegret),1).transpose(), 1)[i-1]/i for i in range(1,T+1)])
        if iExperiment==noOfExperiments-1:
            axs2[0].plot(np.arange(1, T + 1)*deadline, np.max([pessimalOptimalGap*t*deadline for t in range(1,(T+1))], 1), color='r', linestyle='--')
        axs2[0].set_xlabel("seconds s")
        axs2[0].set_title('average maximum cumulative pessimal regret over time steps')
        maxCumSumRegretEnd.append(np.max(np.cumsum(np.array(stableRegret),1).transpose(), 1)[-1])

        # h: plot max cum regret
        if showLatex:
            axsh[0].plot(subsampled_idx * deadline, np.max(np.cumsum(np.array(stableRegret), 1).transpose()[subsampled_idx], 1), marker=mymarkers[expList.index(iExperiment)], markersize=4, color=mycolors[expList.index(iExperiment)])
            if iExperiment == 2:
                axsh[0].plot(subsampled_idx * deadline,
                             np.max([pessimalOptimalGap * t * deadline for t in subsampled_idx], 1), color='r',
                             linestyle='--')
            axsh[0].set_xlabel("Time [s]")
            axsh[0].set_ylabel(r"$R_k$ [monetary units]")

            # h2: plot max cum regret
            if iExperiment == 1:
                ax5.plot(subsampled_idx * deadline,
                         np.max(np.cumsum(np.array(stableRegret), 1).transpose()[subsampled_idx], 1),
                         color=mycolors[expList.index(iExperiment)], label=label0[expList.index(iExperiment)])
            else:
                ax5.plot(subsampled_idx * deadline,
                             np.max(np.cumsum(np.array(stableRegret), 1).transpose()[subsampled_idx], 1),
                             marker=mymarkers[expList.index(iExperiment)], markersize=4,
                             color=mycolors[expList.index(iExperiment)], label=label0[expList.index(iExperiment)])
            if iExperiment == 6:
                ax5.plot(subsampled_idx * deadline,
                             np.max([pessimalOptimalGap * t * deadline for t in subsampled_idx], 1), color='r',
                             linestyle='--', label="Galey-Shapley")
            ax5.set_xlabel("Time [s]")
            ax5.set_ylabel(r"$R_k$ [monetary units]")

            # bernd var:
            cumRegretBernd.append(np.max(np.cumsum(np.array(stableRegret), 1).transpose(), 1))

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

            if showLatex:
                #h:
                axsh[2].plot([i * deadline for i in range(1, T)], stability[1:], marker=mymarkers[expList.index(iExperiment)], markersize=4, color=mycolors[expList.index(iExperiment)])
                axsh[2].set_title('probability of stability')
                axsh[2].set_xlabel("Time [s]")

                #h2:
                if iExperiment == 1:
                    ax8.plot([i * deadline for i in subsampled_idx], noOfUnstableMatches[1:][subsampled_idx],
                             color=mycolors[expList.index(iExperiment)], label=label0[expList.index(iExperiment)])
                else:
                    ax8.plot([i * deadline for i in subsampled_idx], noOfUnstableMatches[1:][subsampled_idx],
                                 marker=mymarkers[expList.index(iExperiment)], markersize=4,
                                 color=mycolors[expList.index(iExperiment)], label=label0[expList.index(iExperiment)])
                ax8.set_ylabel("Number of blocking pairs")
                ax8.set_xlabel("Time [s]")




            axs2[3].plot([i * deadline for i in range(1, T)], noOfUnstableMatches[1:])
            axs2[3].set_title('no. of unstable matches over t (based on true values)')
            axs2[3].set_xlabel("seconds")

            # bernd var
            stabilityBernd.append(noOfUnstableMatches)

        # plot exploration var over time
        axs[1, 3].plot([i*deadline for i in range(1, T)], [((1 / t * explore_var)*((1 / t * explore_var)<=1) + ((1 / t * explore_var)>1)*1) for t in range(1, T)])
        axs[1, 3].set_xlabel("seconds")
        axs[1, 3].set_title('exploration probability over time')

        # plot freesensingDone over time
        axs[2, 3].plot([i * deadline for i in range(0, T)], np.cumsum(freeSensingDone)) #[np.cumsum(freeSensingDone)[i-1]/(i*noOfTasks) for i in range(1,T+1)])
        axs[2, 3].set_xlabel("seconds")
        axs[2, 3].set_title('cumulative no. of free sensing bids')

        if showLatex:
            # h: plot freesensingDone over time
            if iExperiment == 2:
                axsh[3].plot([i * deadline for i in subsampled_idx], np.cumsum(freeSensingDone)[subsampled_idx], marker=mymarkers[expList.index(iExperiment)], markersize=4, color=mycolors[expList.index(iExperiment)])
                axsh[3].set_xlabel("Time [s]")
                axsh[3].set_ylabel("cumulated number of free-sensing bids")

            # h2: plot freesensingDone over time
            if iExperiment == 1:
                ax6.plot([i * deadline for i in subsampled_idx], np.cumsum(freeSensingDone)[subsampled_idx],
                             color=mycolors[expList.index(iExperiment)], label=label0[expList.index(iExperiment)])
                ax6.set_xlabel("Time [s]")
                ax6.set_ylabel("Free sensing offers")

                # bernd var
                noOfFreeSensingOffersBernd.append(np.cumsum(freeSensingDone))

        # plot global reward over time
        axs2[2].plot([i*deadline for i in range(0, T)], globalReward)
        axs2[2].set_title('global reward over time')
        axs2[2].set_xlabel("seconds")
        if iExperiment == noOfExperiments - 1:
            axs2[2].axhline(y=np.max(globRew, 0), color='r', linestyle='--')

        if showLatex:
            # h: plot global reward over time
            axsh[1].plot([i * deadline for i in subsampled_idx], globalReward[subsampled_idx], marker=mymarkers[expList.index(iExperiment)], markersize=4, color=mycolors[expList.index(iExperiment)])
            axsh[1].set_ylabel(r'$U_t^{SW}$ [monetary units]')
            axsh[1].set_xlabel("Time [s]")
            if iExperiment == 6:
                axsh[1].axhline(y=np.max(globRew, 0), color='r', linestyle='--')
            axsh[1].set_ylim([1.4, globRew + 0.1])

            # h2: plot global reward over time
            if iExperiment == 1: #i * deadline for i in subsampled_idx
                ax7.plot([i*deadline for i in range(0, T)], globalReward,
                         color=mycolors[expList.index(iExperiment)], label=label0[expList.index(iExperiment)])
            else:
                ax7.plot([i*deadline for i in range(0, T)], globalReward,
                             marker=mymarkers[expList.index(iExperiment)], markersize=4,markevery=100,
                             color=mycolors[expList.index(iExperiment)], label=label0[expList.index(iExperiment)])
            ax7.set_ylabel(r'$U_t^{SW}$ [monetary units]')
            ax7.set_xlabel("Time [s]")
            overallRewardBernd.append(globalReward)
            if iExperiment == 6:
                ax7.axhline(y=np.max(globRew, 0), color='r', linestyle='--', label="Optimum")
                overallRewardBernd.append(np.max(globRew, 0))
            ax7.set_ylim([0, globRew + 0.1])




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

    print("no of monte carlo iterations:" + str(len(data[25][0])))

    axs[1, 0].legend(["exp " + str(i) for i in range(noOfExperiments)])
    axs[2, 1].legend(["exp " + str(i) for i in range(noOfExperiments)])
    axs[2, 0].legend(["exp " + str(i) for i in range(noOfExperiments)])
    axs2[2].legend(["exp " + str(i) for i in range(noOfExperiments)])
    axs2[1].legend(["exp " + str(i) for i in range(noOfExperiments)])
    axs2[0].legend(["exp " + str(i) for i in range(noOfExperiments)])

    if showLatex:



        # --------------------  Bernd:
        #------------------------------
        ax5.set_xticks(xticks, minor=False)
        ax6.set_xticks(xticks, minor=False)
        ax7.set_xticks(xticks, minor=False)
        ax8.set_xticks(xticks, minor=False)
        #plt5.xlim(xlimits)
        #plt5.yticks(yticks)
        #plt5.ylim(ylimits)
        #plt5.grid(linestyle='--')
        ax5.grid(linestyle='--')
        ax6.grid(linestyle='--')
        ax7.grid(linestyle='--')
        ax8.grid(linestyle='--')

        ax5.legend(["CA-MAB-FS", "CA-MAB","Greedy","Random",  "Galey-Shapley"])
        ax6.legend(["CA-MAB-FS"])
        ax7.legend(["CA-MAB-FS", "CA-MAB","Greedy","Random", "Optimum"])
        ax8.legend(["CA-MAB-FS", "CA-MAB","Greedy", "Random"  ])

        #ax5.legend(loc='upper left', handletextpad=0.4, ncol=2, columnspacing=0.4, bbox_to_anchor=(-0.305, 1.62))
        #ax6.legend(loc='upper left', handletextpad=0.4, ncol=2, columnspacing=0.4, bbox_to_anchor=(-0.23, 1.38))
        #ax7.legend(loc='upper left', handletextpad=0.4, ncol=2, columnspacing=0.4, bbox_to_anchor=(-0.23, 1.62))
        #ax8.legend(loc='upper left', handletextpad=0.4, ncol=2, columnspacing=0.4, bbox_to_anchor=(-0.23, 1.48))
        ax5.legend(loc='upper left', handletextpad=0.4, ncol=2, columnspacing=0.4, bbox_to_anchor=(-0.03, 1.52))
        ax6.legend(loc='upper left', handletextpad=0.4, ncol=2, columnspacing=0.4, bbox_to_anchor=(-0.03, 1.28))
        ax7.legend(loc='upper left', handletextpad=0.4, ncol=2, columnspacing=0.4, bbox_to_anchor=(-0.03, 1.52))
        ax8.legend(loc='upper left', handletextpad=0.4, ncol=2, columnspacing=0.4, bbox_to_anchor=(-0.03, 1.38))
        ax8.minorticks_on()
        ax8.tick_params(axis='x', which='minor', direction='out')
        ax7.minorticks_on()
        ax7.tick_params(axis='x', which='minor', direction='out')
        ax6.minorticks_on()
        ax6.tick_params(axis='x', which='minor', direction='out')
        ax5.minorticks_on()
        ax5.tick_params(axis='x', which='minor', direction='out')
    # end bernd----------------------
    #--------------------------------


        #figh.set_size_inches(width, height)
        figh.subplots_adjust(wspace=0.479, bottom=0.176)
        figh.savefig('plot.pdf')

    for i in range(len(maxCumSumRegretEnd_MCSP)):
        print("EXP" + str(i) + ": " +  "-- regret MUs: " + str(maxCumSumRegretEnd[i]) + " -- regret mcsp: " + str(maxCumSumRegretEnd_MCSP[i]))

    #plt.tight_layout()
    if showLatex:
        fig5.set_size_inches(widthFig, heightFig)
        fig6.set_size_inches(widthFig, heightFig)
        fig7.set_size_inches(widthFig, heightFig)
        fig8.set_size_inches(widthFig, heightFig)
        fig5.savefig('plot1.pdf')
        fig6.savefig('plot2.pdf')
        fig7.savefig('plot3.pdf')
        fig8.savefig('plot4.pdf')

    # bernd speichern:
    paperVarsFileName = "paperVars" + ".pkl"
    dataExpected = [cumRegretBernd, overallRewardBernd, stabilityBernd, noOfFreeSensingOffersBernd]
    with open(paperVarsFileName, 'wb') as f:
        pickle.dump(dataExpected, f)

    plt.show()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/