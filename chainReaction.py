import numpy as np


def getHighestImpact(users_of_tasks_pessimal_stablematch_e, taskDuration):
    takeFailureIntoLoss = False
    # build the tree
    highestDamage = 0
    taskWithHighestDamage = np.nan
    for iTask in range(np.size(taskDuration,1)):
        # user looses the task
        users_of_tasks_pessimal_stablematch = np.copy(users_of_tasks_pessimal_stablematch_e)
        userWhichLoosesTask = users_of_tasks_pessimal_stablematch[iTask]
        if takeFailureIntoLoss:
            damage = taskDuration[userWhichLoosesTask,iTask]
        else:
            damage = 0
        # to which task will the user travel in the next step?
        np.delete(users_of_tasks_pessimal_stablematch, iTask)
        damage = chainReaction(userWhichLoosesTask, damage, users_of_tasks_pessimal_stablematch, taskDuration)
        if damage > highestDamage:
            highestDamage = damage
            taskWithHighestDamage = iTask

    return highestDamage, taskWithHighestDamage


def chainReaction(user, damage, users_of_tasks_pessimal_stablematch, taskDuration_n):
    taskDuration = np.copy(taskDuration_n)
    # find new task for user
    task_durations_user = taskDuration[user,:]
    for iTaskToCheck in np.nditer(np.argsort(np.multiply(-1,task_durations_user))):
        # get duration of task to steal
        durationOfUser = taskDuration[user,iTaskToCheck]
        userToSteal = users_of_tasks_pessimal_stablematch[iTaskToCheck]
        durationOfUserWeWantToSteal = taskDuration[userToSteal,iTaskToCheck]
        # find first task which would loose
        if durationOfUserWeWantToSteal > durationOfUser:
            # steal this task from the user
            userFromWhomIsStolen = userToSteal
            damage += durationOfUserWeWantToSteal
            # remove the task
            #np.delete(taskDuration, iTaskToCheck, 1)
            # update user task match
            users_of_tasks_pessimal_stablematch[iTaskToCheck] = user
            # continue with chain reaction
            damage = chainReaction(userFromWhomIsStolen, damage, users_of_tasks_pessimal_stablematch, taskDuration)
            return damage
    # no more stealing possible
    return damage

chainReaction(0, 0, np.array([2,1]), np.array([[1,2],[1.1,2.2],[1.4,1.9]]))
print(getHighestImpact(np.array([2,1,0]), np.array([[1,2,3],[1.1,2.2,3.3],[1.4,1.9,3.6]])))