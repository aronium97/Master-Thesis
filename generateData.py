import numpy as np
import pickle

tztz = True
while tztz:
    customName = "random"

    noOfTasks = 5
    noOfUsers = 5

    beta = 1 # -->oo : globaliy ranked arms
    x_i = np.random.uniform(noOfTasks)
    epsilon_i_k = np.random.logistic(0,1,size=[noOfUsers,noOfTasks])
    mus = beta*x_i + epsilon_i_k
    mu = np.zeros([noOfTasks, noOfUsers])
    mu[0] = np.arange(0,noOfTasks)
    #for user in range(0, noOfUsers):
    #    for task in range(0, noOfTasks):
    #        mu[user][task] = np.sum(mus[user][:] <= mus[user][task])
    for user in range(1,noOfUsers):
        mu[user] = np.roll(mu[user-1],1)

    #mu = np.array([[1,0],[0,1]])

    task_duration = mu + 1 #+ np.random.random([np.size(mu,0), np.size(mu,1)])

    for i in range(0, noOfTasks):
        _, counts = np.unique(task_duration[:,i], return_counts=True)
        if not(counts == 1).all():
            #raise ValueError("no strict ordering of users"  + str(task_duration))
            tztz = True
            break
        else:
            tztz = False

print(task_duration)
#task_duration = np.arange(noOfUsers) + 6*np.diag(np.ones([noOfUsers]))# + np.random.rand(noOfUsers,noOfTasks)#np.random.rand(noOfUsers,noOfTasks) + #10*np.random.rand(noOfUsers,noOfTasks)##np.random.rand(noOfUsers,noOfTasks)#100*np.diag(np.ones([noOfUsers]))#
#task_duration -= np.diag(np.arange(noOfUsers))
#estimated_task_duration = np.zeros([noOfUsers,noOfTasks])# + np.random.rand(noOfUsers,noOfTasks)

pickelFileName = "data/" + str(noOfTasks) + str(noOfUsers) + str(customName)

# Saving the objects:
with open(pickelFileName + ".pkl", 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(task_duration, f)

