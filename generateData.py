import numpy as np
import pickle
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

from shapely.geometry import box
import matplotlib.patches as patches
import random
from shapely.geometry import Point
import matplotlib.pyplot as plt
from copy import copy
import geopandas as gpd
import pandas as pd
import numpy as np
import haversine as hs
from haversine import Unit
from shapely.geometry import Polygon


def vis_random_location_process(num_pt, geo_df):
    """
    Generate num_pt random location coordinates .
    :param num_pt INT number of random location coordinates
    :param geo_df geopandas.geodataframe.GeoDataFrame contains geo data
    """
    polygon = geo_df["geometry"]

    # define boundaries
    bounds_all = polygon.bounds
    minx = min(bounds_all.minx)
    maxx = max(bounds_all.maxx)
    miny = min(bounds_all.miny)
    maxy = max(bounds_all.maxy)

    i = 0
    x_1 = []
    y_1 = []
    x_2 = []
    y_2 = []
    while i < num_pt:
        # generate random location coordinates
        x_t = np.random.uniform(minx, maxx)
        y_t = np.random.uniform(miny, maxy)
        x_1.append(x_t)
        y_1.append(y_t)
        # further check whether it is in the city area
        for p in polygon:
            if Point(x_t, y_t).within(p):
                x_2.append(x_t)
                y_2.append(y_t)
                del x_1[-1]
                del y_1[-1]
                i = i + 1
                break

    rect1 = patches.Rectangle((minx, miny), maxx - minx, maxy - miny, linewidth=1,
                              edgecolor='r', facecolor='none')
    rect2 = copy(rect1)
    rect3 = copy(rect1)
    rect4 = copy(rect1)

    #ax1 = geo_df.plot()
    #ax2 = geo_df.plot()
    #ax3 = geo_df.plot()
    ax4 = geo_df.plot()
    #ax1.add_patch(rect1)
    #ax2.add_patch(rect2)
    #ax3.add_patch(rect3)
    ax4.add_patch(rect4)
    #ax2.plot(x_1, y_1, "ro")
    #ax2.plot(x_2, y_2, "ro")
    #ax3.plot(x_1, y_1, "bo")
    #ax3.plot(x_2, y_2, "ro")
    ax4.plot(x_2, y_2, "ro")
    #ax1.set_title("Step 1: Define boundaries")
    #ax2.set_title("Step 2: Generate rondom locations")
    #ax3.set_title("Step 3: Validate locations")
    ax4.set_title("MAP")
    plt.tight_layout()
    #ax1.figure.savefig("step1.png", dpi=150)
    #ax2.figure.savefig("step2.png", dpi=150)
    #ax3.figure.savefig("step3.png", dpi=150)
    ax4.figure.savefig("map.png", dpi=150)
    plt.show()

    return list(zip(x_2, y_2))

    #hs.haversine((x_2[0], y_2[0]), (x_2[1], y_2[1]), unit=Unit.METERS)

plz_shape_df = gpd.read_file('plz-5stellig.shp', dtype={'plz': str})
plz_region_df = pd.read_csv('plz-5stellig-daten.csv', sep=',', dtype={'plz': str})
# plz_region_df.drop('osm_id', axis=1, inplace=True)
germany_df = pd.merge(left=plz_shape_df, right=plz_region_df, on='plz', how='inner')
düsseldorf_df = germany_df.query('plz == "80636"')
düsseldorf_geo = düsseldorf_df["geometry"]



customName = "random"

noOfTasks = 10
noOfUsers = 15
beta = 1 # -->oo : globaliy ranked arms
x_i = np.random.uniform(noOfTasks)
epsilon_i_k = np.random.logistic(0,1,size=[noOfUsers,noOfTasks])
mus = beta*x_i + epsilon_i_k
mu = np.zeros([noOfUsers, noOfTasks])
mu[0] = np.arange(0,noOfTasks)
#for user in range(0, noOfUsers):
#    for task in range(0, noOfTasks):
#        mu[user][task] = np.sum(mus[user][:] <= mus[user][task])
for user in range(1,noOfUsers):
    mu[user] = np.roll(mu[user-1],1)

#mu = np.ones([noOfUsers, noOfTasks])#np.array([[1,0],[0,1]])

#task_duration = mu + 1 + np.round(np.random.random([noOfUsers, noOfTasks]),5)
#task_duration[:, 2] = np.array([4.5, 4.6, 4.2, 4.4, 4])
#task_duration[4, :] = np.array([0.3, 0.1, 4, 0.5, 0.2])

#mcsp_utility = np.copy(task_duration)
#np.random.shuffle(mcsp_utility.flat)
#mcsp_utility = mcsp_utility + np.round(np.random.random([noOfUsers, noOfTasks]),5)

upload_speed = [20,25,30]#[20,30,50,40.99,30.1,20.1,19.9]
processing_speed = [50,60,55]#[12,10,2,8.1,8.9,9,9.1,12.1]
task_sizes = [1,3,7]# [1,5,3,2,4]

revenuePerMbit = 0.1 # revenue fopr mcsp: €/Mbit for mcsp
costPerSecond = 0.01 # cost for users:   €/sec for users

# for tests:
maxMarge = 0.1
deadline = 0.8
pMoreThan = 0.8

# get distances
locationTupples = vis_random_location_process(noOfUsers+noOfTasks, düsseldorf_df)
# first locations are the tasks
locationOfTasks = locationTupples[:noOfTasks]
locationOfUsers = locationTupples[noOfTasks:]
distances = np.zeros([noOfUsers,noOfTasks])
for i in range(noOfUsers):
    for j in range(noOfTasks):
        distances[i,j] = hs.haversine(locationOfUsers[i], locationOfTasks[j], unit=Unit.KILOMETERS)
alpha_k = np.random.choice(task_sizes, noOfTasks)
mcsp_utility_without_revenue = 1/(1+distances)*alpha_k
beta_i = np.random.choice(processing_speed, noOfUsers)
t_processing = (1/np.reshape(beta_i, [-1, 1])) * (np.ones([noOfUsers, noOfTasks]) + 1*np.random.random([noOfUsers, noOfTasks])) * alpha_k
gamma_i = np.random.choice(upload_speed, noOfUsers)
t_upload = (1/np.reshape(gamma_i, [-1, 1])) * (np.ones([noOfUsers, noOfTasks]) + 1*np.random.random([noOfUsers, noOfTasks])) * alpha_k

#np.random.shuffle(mcsp_utility_without_revenue)

#task_duration = np.array([[0.6,0.1], [0.9,0.3]])#np.array([[3,2], [5,4]])# np.array([[0.6,0.5], [0.9,0.8]])


# test for strict ordering
for i in range(0, noOfTasks):
    _, counts = np.unique(t_processing[:,i] + t_upload[:,i] + mcsp_utility_without_revenue[:,i]*revenuePerMbit, return_counts=True)
    if not(counts == 1).all():
        raise ValueError("no strict ordering of users"  + str(t_processing[:,i] + t_upload[:,i]) + str(mcsp_utility_without_revenue*revenuePerMbit))

# test for max value
if pMoreThan > np.mean(np.mean(mcsp_utility_without_revenue*revenuePerMbit > maxMarge*(t_processing + t_upload)*costPerSecond)):
    raise ValueError("system not profitable" + str(maxMarge*(t_processing + t_upload)*costPerSecond) + str(mcsp_utility_without_revenue*revenuePerMbit))
if pMoreThan > np.mean(np.mean(deadline > (t_processing + t_upload))):
    raise ValueError("deadline too small" + str((t_processing + t_upload))  + " Deadline: " +str(deadline) + " pUnderDeadline: " + str(np.mean(np.mean(deadline > (t_processing + t_upload)))))

print("task duration (processing+upload):")
print(t_processing + t_upload)
print("mcsp utility without prices and revenue:")
print(mcsp_utility_without_revenue)
print("pUnderDeadline:")
print(np.mean(np.mean(deadline > (t_processing + t_upload))))
#task_duration = np.arange(noOfUsers) + 6*np.diag(np.ones([noOfUsers]))# + np.random.rand(noOfUsers,noOfTasks)#np.random.rand(noOfUsers,noOfTasks) + #10*np.random.rand(noOfUsers,noOfTasks)##np.random.rand(noOfUsers,noOfTasks)#100*np.diag(np.ones([noOfUsers]))#
#task_duration -= np.diag(np.arange(noOfUsers))
#estimated_task_duration = np.zeros([noOfUsers,noOfTasks])# + np.random.rand(noOfUsers,noOfTasks)

pickelFileName = "data/" + str(noOfTasks) + str(noOfUsers) + str(customName)
data = [t_processing, t_upload, mcsp_utility_without_revenue]
# Saving the objects:
with open(pickelFileName + ".pkl", 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(data, f)

