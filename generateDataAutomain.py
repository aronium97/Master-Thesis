import pickle
import numpy as np
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
import tqdm

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

    return list(zip(x_2, y_2))

    #hs.haversine((x_2[0], y_2[0]), (x_2[1], y_2[1]), unit=Unit.METERS)

plz_shape_df = gpd.read_file('plz-5stellig.shp', dtype={'plz': str})
plz_region_df = pd.read_csv('plz-5stellig-daten.csv', sep=',', dtype={'plz': str})
# plz_region_df.drop('osm_id', axis=1, inplace=True)
germany_df = pd.merge(left=plz_shape_df, right=plz_region_df, on='plz', how='inner')
düsseldorf_df = germany_df.query('plz == "80636"')
düsseldorf_geo = düsseldorf_df["geometry"]

noOfInstances = 1

customName = "random"

noOfTasks = 10
noOfUsers = 120


upload_speed = [80,85,90]#[20,30,50,40.99,30.1,20.1,19.9]
processing_speed = [50,60,55]#[12,10,2,8.1,8.9,9,9.1,12.1]
task_sizes = [0.2,6,10]# [1,5,3,2,4]

revenuePerMbit = 0.09 # revenue fopr mcsp: €/Mbit for mcsp
costPerSecond = 0.05 # cost for users:   €/sec for users

# for tests:
maxMarge = 0.1
deadline = 0.8
pMoreThan = 0.8

data = []
iInstance = 0
while iInstance < noOfInstances:

    # get distances
    locationTupples = vis_random_location_process(noOfUsers + noOfTasks, düsseldorf_df)
    # first locations are the tasks
    locationOfTasks = locationTupples[:noOfTasks]
    locationOfUsers = locationTupples[noOfTasks:]
    distances = np.ones([noOfUsers, noOfTasks])  # + 0.5*np.random.random([noOfUsers, noOfTasks])
    #for i in range(noOfUsers):
    #    for j in range(noOfTasks):
    #        distances[i, j] += 0.1 * hs.haversine(locationOfUsers[i], locationOfTasks[j], unit=Unit.KILOMETERS)
    alpha_k = np.random.choice(task_sizes, noOfTasks)
    mcsp_utility_without_revenue = 1/(1+distances)*alpha_k
    beta_i = np.random.choice(processing_speed, noOfUsers)
    t_processing = (1/np.reshape(beta_i, [-1, 1])) * (np.ones([noOfUsers, noOfTasks]) +  0.001*np.random.random([noOfUsers, noOfTasks])) * alpha_k
    gamma_i = np.random.uniform(low=80, high=90, size=noOfUsers)#np.random.choice(upload_speed, noOfUsers)
    t_upload = (1/np.reshape(gamma_i, [-1, 1])) * (np.ones([noOfUsers, noOfTasks]) + 0.001*np.random.random([noOfUsers, noOfTasks])) * alpha_k

    # test for strict ordering
    for i in range(0, noOfTasks):
        _, counts = np.unique(t_processing[:,i] + t_upload[:,i] + mcsp_utility_without_revenue[:,i]*revenuePerMbit, return_counts=True)
        if not(counts == 1).all():
            raise ValueError("no strict ordering of users"  + str(t_processing[:,i] + t_upload[:,i]) + str(mcsp_utility_without_revenue*revenuePerMbit))

    # test for max value
    if (pMoreThan > np.mean(np.mean(deadline > (t_processing + t_upload)))) or (pMoreThan > np.mean(np.mean(mcsp_utility_without_revenue*revenuePerMbit > (1+maxMarge)*(t_processing + t_upload)*costPerSecond))):
        Warning("system not profitable" + str(maxMarge*(t_processing + t_upload)*costPerSecond) + str(mcsp_utility_without_revenue*revenuePerMbit))
        Warning("deadline too small" + str((t_processing + t_upload))  + " Deadline: " +str(deadline) + " pUnderDeadline: " + str(np.mean(np.mean(deadline > (t_processing + t_upload)))))
    else:
        print("task duration (processing+upload):")
        print(t_processing + t_upload)
        print("task cost :")
        print((1+maxMarge)*(t_processing + t_upload)*costPerSecond)
        print("mcsp utility without prices and revenue:")
        print(mcsp_utility_without_revenue)
        print("pUnderDeadline:")
        print(np.mean(np.mean(deadline > (t_processing + t_upload))))
        print("pProfitable:")
        print(np.mean(np.mean(mcsp_utility_without_revenue*revenuePerMbit > (1+maxMarge)*(t_processing + t_upload)*costPerSecond)))

        data.append([t_processing, t_upload, mcsp_utility_without_revenue])
        iInstance += 1

pickelFileName = "data/" + str(noOfTasks) + str(noOfUsers) + "_auto_" + str(customName)
# Saving the objects:
with open(pickelFileName + ".pkl", 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(data, f)

