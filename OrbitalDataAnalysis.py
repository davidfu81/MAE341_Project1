from skyfield import api
from skyfield.api import wgs84
from skyfield.api import Distance
from skyfield.functions import length_of
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
ts = api.load.timescale()

#Constants
R_E = 6371000 #Radius of the Earth is 6371 km
G = 6.6743E-11 # Gravitational Constant
M_E = 5.972E24 #Mass of the Earth
MU = G*M_E

#Load Data
data = pd.read_csv("data.csv")
data.loc[:,"Altitude"] = data.loc[:,"Altitude"]*1.609*1000 #Convert to m
data.loc[:,"Distance"] = data.loc[:,"Distance"]*1.609*1000
data.loc[:,"Velocity"] = data.loc[:,"Velocity"]*1.609*1000
converted = data.copy()
converted = converted.rename(columns = {"Azimuth":"Declination", "Elevation":"Right Ascension"})

def convert_AltAz(year, month, day, hour, minute, second, azimuth, elevation, distance):
    t = ts.utc(year, month, day, hour, minute, second)                   # YOUR TIME OF OBSERVATION
    loc = wgs84.latlon(+40.342697, -74.652930)  # YOUR LOCATION

    alt_degrees = elevation                                    # YOUR MEASUREMENTS (ELEVATION)
    az_degrees = azimuth                                   # YOUR MEASUREMENTS (AZIMUTH)

    d = Distance(km=distance/1000)                                # DISTANCE TO OBJECT (NOT ALTITUDE)

    satellite = loc.at(t).from_altaz(alt_degrees=alt_degrees, az_degrees=az_degrees, distance=d)

    icrf_coordinates = satellite.position.km
    m1 = loc.at(t)

    m2 = m1.position.km+icrf_coordinates
    r = length_of(m2)

    l = m2[0] / r
    m = m2[1] / r
    n = m2[2] / r

    declination = math.degrees(math.asin(n))

    if m > 0:
        RA = math.degrees(math.acos(l / math.cos(math.radians(declination))))
    else:
        RA = 360 - math.degrees(math.acos(l / math.cos(math.radians(declination))))
    
    return declination, RA

#Helper Function to get Cartesian coordinates from orbital parameters
#Input: distance from focus, longitude of ascension, argument of periapse, true anomaly, inclination
#Output: array with dimensions (length(theta),3) where each row is a vector with [x,y,z] coordinates
def cartesian_from_Parameter(r, longitude_ascension, argument_periapse, theta, inclination):
    pos = np.zeros((np.shape(theta)[0], 3))
    pos[:,0] = r*(np.cos(longitude_ascension)*np.cos(argument_periapse+theta)-np.sin(longitude_ascension)*np.sin(argument_periapse+theta)*np.cos(inclination))
    pos[:,1] = r*(np.sin(longitude_ascension)*np.cos(argument_periapse+theta)+np.cos(longitude_ascension)*np.sin(argument_periapse+theta)*np.cos(inclination))
    pos[:,2] = r*np.sin(argument_periapse+theta)*np.sin(inclination)
    return pos
####################################################################################################################################################################
#########################################
##### Position and Velocity Vectors #####
#########################################

#Convert Azimuth and Elevation to Declination and Right Ascension
for index, row in data.iterrows():
    declination, RA = convert_AltAz(row["Year"], row["Month"], row["Day"], row["Hour"], row["Minute"], row["Second"]
                                    ,row["Azimuth"], row["Elevation"], row["Distance"])
    converted.loc[index, "Declination"] = declination
    converted.loc[index, "Right Ascension"] = RA

#Get Cartesian coordinates from "spherical" coordinates
position = np.zeros((converted.shape[0],3))
converted["X"] = 0
converted["Y"] = 0
converted["Z"] = 0 
converted.loc[:,"Altitude"] = converted.loc[:,"Altitude"] + R_E
converted = converted.rename(columns = {"Altitude":"R"})
converted.loc[:,"Z"] = converted.loc[:,"R"]*np.sin(converted.loc[:,"Declination"]*np.pi/180)
converted.loc[:,"X"] = converted.loc[:,"R"]*np.cos(converted.loc[:,"Declination"]*np.pi/180)*np.cos(converted.loc[:,"Right Ascension"]*np.pi/180)
converted.loc[:,"Y"] = converted.loc[:,"R"]*np.cos(converted.loc[:,"Declination"]*np.pi/180)*np.sin(converted.loc[:,"Right Ascension"]*np.pi/180)
position[:, 0] = converted.loc[:, "X"]
position[:, 1] = converted.loc[:, "Y"]
position[:, 2] = converted.loc[:, "Z"]
r = np.reshape(np.linalg.norm(position, axis = 1), (-1, 1))

#Velocity Vector - Note row will not have velocity vector as it is computed based on next position coordinate
velocity = np.zeros((converted.shape[0]-1, 3))
converted["Vx"] = 0
converted["Vy"] = 0
converted["Vz"] = 0
for i in range(0,(converted.shape[0]-1)):
    velocity[i,:] = (position[i+1,:]-position[i,:])/10
    converted.loc[i, "Vx"] = (converted.loc[i+1, "X"] - converted.loc[i, "X"])/10
    converted.loc[i, "Vy"] = (converted.loc[i+1, "Y"] - converted.loc[i, "Y"])/10
    converted.loc[i, "Vz"] = (converted.loc[i+1, "Z"] - converted.loc[i, "Z"])/10
    #print(np.sqrt(converted.loc[i, "Vx"]**2 + converted.loc[i, "Vy"]**2 + converted.loc[i, "Vz"]**2)-converted.loc[i, "Velocity"])
    converted.loc[i, "Velocity"] = np.sqrt(converted.loc[i, "Vx"]**2 + converted.loc[i, "Vy"]**2 + converted.loc[i, "Vz"]**2)
converted = converted.drop(converted.shape[0]-1)
speed = np.reshape(np.linalg.norm(velocity, axis = 1), (-1, 1))
position = np.delete(position, -1, axis = 0)
r = np.delete(r, -1, axis = 0)

####################################################################################################################################################################
#########################################
######### Orbital Parameters ############
#########################################
#Compute Semi-Major Axis
semi_major = r/(2-np.multiply(r, np.square(speed)/MU))
a = np.average(semi_major) #SEMI MAJOR AXIS (m)
print("Semi-Major Axis (m): ", a)
T = 2*np.pi*np.sqrt(a**3/MU)/60 #ORBITAL PERIOD (seconds)
print("Orbital Period (min): ", T)

#Eccentricity
e_vec = 1/MU*(np.multiply(np.square(speed)-MU/r, position) - np.multiply(np.reshape(np.diagonal(position @ velocity.T),(-1,1)), velocity))
e_vec = np.average(e_vec, axis = 0)
e = np.linalg.norm(e_vec)
print("Eccentricity: ", e)

#Inclination
h_vec = np.average(np.cross(position, velocity), axis=0)
h = np.linalg.norm(h_vec)
inclination = np.arccos(h_vec[2]/h)
print("Inclination (deg): ", inclination*180/np.pi)

#Longitude of Ascending Node
n_vec = np.cross([0,0,1], h_vec/h)
n = np.linalg.norm(n_vec)

long_ascend = np.arccos(n_vec[0]/n)
if n_vec[1] < 0:
    long_ascend = 2*np.pi - long_ascend
print("Longitude of Ascending Node (deg): ", long_ascend*180/np.pi)


#Argument of Periapse
arg_periapse = np.arccos(np.dot(n_vec, e_vec)/(n*e))
print("Argument of Periapse (deg): ", arg_periapse*180/np.pi)

####################################################################################################################################################################
#########################################
######### Orbit Visualiztion ############
#########################################
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(position[:,0], position[:,1], position[:,2], label="Observations")

# x = np.linspace(-5E6, 5E6, 100)
# y = np.linspace(-5E6, 5E6, 100)
# x, y = np.meshgrid(x, y)
#ax.plot_surface(x,y, (h_vec[0] * x + h_vec[1]*y)/(-h_vec[2]))
#ax.plot_surface(x,y,x*0)
ax.plot([0, 8E6],[0,0],[0,0], color = "green", label="Reference Line")
ax.plot([0, e_vec[0]*8E6/e], [0, e_vec[1]*8E6/e], [0, e_vec[2]*8E6/e], color = 'red', label="Perigee")
ax.plot([0, n_vec[0]*8E6/n], [0, n_vec[1]*8E6/n], [0, n_vec[2]*8E6/n], color = 'purple', label="Line of Nodes")

#Plot Orbital Trajectory
theta_traj = np.linspace(0, 2*np.pi, 360)
r_traj = a*(1-e**2)/(1+e*np.cos(theta_traj))
pos_traj = cartesian_from_Parameter(r_traj, long_ascend, arg_periapse, theta_traj, inclination)
ax.plot(pos_traj[:,0], pos_traj[:,1], pos_traj[:,2], color="orange", label="Calculated Trajectory")

#Plot True Trajectory
r_traj_true = 6753E3*(1-0.001008**2)/(1+0.001008*np.cos(theta_traj))
pos_traj_true = cartesian_from_Parameter(r_traj_true, np.radians(13.72*15), np.radians(191.13), theta_traj, np.radians(41.47))
ax.plot(pos_traj_true[:,0], pos_traj_true[:,1], pos_traj_true[:,2], color="cyan", label="True Trajectory")

#Plot Earth
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = R_E*np.cos(u)*np.sin(v)
y = R_E*np.sin(u)*np.sin(v)
z = R_E*np.cos(v)
ax.plot_wireframe(x, y, z, color="brown", label="Earth")
ax.legend(loc=0)



#Anomalies
converted["True Anomaly"] = 0
converted["Eccentric Anomaly"] = 0
converted["Mean Anomaly"] = 0
converted["Hours Past Perigee"] = 0
for i in range(0, converted.shape[0]):
    theta = np.arccos(np.dot(e_vec,position[i,:])/e/r[i])
    epsilon = np.arccos(np.cos(theta)*r[i]/a+e)
    if(np.dot(position[i,:], velocity[i,:])<0):
        theta = 2*np.pi - theta
        epsilon = - epsilon
    Me = epsilon - e*np.sin(epsilon)
    t_past_peri = Me*2*np.pi/T
    converted.loc[i, "True Anomaly"] = np.degrees(theta)
    converted.loc[i, "Eccentric Anomaly"] = np.degrees(epsilon)
    converted.loc[i, "Mean Anomaly"] = Me
    converted.loc[i, "Hours Past Perigee"] = t_past_peri


converted.loc[:, ["Velocity", "R", "X", "Y", "Z", "Vx", "Vy", "Vz"]] = converted.loc[:, ["Velocity", "R", "X", "Y", "Z", "Vx", "Vy", "Vz"]]
#print(converted)
converted.to_csv('converted.csv')

####################################################################################################################################################################
#########################################
### Prediction from Kepler's Equation ###
#########################################

# Helper function to solve Kepler's equation through iteration given a mean anomaly and eccentricity.
# Only has a unique solution if e <= 1
# Returns mean anomaly and eccentric anomaly between 0 and 2pi
def solve_Kepler(mean_anomaly, e):
    error = float('inf')
    while mean_anomaly > 2*np.pi or mean_anomaly < 0:
        mean_anomaly = mean_anomaly - np.sign(mean_anomaly)*2*np.pi

    epsilon_range = np.linspace(0, 2*np.pi, 10)
    
    while error > 0.0001:
        errors = np.abs(mean_anomaly-(epsilon_range-e*np.sin(epsilon_range)))
        min_index = np.argmin(errors)
        error = errors[min_index]
        if min_index > 0 and min_index < 9:
            if errors[min_index-1] < errors[min_index+1]:
                epsilon_range = np.linspace(epsilon_range[min_index-1], epsilon_range[min_index], 10)
            else:
                epsilon_range = np.linspace(epsilon_range[min_index], epsilon_range[min_index+1], 10)
        elif min_index > 0:
            epsilon_range = np.linspace(epsilon_range[min_index-1], epsilon_range[min_index], 10)
        else:
            epsilon_range = np.linspace(epsilon_range[min_index], epsilon_range[min_index+1], 10)

    return mean_anomaly, epsilon_range[min_index]

### Prediction time: 20:00:00PM - 20:00:50 on 10/8/2023
prediction = pd.DataFrame()
prediction["Year"]= 2023*np.ones(6)
prediction["Month"] = 10*np.ones(6)
prediction["Day"] = 8*np.ones(6)
prediction["Hour"] = 20*np.ones(6)
prediction["Minute"] = 3*np.ones(6)
prediction["Second"] = np.linspace(0, 50, 6)
deltaT = prediction.loc[:,"Second"] + (prediction.loc[:,"Day"]-5)*3600*24 + (prediction.loc[:,"Hour"]-19)*3600+(prediction.loc[:,"Minute"]-57)*60
mean_anomaly = 2*np.pi*deltaT/(92.073*60) - converted.loc[converted.shape[0]-1, "Mean Anomaly"]
eccentric = np.zeros(mean_anomaly.shape[0])
for i in range(0, mean_anomaly.shape[0]):
    mean_anomaly[i], eccentric[i] = solve_Kepler(mean_anomaly[i], e)
radius = a*(1-e*np.cos(eccentric))
true = np.arccos(a*(np.cos(eccentric)-e)/radius)
true[eccentric > np.pi] = 2*np.pi - true[eccentric > np.pi]
position_pred = cartesian_from_Parameter(radius, long_ascend, arg_periapse, true, inclination)

prediction["Mean Anomaly"] = mean_anomaly
prediction["Eccentric Anomaly"] = np.degrees(eccentric)
prediction["Radius"] = radius/1000
prediction["True Anomaly"] = np.degrees(true)
prediction["X"] = position_pred[:,0]/1000
prediction["Y"] = position_pred[:,1]/1000
prediction["Z"] = position_pred[:,2]/1000
prediction.to_csv("prediction.csv")

#Actual Data
actual = pd.read_csv("verification.csv")
actual.loc[:,"Altitude"] = actual.loc[:,"Altitude"]*1.609*1000 #Convert to m
actual.loc[:,"Distance"] = actual.loc[:,"Distance"]*1.609*1000
actual.loc[:,"Velocity"] = actual.loc[:,"Velocity"]*1.609*1000
for index, row in actual.iterrows():
    declination, RA = convert_AltAz(row["Year"], row["Month"], row["Day"], row["Hour"], row["Minute"], row["Second"]
                                    ,row["Azimuth"], row["Elevation"], row["Distance"])
    actual.loc[index, "Azimuth"] = declination
    actual.loc[index, "Elevation"] = RA
actual = actual.rename(columns = {"Azimuth":"Declination", "Elevation":"Right Ascension"})

actual_position = np.zeros((actual.shape[0],3))
actual.loc[:,"Altitude"] = actual.loc[:,"Altitude"] + R_E
actual = actual.rename(columns = {"Altitude":"R"})
actual_position[:,2] = actual.loc[:,"R"]*np.sin(actual.loc[:,"Declination"]*np.pi/180)
actual_position[:,0] = actual.loc[:,"R"]*np.cos(actual.loc[:,"Declination"]*np.pi/180)*np.cos(actual.loc[:,"Right Ascension"]*np.pi/180)
actual_position[:,1] = actual.loc[:,"R"]*np.cos(actual.loc[:,"Declination"]*np.pi/180)*np.sin(actual.loc[:,"Right Ascension"]*np.pi/180)
actual["X"] = actual_position[:,0]/1000
actual["Y"] = actual_position[:,1]/1000
actual["Z"] = actual_position[:,2]/1000
actual.to_csv("actual.csv")

#Plot Prediction
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(position_pred[:,0], position_pred[:,1], position_pred[:,2], label="Predictions")
ax.scatter(actual_position[:,0], actual_position[:, 1], actual_position[:,2], color="purple", label="Actual Positions")
ax.plot(pos_traj[:,0], pos_traj[:,1], pos_traj[:,2], color="orange", label="Calculated Trajectory")


ax.plot([0, 8E6],[0,0],[0,0], color = "green", label="Reference Line")
ax.plot([0, e_vec[0]*8E6/e], [0, e_vec[1]*8E6/e], [0, e_vec[2]*8E6/e], color = 'red', label="Perigee")
ax.plot([0, n_vec[0]*8E6/n], [0, n_vec[1]*8E6/n], [0, n_vec[2]*8E6/n], color = 'purple', label="Line of Nodes")

#Plot Earth
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = R_E*np.cos(u)*np.sin(v)
y = R_E*np.sin(u)*np.sin(v)
z = R_E*np.cos(v)
ax.plot_wireframe(x, y, z, color="brown", label="Earth")
ax.legend(loc=0)

plt.show()