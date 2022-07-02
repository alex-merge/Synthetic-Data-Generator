# -*- coding: utf-8 -*-
"""
SDG (Synthetic Data Generator) class allows you to create data to try your 
scripts.

@author: Alex-932
@version : 0.2
"""

import numpy as np
import pandas as pd
# from random import randrange
import matplotlib.pyplot as plt

class SDG():
    
    def __init__(self):
        "SDG allows you to create synthetic data through several methods."
        self.data = pd.DataFrame(columns = ["x", "y", "z", "tp", "track", 
                                            "objID"], dtype = "float")
        self.objectID = 0
        self.track = 0
        self.objectTable = pd.DataFrame(columns = ["tp", "type", 
                                                   "origin", "radius"],
                                        dtype = "object")
        self.version = "0.2"
        
    def toCartesian(radius, azimuth, elevation, origin = None):
        """
        Return the cartesian coordinates of a point that is in spherical. 

        Parameters
        ----------
        radius : float
            Distance between the point and the center of the sphere.
        azimuth : float
            Angle between the point and the x axis.
        elevation : float
            Angle between the point and the z axis.

        Returns
        -------
        tuple
            x, y, z coordinates.

        """
        if origin == None:
            return (radius*(np.sin(elevation))*(np.cos(azimuth)),
                    radius*(np.sin(elevation))*(np.sin(azimuth)),
                    radius*(np.cos(elevation)))
        else :
            return (radius*(np.sin(elevation))*(np.cos(azimuth))+origin[0],
                    radius*(np.sin(elevation))*(np.sin(azimuth))+origin[1],
                    radius*(np.cos(elevation))+origin[2])
    
    def toSpherical(x, y, z, origin = None):
        """
        Return the spherical coordinates of a point that is in cartesian.

        Parameters
        ----------
        x : float
            Distance on the x axis.
        y : float
            Distance on the y axis.
        z : float
            Distance on the z axis.

        Returns
        -------
        tuple
            radius, azimuth, elevation coordinates.

        """
        if origin != None :
            x -= origin[0]
            y -= origin[1]
            z -= origin[2]
            
        return (np.sqrt(x**2+y**2+z**2),
                np.arctan2(y, x),
                np.arccos(z/(np.sqrt(x**2+y**2+z**2))))
        
    def addSphere(self, tp = 0, origin = [0, 0, 0], radius = 10, track = None, 
                  samples = 1000):
        
        if track == None:
            track = self.track
            
        # Adding the object info in the table.
        self.objectTable.loc[self.objectID] = [tp, "Sphere", 
                                               origin, radius]
        points = []
        
        # Golden angle in radians
        phi = np.pi * (3 - np.sqrt(5))
    
        for i in range(samples):
            y = 1-(i/float(samples-1))*2
            rho = np.sqrt(1-y*y)
            
            # Golden angle increment
            theta = phi*i
    
            x = np.cos(theta)*rho
            z = np.sin(theta)*rho
            
            points.append([x, y, z, tp, track, self.objectID])
            track += 1
            
        points = pd.DataFrame(points, columns = ["x", "y", "z", "tp", "track",
                                                 "objID"], 
                                      dtype = "float")
        
        # Shifting the coordinates to get the center on the origin position.
        points["x"] = (points["x"]+origin[0])*radius
        points["y"] = (points["y"]+origin[1])*radius
        points["z"] = (points["z"]+origin[2])*radius
        
        # Saving and updating the different object variables.
        self.data = pd.concat([self.data, points], ignore_index = True)
        self.objectID += 1
        self.track = max([track+1, self.track])
        
    def rotatingSphere(self, origin = [0, 0, 0], radius = 10, nframe = 40, 
                       sample = 1000, tp = 0, azimuth = 15, elevation = 0):
        
        track = self.track
        self.addSphere(tp, origin, radius, track, sample)
        
        for frame in range(1, nframe):
            objID = self.objectID
            self.addSphere(frame+tp, origin, radius, track, sample)
            pointsID = self.data[self.data["objID"] == objID].index
            for point in pointsID:
                cvalues = self.data.loc[point]
                svalues = list(SDG.toSpherical(cvalues["x"], cvalues["y"], 
                                cvalues["z"], origin))
                svalues[1] += frame*azimuth
                svalues[2] += frame*elevation
                self.data.loc[point, ["x", "y", "z"]] = list(SDG.toCartesian(
                    svalues[0], svalues[1], svalues[2], origin))
        
    def showData(self, TP):
        """
        Show the points for the given time point.

        Parameters
        ----------
        TP : int
            Time point.

        """
        if TP == "all" :
            for tp in self.data["tp"].value_counts().index.sort_values() :
                self.showData(tp)
            return None
        data = self.data[self.data["tp"] == TP]
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(data["x"], data["y"], data["z"])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        plt.close()
        
    def exportCSV(self, savepath, OAT = True):
        pass
        if OAT :
            self.data.to_csv
        
if __name__ == "__main__":
    t = SDG()
        
# Old unused methods :

# def addSphere(self, TP, origin, radius, noise, resolution, mode = "new",
#               rot_azimuth = None, rot_elevation = None):
#     """
#     Creates a sphere of points centered at the origin coordinates.

#     Parameters
#     ----------
#     TP : int
#         Time point at which the object should be.
#     origin : list
#         Coordinates of the center of the sphere : [x, y, z]
#     radius : float
#         Distance between the center and the point.
#     noise : int
#         Set the interval for the noise : radius +/- a random int between 
#         -noise and +noise.
#     resolution : int
#         Number of point per revolution.

#     """
#     if mode == "new":
#         # Adding the object info in the table.
#         self.objectTable.loc[self.objectID] = [TP, "Sphere", self.track, 
#                                                origin, radius, noise, 
#                                                resolution]
    
#     # Getting the several angles possible.
#     azimuths = np.linspace(0, 2*np.pi, 2*resolution)
#     elevations = np.linspace(0, np.pi, resolution)
    
#     # Computing all the spherical coordinates available.
#     data = pd.DataFrame(columns = ["az", "elev", "rad"], dtype = "float")
#     for az in azimuths:
#         for elev in elevations:
#             data.loc[az+elev] = [az, elev, 
#                                  radius+randrange(-noise, noise+1)]
            
#     if mode == "rotate":
#         # Adding the angle values to the existing ones
#         if type(rot_azimuth) in [float, int]:
#             data["az"] += rot_azimuth
#         if type(rot_elevation) in [float, int]:
#             data["elev"] += rot_elevation
    
#     # Converting the coordinates into cartesian.
#     cartesian = pd.DataFrame(columns = ["x", "y", "z", "tp"], 
#                              dtype = "float")
#     for i in range(data.shape[0]):
#         values = data.iloc[i]
#         cartesian.loc[i] = list(SDG.toCartesian(values["rad"],
#                                             values["az"],
#                                             values["elev"]))+[TP]
        
#     # Offsetting the values by the center of the sphere coordinates.
#     cartesian["x"] += origin[0]
#     cartesian["y"] += origin[1]
#     cartesian["z"] += origin[2]
    
#     if mode == "new":
#         # Adding the points coordinates to the main dataframe.
#         cartesian["track"] = cartesian.shape[0]*[self.track]
#         self.data = pd.concat([self.data, cartesian], ignore_index = True)
#         self.objectID += 1
#         self.track += 1
#     if mode == "rotate":
#         return cartesian        
        