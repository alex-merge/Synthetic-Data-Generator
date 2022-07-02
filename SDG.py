# -*- coding: utf-8 -*-
"""
SDG (Synthetic Data Generator) class allows you to create data to try your 
scripts.

@author: Alex-932
@version : 0.1
"""

import numpy as np
import pandas as pd
from random import randrange
import matplotlib.pyplot as plt

class SDG():
    
    def __init__(self):
        "SDG allows you to create synthetic data through several methods."
        self.data = pd.DataFrame(columns = ["x", "y", "z"], dtype = "float")
        self.objectID = 0
        self.objectTable = pd.DataFrame(columns = ["tp", "type", "origin", 
                                                   "radius", "noise", 
                                                   "resolution"],
                                        dtype = "object")
        self.version = "0.1"
        
    def toCartesian(radius, azimuth, elevation):
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
        return (radius*(np.sin(elevation))*(np.cos(azimuth)),
                radius*(np.sin(elevation))*(np.sin(azimuth)),
                radius*(np.cos(elevation)))
    
    def toSpherical(x, y, z):
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
        return (np.sqrt(x**2+y**2+z**2),
                np.arctan(y/x),
                np.arccos(z/(np.sqrt(x**2+y**2+z**2))))
        
    def addSphere(self, TP, origin, radius, noise, resolution):
        """
        Creates a sphere of points centered at the origin coordinates.

        Parameters
        ----------
        TP : int
            Time point at which the object should be.
        origin : list
            Coordinates of the center of the sphere : [x, y, z]
        radius : float
            Distance between the center and the point.
        noise : int
            Set the interval for the noise : radius +/- a random int between 
            -noise and +noise.
        resolution : int
            Number of point per revolution.

        """
        # Adding the object info in the table.
        self.objectTable.loc[self.objectID] = [TP, "Sphere", origin, radius, 
                                               noise, resolution]
        self.objectID += 1
        
        # Getting the several angles possible.
        azimuths = np.linspace(0, 2*np.pi, 2*resolution)
        elevations = np.linspace(0, np.pi, resolution)
        
        # Computing all the spherical coordinates available.
        data = pd.DataFrame(columns = ["az", "elev", "rad"], dtype = "float")
        for az in azimuths:
            for elev in elevations:
                data.loc[az+elev] = [az, elev, 
                                     radius+randrange(-noise, noise+1)]
        
        # Converting the coordinates into cartesian.
        cartesian = pd.DataFrame(columns = ["x", "y", "z", "tp"], 
                                 dtype = "float")
        for i in range(data.shape[0]):
            values = data.iloc[i]
            cartesian.loc[i] = list(SDG.toCartesian(values["rad"],
                                                values["az"],
                                                values["elev"]))+[TP]
            
        # Offsetting the values by the center of the sphere coordinates.
        cartesian["x"] += origin[0]
        cartesian["y"] += origin[1]
        cartesian["z"] += origin[2]
        
        # Adding the points coordinates to the main dataframe.
        self.data = pd.concat([self.data, cartesian], ignore_index = True)
        
    def showData(self, TP):
        """
        Show the points for the given time point.

        Parameters
        ----------
        TP : int
            Time point.

        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(self.data["x"], self.data["y"], self.data["z"])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        plt.close()
        
if __name__ == "__main__":
    t = SDG()
        
        
        