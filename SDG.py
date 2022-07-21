# -*- coding: utf-8 -*-
"""
SDG (Synthetic Data Generator) class allows you to create data to try your 
scripts.

@author: Alex-932
@version : 0.2.6
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from scipy.ndimage.filters import gaussian_filter

class SDG():
    
    def __init__(self):
        "SDG allows you to create synthetic data through several methods."
        self.data = pd.DataFrame(columns = ["x", "y", "z", "tp", "track", 
                                            "objID", "target", "metaObjID"], 
                                 dtype = "float")
        self.objectID = 0
        self.metaObjID = 0
        self.track = 0
        self.objectTable = pd.DataFrame(columns = ["tp", "type", 
                                                   "origin", "radius"],
                                        dtype = "object")
        self.version = "0.2.5"
        
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
        """
        Add a Fibonacci sphere at the given coordinates.

        Parameters
        ----------
        tp : int, optional
            Time point (or frame) the sphere will be added in. The default is 
            0.
        origin : list, optional
            Coordinates of the center of the sphere. The default is [0, 0, 0].
            Format : [X, Y, Z]
        radius : float, optional
            Radius of the sphere. The default is 10.
        track : int, optional
            Starting track ID. The default is None.
            Each point has its trackID and this parameters set the first one.
            Used with the addRotatingSphere() method.
        samples : int, optional
            Number of points in the sphere. The default is 1000.
            
        Return
        ------
        
        Update self.data with the new points.

        """
        
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
            
            points.append([x, y, z, tp, track, self.objectID, self.metaObjID])
            track += 1
            
        points = pd.DataFrame(points, columns = ["x", "y", "z", "tp", "track",
                                                 "objID", "metaObjID"], 
                                      dtype = "float")
        
        # Shifting the coordinates to get the center on the origin position.
        points["x"] = points["x"]*radius+origin[0]
        points["y"] = points["y"]*radius+origin[1]
        points["z"] = points["z"]*radius+origin[2]
        
        # Saving and updating the different object variables.
        self.data = pd.concat([self.data, points], ignore_index = True)
        self.objectID += 1
        self.metaObjID += 1
        self.track = max([track+1, self.track])
                
    def addRotatingSphere(self, origin = [0, 0, 0], radius = 10, nframe = 20, 
                          sample = 1000, tp = 0, 
                          theta_x = 15, theta_y = 0, theta_z = 0):
        """
        Add a rotating sphere in the volume. The sphere rotate by offsetting 
        the points by {azimuth}° azimuth and {elevation}° elevation between 
        each frames.

        Parameters
        ----------
        origin : list, optional
            Coordinates of the center of the sphere. The default is [0, 0, 0].
            Format : [X, Y, Z]
        radius : float, optional
            Radius of the sphere. The default is 10.
        nframe : int, optional
            Number of frames in which the sphere is. The default is 20.
        sample : int, optional
            Number of points in the sphere. The default is 1000.
        tp : int, optional
            Starting time point. The default is 0.
        theta_x : float, optional
            X angle increment in degree. The default is 15.
        theta_y : float, optional
            Y angle increment in degree. The default is 0.
        theta_z : float, optional
            Z angle increment in degree. The default is 0.

        Returns
        -------
        Update self.data.

        """
        
        # Saving the first track ID and creating the first sphere.
        track = self.track
        metaObjID = self.metaObjID
        self.addSphere(tp, origin, radius, track, sample)
        
        # Converting degree to radian
        theta_x = theta_x*np.pi/180
        theta_y = theta_y*np.pi/180
        theta_z = theta_z*np.pi/180
        
        # Creating as much sphere as asked with nframe.
        for frame in range(1, nframe):
            
            # Temporarily saving the objectID the sphere will have.
            objID = self.objectID
            
            # Getting the IDs of the current sphere's first point and previous
            # sphere's first point.
            prevID = self.data[self.data["objID"] == objID-1].index
            newID = prevID+sample
            
            # Adding the ID of the current sphere's spots to the previous 
            # sphere's spots. That way we establish a link.  
            self.data.loc[prevID, "target"] = list(newID)
            itrack = track
            
            # Iterating through the points of the current sphere.
            for point in prevID:
                
                # Getting the cartesian coordinates.
                cvalues = self.data.loc[point, ["x", "y", "z"]].copy()
                
                # Retrieving the offset from the centroid to [0, 0, 0].
                cvalues["x"] -= origin[0]
                cvalues["y"] -= origin[1]
                cvalues["z"] -= origin[2]
                
                # Applying X rotation
                tempy = cvalues["y"].copy()
                cvalues["y"] = cvalues["y"]*np.cos(theta_x)-\
                    cvalues["z"]*np.sin(theta_x)
                cvalues["z"] = tempy*np.sin(theta_x)+\
                    cvalues["z"]*np.cos(theta_x)
                
                # Applying Y rotation
                tempx = cvalues["x"].copy()
                cvalues["x"] = cvalues["x"]*np.cos(theta_y)+\
                    cvalues["z"]*np.sin(theta_y)
                cvalues["z"] = -tempx*np.sin(theta_y)+\
                    cvalues["z"]*np.cos(theta_y)
                    
                # Applying Z rotation
                tempx = cvalues["x"].copy()
                cvalues["x"] = cvalues["x"]*np.cos(theta_z)-\
                    cvalues["y"]*np.sin(theta_z)
                cvalues["y"] = tempx*np.sin(theta_z)+\
                    cvalues["y"]*np.cos(theta_z)
                    
                # Readding the offset from the centroid to [0, 0, 0].
                cvalues["x"] += origin[0]
                cvalues["y"] += origin[1]
                cvalues["z"] += origin[2]
                
                # Converting them back and saving them in the data dataframe.
                self.data.loc[point+sample] = list(cvalues)+\
                    [frame, itrack, objID, np.nan, metaObjID]
                
                itrack += 1
            
            self.objectID += 1
            
    def addDrift(self, metaObjID, drift = [1, 1, 1], randomized = False, 
                 variation = 10):
        """
        Add drift to a moving object like 

        Parameters
        ----------
        metaObjID : int
            ID of the metaObject (rotating sphere for example).
        drift : list, optional
            Drift values : [dX, dY, dZ]. The default is [1, 1, 1].
        randomized : bool, optional
            If True, Drift values are randomized using variation value. 
            Default is False.
        variation : float, optional
            Range of variation : {variation}*[-1,1]+Value. The default is 10.

        """
        # Subsampling the dataset and getting the objectIDs from it.
        subdata = self.data[self.data["metaObjID"] == metaObjID].copy()
        objectIDs = subdata["objID"].unique().tolist()
        
        # List to save the total displacement.
        driftSum = [0, 0, 0]
        
        # Iterating over objectIDs.
        for objID in objectIDs:
            
            # Creating a temporary list to prevent modification of the drift 
            # values
            tempDrift = drift.copy()
            
            if randomized :
                for i in range(len(drift)):
                    # Adding the variation*[1;-1] to the drift.
                    tempDrift[i] += (variation*np.random.sample()-\
                                     variation*np.random.sample())
                    
                    driftSum[i] += tempDrift[i]
                    
            else :
                for i in range(len(drift)):
                    driftSum[i] += tempDrift[i]
            
            # Adding the sum of the total displacement to all points from a 
            # given object.
            for ID in subdata[subdata["objID"] == objID].index:
                self.data.loc[ID, "x"] += driftSum[0]
                self.data.loc[ID, "y"] += driftSum[1]
                self.data.loc[ID, "z"] += driftSum[2]          
        
    def addBlob(self, tp, origin = [0, 0, 0], radius = 5, nframe = 20, 
                sample = 20):
        pass
            
    def addVariability(self, metaObjID, variation = 1):
        """
        Modify the coordinates of the points belonging to the given object

        Parameters
        ----------
        metaObjID : int
            ID of the object.
        variation : float, optional
            Range of variation : {variation}*[-1,1]. The default is 1.

        """
        IDs = self.data[self.data["metaObjID"] == metaObjID].index
        
        for ID in IDs:
            for axis in ["x", "y", "z"]:
                self.data.loc[ID, axis] += (variation*np.random.sample()-\
                                            variation*np.random.sample())
                    
    def addOrganoid(self, sample = 200, origin = [100, 100, 100], radius = 15, 
                    nframe = 20, tp = 0, rotation = [15, 10, 5],
                    drift = [1, 1, 1], randomizedDrift = True, 
                    driftVariation = 1, variability = 1):
        """
        Create an organoid like object using existing methods.

        Parameters
        ----------
        sample : int, optional
            Number of points in the sphere. The default is 200.
        origin : list, optional
            Coordinates of the center of the sphere. The default is [0, 0, 0].
            Format : [X, Y, Z]
        radius : float, optional
            Radius of the sphere. The default is 15.
        nframe : int, optional
            Number of frames in which the sphere is. The default is 20.
        tp : int, optional
            Starting time point. The default is 0.
        rotation : list of float, optional
            Rotation angle (°) for each axis. The default is [15, 10, 5].
        drift : list, optional
            Drift values : [dX, dY, dZ]. The default is [1, 1, 1].
        randomizedDrift : bool, optional
            If True, Drift values are randomized using variation value. 
            Default is True.
        driftVariation : float, optional
            Range of variation : {variation}*[-1,1]+Value. The default is 1.
        variability : float, optional
            Range of variation : {variation}*[-1,1]. The default is 1.

        """
        # Getting the metaObjectID.
        metaObjID = self.metaObjID
        
        # Adding the rotating sphere.
        self.addRotatingSphere(origin, radius, nframe, sample, tp, rotation[0],
                               rotation[1], rotation[2])
        
        # Adding drift to the sphere.
        self.addDrift(metaObjID, drift, randomizedDrift, driftVariation)
        
        # Adding some noise to the points coordinates.
        self.addVariability(metaObjID, variability)
        
                
    def exportToTifs(self, savepath, gaussianSigma = 1, mode = "single"):
        TP = self.data["tp"].value_counts(ascending = True).index
        
        if mode == "stacked":
            stackedArray = []
        
        for tp in TP:
            array = np.zeros((int(self.data["z"].max()+10),
                              int(self.data["y"].max()+10),
                              int(self.data["x"].max()+10)))
            
            filename = "synth_"+str(int(tp))+".tif"
            
            subdata = self.data[self.data["tp"] == tp].copy()
            
            array[subdata["z"].astype(int),
                  subdata["y"].astype(int), 
                  subdata["x"].astype(int)] = 10
            
            blurredArray = gaussian_filter(array, sigma = gaussianSigma)
            
            if mode == "single":
                tifffile.imwrite(savepath+'\\'+filename, 
                                 blurredArray.astype("float32"), imagej = True)
            
            elif mode == "stacked":
                stackedArray.append(blurredArray)

        if mode == "stacked":
            tifffile.imwrite(savepath+"\\synth_stacked.tif", 
                             np.array(stackedArray).astype("float32"), 
                             imagej = True, 
                             metadata={'axes': 'TZYX'})
        
    def showData(self, TP = "all", tracks = "all", limits = True, lim = None):
        """
        Show the path that the given track(s) are taking in 3D.

        Parameters
        ----------
        TP : float, int or list, optional
            ID of the time points to show. The default is "all". 
        track : float, int or list, optional
            ID of the track(s) to show. The default is "all".
        limits : list, optional
            List of the limits. The default is None.
            Format : [x_min, x_max, y_min, y_max, z_min, z_max]

        """
        # Creating the figure and the axes.
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        # Setting iteration through time points if multiple ones.
        if TP == "all" :
            for tp in self.data["tp"].value_counts().index.sort_values() :
                self.showData(tp, tracks, limits)
            return None
        if type(TP) == list :
            for tp in TP :
                self.showData(tp, tracks, limits)
            return None
        
        # Setting track as a list if only one is wished.
        if type(tracks) in [int, float]:
            tracks = [tracks]
        
        # Setting the limits of the figure.
        if lim is None:
            lim = pd.Series([self.data["x"].min()-5, self.data["x"].max()+5,
                             self.data["y"].min()-5, self.data["y"].max()+5,
                             self.data["z"].min()-5, self.data["z"].max()+5],
                            index = ["xm", "xM", "ym", "yM", "zm", "zM"],
                            dtype = "float")
        else :
            lim = pd.Series(lim, 
                            index = ["xm", "xM", "ym", "yM", "zm", "zM"],
                            dtype = "float")
        
        # Extracting the points at the given timepoints.
        data = self.data[self.data["tp"] == TP]
        
        # Keeping the wanted tracks.
        if tracks != "all":
            data = data[data["track"].isin(tracks)]
        
        # Scatter plotting the data.
        ax.scatter(data["x"], data["y"], data["z"])
        
        # Setting axis labels.
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        if limits :
            # Setting the axis' limits. 
            ax.set_xlim3d([lim["xm"], lim["xM"]])
            ax.set_ylim3d([lim["ym"], lim["yM"]])
            ax.set_zlim3d([lim["zm"], lim["zM"]])
        
        plt.show()
        plt.close()
        
    def showPath(self, track = "all", limits = None):
        """
        Show the path that the given track(s) are taking in 3D.

        Parameters
        ----------
        track : float, int or list, optional
            ID of the track(s) to show. The default is "all".
        limits : list, optional
            List of the limits. The default is None.
            Format : [x_min, x_max, y_min, y_max, z_min, z_max]

        """
        
        # Creating the figure and the axes.
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        # Setting the iteration through track if more than one.
        if track == "all":
            track = list(self.data["track"].value_counts(ascending = True))
        elif type(track) in [float, int]:
            track = [track]
        
        # Setting the limits of the figure.
        if limits == None:
            lim = pd.Series([self.data["x"].min()-5, self.data["x"].max()+5,
                             self.data["y"].min()-5, self.data["y"].max()+5,
                             self.data["z"].min()-5, self.data["z"].max()+5],
                            index = ["xm", "xM", "ym", "yM", "zm", "zM"],
                            dtype = "float")
        else :
            lim = pd.Series(limits, 
                            index = ["xm", "xM", "ym", "yM", "zm", "zM"],
                            dtype = "float")
        
        # Getting the data and plotting tracks one by one.
        for trackID in track :
            data = self.data[self.data["track"] == trackID]
            ax.plot(data["x"], data["y"], data["z"])
        
        # Setting axis labels.
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Setting the axis' limits. 
        ax.set_xlim3d([lim["xm"], lim["xM"]])
        ax.set_ylim3d([lim["ym"], lim["yM"]])
        ax.set_zlim3d([lim["zm"], lim["zM"]])

        plt.show()
        plt.close()
        
        
    def exportCSV(self, savepath, OAT = False):
        """
        Save self.data as a .csv in savepath.

        Parameters
        ----------
        savepath : str
            Path of the output file.
        OAT : bool, optional
            If True, save the dataset to be imported within OAT. 
            The default is False.

        """
        if OAT :
            tracksDir = savepath+'\\data\\tracks\\'
            tifDir = savepath+'\\data\\organoid_images\\'
            
            data = self.data.copy()
            data["ID"] = data.index
            
            emptyRows = pd.DataFrame([[-1]*len(data.columns) 
                                      for row in range(3)],
                                     columns = data.columns,
                                     dtype = "int")
            
            data = pd.concat([emptyRows, data])
            
            data["FRAME"] = data["tp"]
            data["QUALITY"] = [0]*data.shape[0]
            data.rename(columns = {"track" : "TRACK_ID", "x": "POSITION_X", 
                                   "y":"POSITION_Y", "z" : "POSITION_Z", 
                                   "tp": "POSITION_T", 
                                   "target": "SPOT_TARGET_ID"}, 
                        inplace = True)
            
            tracks = data.drop(columns = "SPOT_TARGET_ID")
            tracks.to_csv(tracksDir+"tracks.csv")
            
            data.rename(columns = {"ID": "SPOT_SOURCE_ID"}, inplace = True)
            
            edges = data.loc[:,["SPOT_TARGET_ID", "SPOT_SOURCE_ID"]]
            edges.dropna(inplace = True)
            edges.to_csv(tracksDir+"edges.csv")
            
        else :
            self.data.to_csv(savepath)
        
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
        