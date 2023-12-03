 # -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:15:25 2019

@author: fgome
"""

import pandas as pd
import matplotlib.pyplot as plt

def GeneratePolygon():
    pts = [[0,0], [0,-5], [3,-7], [0, -9], [5, -9], [7,-7], [7,-9], [12, -9], [12, -6], [14, -6], [14, -9], [16, -7], [14, -3], [8, -3], [6, -1], [5, -3], [6,-3], [7, -4], [3, -5], [0,0]]
    df = pd.DataFrame.from_records(pts, columns = ["x","y"])
    return df

if __name__ == "__main__":
    df = GeneratePolygon()
    print(df)
    plt.scatter(df["x"], df["y"], c="k")
    plt.plot(df["x"], df["y"],c="k")
    plt.title("Galeria de Arte")
    plt.axis("off")
    
