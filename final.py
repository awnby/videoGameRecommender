import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib import *
import sys
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score

imdb = pd.read_csv("imdb.csv")
