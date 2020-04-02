from crm_rfm_modeling import rfm
from crm_rfm_modeling.rfm import RFM

import pandas as pd
online = pd.read_csv('C:/Users/nikit/Downloads/datar.csv')

import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import numpy as np


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

online = online.rename(columns={"Дата и время": "InvoiceDay"})
online
