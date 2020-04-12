# https://deeplearningcourses.com/c/deep-learning-prerequisites-the-numpy-stack-in-python
# https://www.udemy.com/deep-learning-prerequisites-the-numpy-stack-in-python

from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import pandas as pd
import matplotlib.pyplot as plt


# load in the data
df = pd.read_csv('../../large_files/train.csv')

im = df.groupby(['label']).mean().values.reshape(280,28)

plt.figure(figsize = (20,20)) 
plt.imshow(255-im, cmap="gray")
