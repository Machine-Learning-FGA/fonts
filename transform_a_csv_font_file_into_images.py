#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Transform a single csv font file into images.
For more details on how this script works see:
print_font_images.ipynb
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv = pd.read_csv("./BOOK.csv")
csv = csv.query("strength==0.4 and italic==0.0 and orientation==0.0")

# the "if" is because jupyter is being a bitch with its cache...
csv.drop("font", axis=1, inplace=True)        if "font" in csv else ...
csv.drop("fontVariant", axis=1, inplace=True) if "fontVariant" in csv else ...
csv.drop("strength", axis=1, inplace=True)    if "strength" in csv else ...
csv.drop("italic", axis=1, inplace=True)      if "italic" in csv else ...
csv.drop("orientation", axis=1, inplace=True) if "orientation" in csv else ...
csv.drop("m_top", axis=1, inplace=True)       if "m_top" in csv else ...
csv.drop("m_left", axis=1, inplace=True)      if "m_left" in csv else ...
csv.drop("italic", axis=1, inplace=True)      if "italic" in csv else ...
csv.drop("originalH", axis=1, inplace=True)   if "originalH" in csv else ...
csv.drop("originalW", axis=1, inplace=True)   if "originalW" in csv else ...
csv.drop("h", axis=1, inplace=True)           if "h" in csv else ...
csv.drop("w", axis=1, inplace=True)           if "w" in csv else ...

labels = csv['m_label']
csv.drop("m_label", axis=1, inplace=True) if "m_label" in csv else ...

features = csv.as_matrix()

index = 0
for row in features:
    dow_data = np.array(row)
    dow_data = dow_data.reshape((20, 20))

    # Prepare and save each image
    plt.imshow(dow_data)
    plt.axis('off')
    plt.savefig("./figures/%s" % labels[index], bbox_inches='tight')
    plt.clf() # clear plt for the next plot

    index += 1
