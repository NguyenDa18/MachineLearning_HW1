import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

# construct argument parse
ap = argparse.ArgumentParser()

dip_df = pd.read_csv("dip-har-eff.csv")
drift_df = pd.read_csv("drift-har-eff.csv")
set_df = pd.read_csv("set-har-eff.csv")

## Sigmoid Activation
def sigmoid_activation(x):
    # compute and return sigmoid activation val for given input val
    return 1.0 / (1 + np.exp(-x))

print(drift_df.head(5))
