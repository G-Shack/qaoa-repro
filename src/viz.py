# src/viz.py
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_heatmap(grid, gamma_vals, beta_vals, title, fname=None):
    plt.figure(figsize=(6,5))
    plt.imshow(grid, origin='lower', aspect='auto', extent=[gamma_vals[0], gamma_vals[-1], beta_vals[0], beta_vals[-1]])
    plt.colorbar()
    plt.xlabel('gamma')
    plt.ylabel('beta')
    plt.title(title)
    if fname:
        plt.savefig(fname, bbox_inches='tight', dpi=200)
    plt.show()

def plot_histogram(values, title, fname=None, bins=30):
    plt.figure(figsize=(6,4))
    plt.hist(values, bins=bins)
    plt.title(title)
    if fname:
        plt.savefig(fname, bbox_inches='tight', dpi=200)
    plt.show()
