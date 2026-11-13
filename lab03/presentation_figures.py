"""Neutral candidate illustrations for the presentation before model selection."""
import numpy as np
import networkx as nx
from .model import ROOT,graph
from .plotting import draw_tree,save
import matplotlib.pyplot as plt


def make_figures():
    mask=23  # An example from the library, not a pre-imposed answer.
    fig,ax=plt.subplots(figsize=(7.6,3.8))
    draw_tree(mask,ax,expanded=False,title='Пример кандидата: включены 4 из 6 ветвей')
    for text in ax.texts:text.set_fontsize(10)
    save(fig,'candidate_tree')
    g=graph(mask);order=list(nx.topological_sort(g))
    fig,ax=plt.subplots(figsize=(5.2,5.2),layout='constrained')
    ax.imshow(nx.to_numpy_array(g,nodelist=order),cmap='Blues',vmin=0,vmax=1)
    ax.set(xlabel='Вершина-приёмник',ylabel='Вершина-источник',
        title=f'Матрица того же дерева: {len(order)} вершины')
    save(fig,'candidate_adjacency')
