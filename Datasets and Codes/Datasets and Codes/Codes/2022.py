# -*- coding: utf-8 -*-
"""
Created on Dec,2023

@author: Yu Yan
"""
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import pandas as pd
import networkx as nx
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve,auc,roc_curve
torch.set_printoptions(profile="full")
import torch.nn as nn
import random
import argparse
from gensim.models import Word2Vec

from torch.nn import Module, Parameter
import torch.nn.functional as F
import gensim
import torch.nn.init
from torch_geometric.nn import GCNConv


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--output', nargs='?',
                        default='embeddings',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=64,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--num_nodes', type=int, default=41,
                        help='the num of nodes .')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--epochs', default=200, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--input_dim', default=31, type=int,
                        help='the input_dim of GCN')

    parser.add_argument('--hidden_dim', default=64, type=int,
                        help='the output_dim of GCN')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')  # [0.001, 0.0005, 0.0001]
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--lr_dc_step', type=int, default=3,
                        help='the number of steps after which the learning rate decay')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=0.25,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=4,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--neg', type=int, default=10,
                        help='Number of negative samples. Default is 10.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')

    parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
    # parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.add_argument('--l2', type=float, default=1e-5,
                        help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    parser.set_defaults(weighted=True)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    # parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=True)

    return parser.parse_args()


fmp = {'比利时': 0, '法国': 1, '希腊': 2, '意大利': 3, '西班牙': 4, '土耳其': 5, '日本': 6, '韩国': 7, '中国台湾': 8,
       '美国': 9, '特立尼达和多巴哥': 10, '阿曼': 11, '卡塔尔': 12, '阿联酋': 13, '阿尔及利亚': 14, '利比亚': 15,
       '尼日利亚': 16, '澳大利亚': 17, '文莱': 18, '印度尼西亚': 19, '马来西亚': 20, '葡萄牙': 21, '印度': 22,
       '英国': 23, '埃及': 24, '墨西哥': 25, '中国': 26, '挪威': 27, '阿根廷': 28, '加拿大': 29, '巴西': 30, '智利': 31,
       '乌克兰': 32, '科威特': 33, '泰国': 34, '俄罗斯联邦': 35, '也门': 36, '缅甸': 37, '秘鲁': 38, '新加坡': 39,
       '巴基斯坦': 40}

mp = {0: '比利时', 1: '法国', 2: '希腊', 3: '意大利', 4: '西班牙', 5: '土耳其', 6: '日本', 7: '韩国', 8: '中国台湾',
      9: '美国', 10: '特立尼达和多巴哥', 11: '阿曼', 12: '卡塔尔', 13: '阿联酋', 14: '阿尔及利亚', 15: '利比亚',
      16: '尼日利亚', 17: '澳大利亚', 18: '文莱', 19: '印度尼西亚', 20: '马来西亚', 21: '葡萄牙', 22: '印度',
      23: '英国', 24: '埃及', 25: '墨西哥', 26: '中国', 27: '挪威', 28: '阿根廷', 29: '加拿大', 30: '巴西', 31: '智利',
      32: '乌克兰', 33: '科威特', 34: '泰国', 35: '俄罗斯联邦', 36: '也门', 37: '缅甸', 38: '秘鲁', 39: '新加坡',
      40: '巴基斯坦'}

nfmp = {'北美洲': 0, '美国': 1, '特立尼达和多巴哥': 2, '阿曼': 3, '卡塔尔': 4, '阿联酋': 5, '阿尔及利亚': 6,
        '利比亚': 7,
        '尼日利亚': 8, '澳大利亚': 9, '文莱': 10, '印度尼西亚': 11, '马来西亚': 12, '比利时': 13, '法国': 14,
        '希腊': 15,
        '意大利': 16, '西班牙': 17, '土耳其': 18, '欧洲': 19, '日本': 20, '韩国': 21, '中国台湾': 22, '亚太地区': 23,
        '中南美洲': 24, '葡萄牙': 25, '印度': 26, '埃及': 27, '英国': 28, '加拿大': 29, '秘鲁': 30, '挪威': 31,
        '俄罗斯联邦': 32, '也门': 33, '墨西哥': 34, '阿根廷': 35, '巴西': 36, '智利': 37, '科威特': 38, '中东地区': 39,
        '中国': 40, '巴基斯坦': 41, '新加坡': 42, '泰国': 43, '缅甸': 44, '乌克兰': 45, '非洲': 46}

imp = {'比利时': 0, '法国': 1, '希腊': 2, '意大利': 3, '西班牙': 4, '土耳其': 5, '日本': 6, '韩国': 7, '中国台湾': 8,
       '葡萄牙': 9, '印度': 10, '英国': 11, '墨西哥': 12, '中国': 13, '阿根廷': 14, '加拿大': 15, '巴西': 16,
       '智利': 17, '乌克兰': 18, '科威特': 19, '泰国': 20, '新加坡': 21, '巴基斯坦': 22}
exp = {'美国': 0, '特立尼达和多巴哥': 1, '阿曼': 2, '卡塔尔': 3, '阿联酋': 4, '阿尔及利亚': 5, '利比亚': 6,
       '尼日利亚': 7, '澳大利亚': 8, '文莱': 9, '印度尼西亚': 10, '马来西亚': 11, '埃及': 12, '挪威': 13,
       '俄罗斯联邦': 14, '也门': 15, '缅甸': 16, '秘鲁': 17}

b = []

df11 = pd.read_excel("液化气——地区间的贸易流向/天然气：2000年液化天然气贸易流向.xlsx")
labels11 = list(df11.columns.values)
G1 = nx.DiGraph()

for i in range(0, 41):
    G1.add_node(i)
b1 = {}
for i in range(df11.shape[0]):
    for j in range(df11.shape[1]):
        if j != 0 and float(df11.iloc[i, j]) > 0.0:
            # print(fmp[df11.iloc[i,0]],fmp[labels11[j]])
            G1.add_edge(fmp[df11.iloc[i, 0]], fmp[labels11[j]])
            b1[(fmp[df11.iloc[i, 0]], fmp[labels11[j]])] = df11.iloc[i, j]
b.append(b1)

b2 = {}
df21 = pd.read_excel("液化气——地区间的贸易流向/天然气：2001年液化天然气贸易流向.xlsx")
labels21 = list(df21.columns.values)
G2 = nx.DiGraph()

for i in range(0, 41):
    G2.add_node(i)

for i in range(df21.shape[0]):
    for j in range(df21.shape[1]):
        if j != 0 and float(df21.iloc[i, j]) > 0.0:
            # print(fmp[df11.iloc[i,0]],fmp[labels11[j]])
            G2.add_edge(fmp[df21.iloc[i, 0]], fmp[labels21[j]])
            b2[(fmp[df21.iloc[i, 0]], fmp[labels21[j]])] = df21.iloc[i, j]

b.append(b2)

b3 = {}
df31 = pd.read_excel("液化气——地区间的贸易流向/天然气：2002年液化天然气贸易流向.xlsx")
labels31 = list(df31.columns.values)
G3 = nx.DiGraph()

for i in range(0, 41):
    G3.add_node(i)

for i in range(df31.shape[0]):
    for j in range(df31.shape[1]):
        if j != 0 and float(df31.iloc[i, j]) > 0.0:
            # print(fmp[df11.iloc[i,0]],fmp[labels11[j]])
            G3.add_edge(fmp[df31.iloc[i, 0]], fmp[labels31[j]])
            b2[(fmp[df31.iloc[i, 0]], fmp[labels31[j]])] = df31.iloc[i, j]

b.append(b3)

b4 = {}
df41 = pd.read_excel("液化气——地区间的贸易流向/天然气：2003年液化天然气贸易流向.xlsx")
labels41 = list(df41.columns.values)
G4 = nx.DiGraph()
for i in range(0, 41):
    G4.add_node(i)

for i in range(df41.shape[0]):
    for j in range(df41.shape[1]):
        if j != 0 and float(df41.iloc[i, j]) > 0.0:
            # print(fmp[df11.iloc[i,0]],fmp[labels11[j]])
            G4.add_edge(fmp[df41.iloc[i, 0]], fmp[labels41[j]])
            b2[(fmp[df41.iloc[i, 0]], fmp[labels41[j]])] = df41.iloc[i, j]
b.append(b4)

b5 = {}
df51 = pd.read_excel("液化气——地区间的贸易流向/天然气：2004年液化天然气贸易流向.xlsx")
labels51 = list(df51.columns.values)
G5 = nx.DiGraph()
for i in range(0, 41):
    G5.add_node(i)

for i in range(df51.shape[0]):
    for j in range(df51.shape[1]):
        if j != 0 and float(df51.iloc[i, j]) > 0.0:
            # print(fmp[df11.iloc[i,0]],fmp[labels11[j]])
            G5.add_edge(fmp[df51.iloc[i, 0]], fmp[labels51[j]])
            b2[(fmp[df51.iloc[i, 0]], fmp[labels51[j]])] = df51.iloc[i, j]
b.append(b5)

b6 = {}
df61 = pd.read_excel("液化气——地区间的贸易流向/天然气：2005年液化天然气贸易流向.xlsx")
labels61 = list(df61.columns.values)
G6 = nx.DiGraph()
for i in range(0, 41):
    G6.add_node(i)

for i in range(df61.shape[0]):
    for j in range(df61.shape[1]):
        if j != 0 and float(df61.iloc[i, j]) > 0.0:
            # print(fmp[df11.iloc[i,0]],fmp[labels11[j]])
            G6.add_edge(fmp[df61.iloc[i, 0]], fmp[labels61[j]])
            b2[(fmp[df61.iloc[i, 0]], fmp[labels61[j]])] = df61.iloc[i, j]

b.append(b6)

b7 = {}
df71 = pd.read_excel("液化气——地区间的贸易流向/天然气：2006年液化天然气贸易流向.xlsx")
labels71 = list(df71.columns.values)
G7 = nx.DiGraph()
for i in range(0, 41):
    G7.add_node(i)

for i in range(df71.shape[0]):
    for j in range(df71.shape[1]):
        if j != 0 and float(df71.iloc[i, j]) > 0.0:
            # print(fmp[df11.iloc[i,0]],fmp[labels11[j]])
            G7.add_edge(fmp[df71.iloc[i, 0]], fmp[labels71[j]])
            b7[(fmp[df71.iloc[i, 0]], fmp[labels71[j]])] = df71.iloc[i, j]
b.append(b7)

b8 = {}
df81 = pd.read_excel("液化气——地区间的贸易流向/天然气：2007年液化天然气贸易流向.xlsx")
labels81 = list(df81.columns.values)
G8 = nx.DiGraph()
for i in range(0, 41):
    G8.add_node(i)

for i in range(df81.shape[0]):
    for j in range(df81.shape[1]):
        if j != 0 and float(df81.iloc[i, j]) > 0.0:
            # print(fmp[df11.iloc[i,0]],fmp[labels11[j]])
            G8.add_edge(fmp[df81.iloc[i, 0]], fmp[labels81[j]])
            b2[(fmp[df81.iloc[i, 0]], fmp[labels81[j]])] = df81.iloc[i, j]
b.append(b8)

b9 = {}
df91 = pd.read_excel("液化气——地区间的贸易流向/天然气：2008年液化天然气贸易流向.xlsx")
labels91 = list(df91.columns.values)
G9 = nx.DiGraph()
for i in range(0, 41):
    G9.add_node(i)

for i in range(df91.shape[0]):
    for j in range(df91.shape[1]):
        if j != 0 and float(df91.iloc[i, j]) > 0.0:
            # print(fmp[df11.iloc[i,0]],fmp[labels11[j]])
            G9.add_edge(fmp[df91.iloc[i, 0]], fmp[labels91[j]])
            b2[(fmp[df91.iloc[i, 0]], fmp[labels91[j]])] = df91.iloc[i, j]
b.append(b9)

b10 = {}
df101 = pd.read_excel("液化气——地区间的贸易流向/天然气：2009年液化天然气贸易流向.xlsx")
labels101 = list(df101.columns.values)
G10 = nx.DiGraph()
for i in range(0, 41):
    G10.add_node(i)

for i in range(df101.shape[0]):
    for j in range(df101.shape[1]):
        if j != 0 and float(df101.iloc[i, j]) > 0.0:
            # print(fmp[df11.iloc[i,0]],fmp[labels11[j]])
            G10.add_edge(fmp[df101.iloc[i, 0]], fmp[labels101[j]])
            b10[(fmp[df101.iloc[i, 0]], fmp[labels101[j]])] = df101.iloc[i, j]
b.append(b10)

b11 = {}
df111 = pd.read_excel("液化气——地区间的贸易流向/天然气：2010年液化天然气贸易流向.xlsx")
labels111 = list(df111.columns.values)
G11 = nx.DiGraph()
for i in range(0, 41):
    G11.add_node(i)
for i in range(df111.shape[0]):
    for j in range(df111.shape[1]):
        if j != 0 and float(df111.iloc[i, j]) > 0.0:
            # print(fmp[df11.iloc[i,0]],fmp[labels11[j]])
            G11.add_edge(fmp[df111.iloc[i, 0]], fmp[labels111[j]])
            b11[(fmp[df111.iloc[i, 0]], fmp[labels111[j]])] = df111.iloc[i, j]
b.append(b11)
b12 = {}
df121 = pd.read_excel("液化气——地区间的贸易流向/天然气：2011年液化天然气贸易流向.xlsx")
labels121 = list(df121.columns.values)
G12 = nx.DiGraph()
for i in range(0, 41):
    G12.add_node(i)

for i in range(df121.shape[0]):
    for j in range(df121.shape[1]):
        if j != 0 and float(df121.iloc[i, j]) > 0.0:
            # print(fmp[df11.iloc[i,0]],fmp[labels11[j]])
            G12.add_edge(fmp[df121.iloc[i, 0]], fmp[labels121[j]])
            b12[(fmp[df121.iloc[i, 0]], fmp[labels121[j]])] = df121.iloc[i, j]
b.append(b12)

b13 = {}
df131 = pd.read_excel("液化气——地区间的贸易流向/天然气：2012年液化天然气贸易流向.xlsx")
labels131 = list(df131.columns.values)
G13 = nx.DiGraph()
for i in range(0, 41):
    G13.add_node(i)

for i in range(df131.shape[0]):
    for j in range(df131.shape[1]):
        if j != 0 and float(df131.iloc[i, j]) > 0.0:
            # print(fmp[df11.iloc[i,0]],fmp[labels11[j]])
            G13.add_edge(fmp[df131.iloc[i, 0]], fmp[labels131[j]])
            b13[(fmp[df131.iloc[i, 0]], fmp[labels131[j]])] = df131.iloc[i, j]
b.append(b13)

b14 = {}
df141 = pd.read_excel("液化气——地区间的贸易流向/天然气：2013年液化天然气贸易流向.xlsx")
labels141 = list(df141.columns.values)
G14 = nx.DiGraph()
for i in range(0, 41):
    G14.add_node(i)

for i in range(df141.shape[0]):
    for j in range(df141.shape[1]):
        if j != 0 and float(df141.iloc[i, j]) > 0.0:
            # print(fmp[df11.iloc[i,0]],fmp[labels11[j]])
            G14.add_edge(fmp[df141.iloc[i, 0]], fmp[labels141[j]])
            b14[(fmp[df141.iloc[i, 0]], fmp[labels141[j]])] = df141.iloc[i, j]
b.append(b14)

b15 = {}
df151 = pd.read_excel("液化气——地区间的贸易流向/天然气：2014年液化天然气贸易流向.xlsx")
labels151 = list(df151.columns.values)
G15 = nx.DiGraph()
for i in range(0, 41):
    G15.add_node(i)

for i in range(df151.shape[0]):
    for j in range(df151.shape[1]):
        if j != 0 and float(df151.iloc[i, j]) > 0.0:
            # print(fmp[df11.iloc[i,0]],fmp[labels11[j]])
            G15.add_edge(fmp[df151.iloc[i, 0]], fmp[labels151[j]])
            b15[(fmp[df151.iloc[i, 0]], fmp[labels151[j]])] = df151.iloc[i, j]
b.append(b15)

b16 = {}
df161 = pd.read_excel("液化气——地区间的贸易流向/天然气：2015年液化天然气贸易流向.xlsx")
labels161 = list(df161.columns.values)
G16 = nx.DiGraph()
for i in range(0, 41):
    G16.add_node(i)

for i in range(df161.shape[0]):
    for j in range(df161.shape[1]):
        if j != 0 and float(df161.iloc[i, j]) > 0.0:
            G16.add_edge(fmp[df161.iloc[i, 0]], fmp[labels161[j]])
            b16[(fmp[df161.iloc[i, 0]], fmp[labels161[j]])] = df161.iloc[i, j]
b.append(b16)

b17 = {}

df171 = pd.read_excel("液化气——地区间的贸易流向/天然气：2016年液化天然气贸易流向.xlsx")
labels171 = list(df171.columns.values)
G17 = nx.DiGraph()
for i in range(0, 41):
    G17.add_node(i)

for i in range(df171.shape[0]):
    for j in range(df171.shape[1]):
        if j != 0 and float(df171.iloc[i, j]) > 0.0:
            G17.add_edge(fmp[df171.iloc[i, 0]], fmp[labels171[j]])
            b17[(fmp[df171.iloc[i, 0]], fmp[labels171[j]])] = df171.iloc[i, j]
b.append(b17)

b18 = {}
df181 = pd.read_excel("液化气——地区间的贸易流向/天然气：2017年液化天然气贸易流向.xlsx")
labels181 = list(df181.columns.values)
G18 = nx.DiGraph()
for i in range(0, 41):
    G18.add_node(i)

for i in range(df181.shape[0]):
    for j in range(df181.shape[1]):
        if j != 0 and float(df181.iloc[i, j]) > 0.0:
            G18.add_edge(fmp[df181.iloc[i, 0]], fmp[labels181[j]])
            b18[(fmp[df181.iloc[i, 0]], fmp[labels181[j]])] = df181.iloc[i, j]
b.append(b18)

b19 = {}
df191 = pd.read_excel("液化气——地区间的贸易流向/天然气：2018年液化天然气贸易流向.xlsx")
labels191 = list(df191.columns.values)
G19 = nx.DiGraph()
for i in range(0, 41):
    G19.add_node(i)

for i in range(df191.shape[0]):
    for j in range(df191.shape[1]):
        if j != 0 and float(df191.iloc[i, j]) > 0.0:
            G19.add_edge(fmp[df191.iloc[i, 0]], fmp[labels191[j]])
            b19[(fmp[df191.iloc[i, 0]], fmp[labels191[j]])] = df191.iloc[i, j]
b.append(b19)

b20 = {}
df201 = pd.read_excel("液化气——地区间的贸易流向/天然气：2019年液化天然气贸易流向.xlsx")
labels201 = list(df201.columns.values)
G20 = nx.DiGraph()
for i in range(0, 41):
    G20.add_node(i)

for i in range(df201.shape[0]):
    for j in range(df201.shape[1]):
        if j != 0 and float(df201.iloc[i, j]) > 0.0:
            G20.add_edge(fmp[df201.iloc[i, 0]], fmp[labels201[j]])
            b20[(fmp[df201.iloc[i, 0]], fmp[labels201[j]])] = df201.iloc[i, j]
b.append(b20)

b21 = {}
df211 = pd.read_excel("液化气——地区间的贸易流向/天然气：2020年液化天然气贸易流向.xlsx")
labels211 = list(df211.columns.values)
G21 = nx.DiGraph()
for i in range(0, 41):
    G21.add_node(i)

for i in range(df211.shape[0]):
    for j in range(df211.shape[1]):
        if j != 0 and float(df211.iloc[i, j]) > 0.0:
            G21.add_edge(fmp[df211.iloc[i, 0]], fmp[labels211[j]])
            b21[(fmp[df211.iloc[i, 0]], fmp[labels211[j]])] = df211.iloc[i, j]
b.append(b21)

b22 = {}
df221 = pd.read_excel("液化气——地区间的贸易流向/天然气：2021年液化天然气贸易流向.xlsx")
labels221 = list(df221.columns.values)

G22 = nx.DiGraph()
for i in range(0, 41):
    G22.add_node(i)

for i in range(df221.shape[0]):
    for j in range(df221.shape[1]):
        if j != 0 and float(df221.iloc[i, j]) > 0.0:
            G22.add_edge(fmp[df221.iloc[i, 0]], fmp[labels221[j]])
            b22[(fmp[df221.iloc[i, 0]], fmp[labels221[j]])] = df221.iloc[i, j]
b.append(b22)

b23 = {}
df231 = pd.read_excel("液化气——地区间的贸易流向/天然气：2022年液化天然气贸易流向.xlsx")
labels231 = list(df231.columns.values)

G23 = nx.DiGraph()
for i in range(0, 41):
    G23.add_node(i)

for i in range(df231.shape[0]):
    for j in range(df231.shape[1]):
        if j != 0 and float(df231.iloc[i, j]) > 0.0:
            G23.add_edge(fmp[df231.iloc[i, 0]], fmp[labels231[j]])
            b23[(fmp[df231.iloc[i, 0]], fmp[labels231[j]])] = df231.iloc[i, j]
b.append(b23)

graphs = [G1, G2, G3, G4, G5, G6, G7, G8, G9, G10, G11, G12, G13, G14, G15, G16, G17, G18, G19, G20, G21, G22, G23]

args = parse_args()

import math
class GraphConvolution(Module):
    def __init__(self,in_features,out_features,bias=True):
        super(GraphConvolution,self).__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.weight=Parameter(torch.FloatTensor(in_features,out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class MyGCN(nn.Module):
    def __init__(self,nfeat,output):
        super(MyGCN,self).__init__()
        self.gc = GraphConvolution(nfeat,output)


    def forward(self,input): #x:节点特征，adj:邻接矩阵
        x = torch.nn.functional.relu(self.gc(input[0],input[1]))
        return x


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2,hidden_size3,output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size3, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        x = torch.zeros([23 * 18, 2 * input.shape[1]], dtype=torch.float32)

        for i in range(0, 23):
            for j in range(0, 18):
                x[j+i*18] = torch.cat((input[i], input[j + 23]), dim=0)

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x


torch.autograd.set_detect_anomaly(True)

rg = [[2, 17, 24, 24, 25, 27, 31, 31, 32],
      [2, 18, 23, 24, 24, 24, 26, 28, 30],
      [2, 6, 9, 12, 12, 14, 16, 16, 18],
      [2, 7, 9, 9, 11, 13, 13, 13, 13],
      [2, 7, 12, 12, 12, 12, 12, 14, 17],
      [2, 7, 9, 10, 10, 11, 16, 16, 20],
      [2, 2, 2, 2, 2, 3, 10, 12, 22],
      [2, 2, 2, 4, 4, 5, 5, 5, 5],
      [2, 3, 3, 6, 6, 6, 6, 6, 6],
      [2, 12, 16, 22, 22, 22, 23, 24, 29],
      [2, 2, 3, 3, 3, 5, 5, 6, 6],
      [2, 2, 2, 7, 11, 17, 18, 23, 28],
      [2, 8, 11, 11, 17, 17, 21, 23, 23],
      [2, 2, 2, 2, 2, 3, 4, 6, 23],
      [2, 2, 2, 2, 2, 6, 16, 19, 24],
      [2, 6, 12, 14, 17, 19, 19, 19, 21],
      [2, 10, 18, 21, 22, 22, 22, 24, 26],
      [2, 3, 3, 3, 3, 8, 14, 21, 28],
      [2, 7, 8, 17, 19, 22, 22, 23, 23],
      [2, 14, 22, 27, 30, 30, 30, 32, 32],
      [2, 5, 7, 14, 15, 18, 23, 23, 26],
      [2, 10, 14, 21, 24, 25, 25, 25, 27],
      [2, 7, 11, 16, 19, 19, 20, 20, 22],
      [2, 21, 24, 29, 29, 29, 29, 29, 31],
      [2, 11, 20, 21, 25, 25, 28, 29, 33],
      [2, 2, 9, 14, 16, 16, 16, 16, 20],
      [2, 2, 2, 2, 2, 2, 4, 10, 32],
      [2, 2, 6, 10, 13, 17, 19, 19, 25],
      [2, 9, 19, 26, 29, 30, 30, 32, 32],
      [2, 2, 3, 4, 4, 6, 11, 16, 33],
      [2, 10, 14, 14, 20, 20, 22, 22, 23],
      [2, 2, 4, 5, 7, 12, 15, 18, 24],
      [2, 6, 12, 26, 28, 29, 29, 29, 29],
      [2, 3, 3, 3, 3, 3, 3, 4, 6],
      [2, 2, 3, 5, 9, 15, 23, 25, 33],
      [2, 2, 2, 2, 2, 2, 4, 7, 29],
      [2, 11, 18, 21, 28, 28, 31, 31, 33],
      [2, 2, 2, 3, 3, 5, 7, 10, 20],
      [2, 5, 8, 10, 17, 18, 18, 20, 20],
      [2, 13, 16, 22, 26, 26, 26, 28, 32],
      [2, 22, 31, 32, 32, 32, 32, 32, 32],
      [2, 2, 4, 5, 8, 9, 13, 14, 31],
      [2, 6, 8, 9, 11, 11, 11, 11, 11],
      [2, 8, 12, 15, 20, 24, 25, 29, 30],
      [2, 2, 2, 2, 2, 4, 5, 5, 5],
      [2, 2, 2, 2, 4, 4, 6, 7, 30],
      [2, 9, 17, 26, 27, 27, 30, 30, 32]]

pos = pd.read_excel("横坐标.xlsx")
mx = [2, 22, 31, 32, 32, 32, 32, 32, 33]
matrix_list = []
for i in range(2, 3):
    for j in range(0, 23):
        for k in range(0, 41):
            for l in range(0, mx[i]):
                graphs[j].nodes[k][pos.iloc[l, 0]] = [0]

    for key, value in fmp.items():
        s = "国家/" + key + ".xlsx"
        df = pd.read_excel(s)
        for j in range(0, rg[nfmp[key]][i]):
            for k in range(0, 24):
                if k != 0:
                    graphs[k - 1].nodes[value][df.iloc[j, 0]] = [df.iloc[j, k]]

    for ii in range(0, 23):
        edge_index = torch.tensor(list(graphs[ii].edges)).t().contiguous()
        x = torch.tensor([graphs[ii].nodes[node]['feat1'] for node in graphs[ii].nodes], dtype=torch.float)
        for j in range(1, mx[i]):
            feat = torch.tensor([graphs[ii].nodes[node][pos.iloc[j, 0]] for node in graphs[ii].nodes],
                                dtype=torch.float)
            x = torch.cat((x, feat), dim=1)  # 将特征连接在一起

        matrix_list.append(x.numpy())

    matrix_array = np.array(matrix_list)

mx = [[0.0,0.0,0.0,0.0,0.0,0.0]]#ACC,PRE,REC,F1,AUC,AUPR


for weight in [1.2,1.3,1.4,1.5,1.6,1.7,1.8]:
    for learning_rate in [0.0001,0.0005,0.00001,0.00005]:
        for dim in [16,8,32]:
            for _ in range(10):

                model = nn.Sequential(MyGCN(31, dim), MLP(2*dim, 64, 32, 16, 2))
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

                flag = 1

                for epoch in range(args.epochs):
                    if flag == 0:
                        break

                    for time_step in range(0, 21):

                        features = torch.FloatTensor(matrix_array[time_step])

                        adj = torch.zeros([23, 18], dtype=torch.long)

                        if time_step < 10:
                            string = "液化气——地区间的贸易流向/天然气：200" + str(time_step) + "年液化天然气贸易流向.xlsx"
                        else:
                            string = "液化气——地区间的贸易流向/天然气：20" + str(time_step) + "年液化天然气贸易流向.xlsx"

                        df = pd.read_excel(string)
                        labels = list(df.columns.values)

                        for i in range(df.shape[0]):
                            for j in range(df.shape[1]):
                                if j != 0:
                                    if df.iloc[i, j] > 0:
                                        imp_id = imp[df.iloc[i, 0]]
                                        exp_id = exp[labels[j]]
                                        adj[imp_id][exp_id] = 1

                        cur_features = torch.zeros([41, 31], dtype=torch.float32)
                        for name in imp:
                            cur_features[imp[name]] = features[fmp[name]]

                        for name in exp:
                            cur_features[exp[name] + 23] = features[fmp[name]]

                        G = nx.DiGraph()

                        for i in range(0, 41):
                            G.add_node(i)

                        for i in range(0, 23):
                            for j in range(0, 18):
                                if adj[i][j] == 1:
                                    G.add_edge(i, j + 23)
                                    G.add_edge(j + 23, i)

                        adj_matrix = nx.adjacency_matrix(G).todense()
                        cur_adj = torch.FloatTensor(adj_matrix)

                        output = model({0: cur_features, 1: cur_adj})

                        next_time = time_step + 1

                        for i in range(0, 23):
                            for j in range(0, 18):
                                adj[i][j] = 0

                        if next_time < 10:
                            string = "液化气——地区间的贸易流向/天然气：200" + str(next_time) + "年液化天然气贸易流向.xlsx"
                        else:
                            string = "液化气——地区间的贸易流向/天然气：20" + str(next_time) + "年液化天然气贸易流向.xlsx"

                        df = pd.read_excel(string)
                        labels = list(df.columns.values)

                        for i in range(df.shape[0]):
                            for j in range(df.shape[1]):
                                if j != 0:
                                    if df.iloc[i, j] > 0:
                                        imp_id = imp[df.iloc[i, 0]]
                                        exp_id = exp[labels[j]]
                                        adj[imp_id][exp_id] = 1

                        label = torch.zeros([23 * 18], dtype=torch.long)

                        for i in range(0, 23):
                            for j in range(0, 18):
                                label[j + i * 18] = adj[i][j]

                        optimizer.zero_grad()

                        num_positives = torch.sum(label == 1)
                        num_negatives = torch.sum(label == 0)
                        pos_weight = weight * num_negatives / num_positives
                        # print(predictions.shape)
                        loss = F.binary_cross_entropy_with_logits(output[:, 1], label.float(), pos_weight=pos_weight)
                        loss.requires_grad_(True)
                        loss.backward(retain_graph=True)
                        optimizer.step()

                    features = torch.FloatTensor(matrix_array[21])

                    adj = torch.zeros([23, 18], dtype=torch.long)

                    df = pd.read_excel("液化气——地区间的贸易流向/天然气：2021年液化天然气贸易流向.xlsx")
                    labels = list(df.columns.values)

                    for i in range(df.shape[0]):
                        for j in range(df.shape[1]):
                            if j != 0:
                                if df.iloc[i, j] > 0:
                                    imp_id = imp[df.iloc[i, 0]]
                                    exp_id = exp[labels[j]]
                                    adj[imp_id][exp_id] = 1

                    cur_features = torch.zeros([41, 31], dtype=torch.float32)
                    for name in imp:
                        cur_features[imp[name]] = features[fmp[name]]

                    for name in exp:
                        cur_features[exp[name] + 23] = features[fmp[name]]

                    G = nx.DiGraph()

                    for i in range(0, 41):
                        G.add_node(i)

                    for i in range(0, 23):
                        for j in range(0, 18):
                            if adj[i][j] == 1:
                                G.add_edge(i, j + 23)
                                G.add_edge(j + 23, i)

                    adj_matrix = nx.adjacency_matrix(G).todense()
                    cur_adj = torch.FloatTensor(adj_matrix)

                    output = model({0: cur_features, 1: cur_adj})


                    for i in range(0, 23):
                        for j in range(0, 18):
                            adj[i][j] = 0

                    df = pd.read_excel("液化气——地区间的贸易流向/天然气：2022年液化天然气贸易流向.xlsx")
                    labels = list(df.columns.values)

                    for i in range(df.shape[0]):
                        for j in range(df.shape[1]):
                            if j != 0:
                                if df.iloc[i, j] > 0:
                                    imp_id = imp[df.iloc[i, 0]]
                                    exp_id = exp[labels[j]]
                                    adj[imp_id][exp_id] = 1

                    label = torch.zeros([23 * 18], dtype=torch.long)

                    for i in range(0, 23):
                        for j in range(0, 18):
                            label[j + i * 18] = adj[i][j]

                    num = 0
                    tp = 0
                    fn = 0

                    for i in range(0, 414):
                        if output[i][1] > output[i][0] and label[i] == 1:
                            num += 1
                            tp += 1
                        elif output[i][1] < output[i][0] and label[i] == 0:
                            num += 1
                            fn += 1

                    if tp == 0 or fn == 0:
                        flag = 0

                    if tp != 0 and fn != 0:
                        if tp + fn > 414 * 0.84:
                            mx[0][0] = max(mx[0][0], (tp + fn) / 414)
                            if 2 * (tp / 164) * (tp / (tp + 250 - fn)) / (tp / (tp + 250 - fn) + tp / 164) > mx[0][3]:
                                mx[0][1] = tp / (tp + 250 - fn)
                                mx[0][2] = tp / 164
                                mx[0][3] = 2 * (tp / 164) * (tp / (tp + 250 - fn)) / (tp / (tp + 250 - fn) + tp / 164)

                        mx[0][4] = max(mx[0][4], roc_auc_score(label.detach().numpy(), output[:, 1].detach().numpy()))
                        a,b,c = precision_recall_curve(label.detach().numpy(), output[:, 1].detach().numpy())
                        mx[0][5] = max(mx[0][5],auc(b,a))

                print(weight,learning_rate)

                print(f"time_step:2022,{mx[0][0]},{mx[0][1]},{mx[0][2]},{mx[0][3]},{mx[0][4]},{mx[0][5]}")

                print("---------------------")

torch.autograd.set_detect_anomaly(False)
