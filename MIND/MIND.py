import copy
import pickle
import random
import gc
import sys
import time
import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from itertools import combinations


def similarity(G1, G2):
    G1_edge = np.where(G1 == 1)
    com_edge = np.where(G2[G1_edge] == 1)
    return com_edge[0].size * 2 / (G1_edge[0].size + np.where(G1 == 1)[0].size) * 1.0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cal_log(x):
    if x <= 0:
        return -1e5
    result = np.log10(x) / np.log10(np.exp(1))
    return result
cal_log_func = np.vectorize(cal_log)



def cal_mutual_info(x, y, mode):
    len_x = len(x) * 1.0
    x_y_dict = {}
    mutual_info = 0
    for xi in range(2):
        for yi in range(2):
            xi_index = np.where(x == xi)[0]
            len_xi_index = len(xi_index) * 1.0
            yi_index = np.where(y == yi)[0]
            len_yi_index = len(yi_index) * 1.0

            x_y_count = len(set(yi_index) & set(xi_index)) * 1.0
            x_y_dict['x_' + str(xi)] = len_xi_index
            x_y_dict['y_' + str(yi)] = len_yi_index
            x_y_dict['x_' + str(xi) + '_y_' + str(yi)] = x_y_count
            x_y_mutual = 0
            if x_y_count > 0:
                x_y_mutual = (x_y_count / len_x) * math.log((len_x * x_y_count) / (len_xi_index * len_yi_index))
            if xi != yi:
                if mode == 1:
                    mutual_info -= x_y_mutual
                elif mode == 2:
                    mutual_info -= abs(x_y_mutual)
                else:
                    mutual_info += x_y_mutual
            else:
                mutual_info += x_y_mutual
    return mutual_info



def find_kmeans_threshold(data, _small_center_select):
    estimator = KMeans(n_clusters=2)
    estimator.fit(data)

    return estimator.cluster_centers_, estimator.labels_



def k_combine2(glist, k):
    return combinations(glist, k)



def cal_precision_recall(graph_label, graph_matrix):
    all_num = graph_label.sum()
    find_num = graph_matrix.sum()
    right_num = (graph_label * graph_matrix).sum()

    epsilon = 1e-5
    precision = right_num / (find_num + epsilon)
    recall = right_num / (all_num + epsilon)
    f1 = (2 * precision * recall) / (precision + recall + epsilon)

    return precision, recall, f1



def read_records_and_graph(record_path, graph_path):
    record_df = pd.read_csv(record_path, sep='\t', header=None)
    record_data = record_df.values

    node_num = record_df.shape[1]
    graph_df = pd.read_csv(graph_path, sep='\t', header=None)
    graph_df.columns = ['u', 'v']
    graph_df = graph_df
    graph_label = np.zeros([node_num, node_num])
    for i in range(graph_df.shape[0]):
        iu = graph_df.loc[i, 'u']
        iv = graph_df.loc[i, 'v']
        graph_label[iu][iv] = 1
    return record_data, graph_label



def save_graph_dict(graph_dict, graph_path):
    fp = open(graph_path, "wb")
    pickle.dump(graph_dict, fp)
    fp.close()
    return True


def load_graph_dict(graph_path):
    fp = open(graph_path, "rb")
    graph_dict = pickle.load(fp)
    return graph_dict


def numpy2dec(line):
    j = 0
    for m in range(line.size):
        j = j + pow(2, line.size - 1 - m) * line[m]
    return int(j)



def cal_parents_left_score(node_record_data, parent_record_data):
    Fi_dict = {}
    xFi_dict = {}
    for i in range(node_record_data.shape[0]):
        x_i = node_record_data[i, 0]
        Fi_i = tuple(parent_record_data[i, :])
        xFi_i = (x_i, ) + Fi_i
        Fi_dict[Fi_i] = Fi_dict.get(Fi_i, 0) + 1.0
        xFi_dict[xFi_i] = xFi_dict.get(xFi_i, 0) + 1.0
    left_sum = 0
    for xFi, xFi_count in xFi_dict.items():
        Fi = xFi[1:]
        Fi_count = Fi_dict[Fi]
        xFi_prob = xFi_count / Fi_count
        xFi_score = xFi_count * math.log(xFi_prob)
        left_sum += xFi_score

    return left_sum



def cal_parents_right_score(node_index, parent_indexs, edge_prob, task_num, alpha):

    father_prob = 1 - edge_prob[:, node_index]
    father_prob[parent_indexs] = 1 - father_prob[parent_indexs]

    father_prob_ = father_prob[np.where(father_prob != (task_num - 1 + alpha) / (task_num - 1 + 2 * alpha))[0]]

    temp = (father_prob.shape[0] - father_prob_.shape[0]) * math.log((task_num - 1 + alpha) / (task_num - 1 + 2 * alpha))
    if father_prob_.shape[0] == 0:
        return temp
    return cal_log_func(father_prob_).sum() + temp



def cal_parents_score(node_index, parent_indexs, node_record_data, parent_record_data, edge_prob, penalize_coef):

    score = cal_parents_left_score(node_record_data, parent_record_data) + \
            penalize_coef * cal_parents_right_score(node_index, parent_indexs, edge_prob)

    return score


def construct_from_records(use_record_data, mutual_info_mode, more_zero_mutual_info, small_center_select, parents_limit):
    node_num = use_record_data.shape[1]

    mutual_infos = np.zeros([node_num, node_num], dtype=np.float32)
    for ri in range(node_num):
        for rj in range(node_num):
            if ri > rj:
                ri_rj_mutualInfo = cal_mutual_info(use_record_data[:, ri:ri + 1], use_record_data[:, rj:rj + 1], mutual_info_mode)
                mutual_infos[ri][rj] = ri_rj_mutualInfo
                mutual_infos[rj][ri] = ri_rj_mutualInfo


    mutual_infos_reshape = np.array(mutual_infos).reshape([-1, 1])
    if more_zero_mutual_info > 0:
        mutual_infos_reshape = mutual_infos_reshape[mutual_infos_reshape > 0].reshape([-1, 1])

    kmeans_centers, kmeans_labels = find_kmeans_threshold(mutual_infos_reshape, small_center_select)

    clf_coef_0_max = mutual_infos_reshape[kmeans_labels == 0].max()
    clf_coef_1_max = mutual_infos_reshape[kmeans_labels == 1].max()
    mutual_info_threshold = min(clf_coef_0_max, clf_coef_1_max)


    mutual_info_graph = np.zeros([node_num, node_num], dtype=np.float32)
    mutual_info_graph[mutual_infos > mutual_info_threshold] = 1

    print("parents_num = ", np.sum(mutual_info_graph, axis=0))

    return mutual_infos, mutual_info_threshold, mutual_info_graph



def construct_multi_diffusion_network(record_data_list, graph_lable_list, mutual_info_mode, more_zero_mutual_info, small_center_select, mutual_info_coef,
    initial_graph_use_AllDataMI, initial_graph_use_topKMI, alpha, parents_limit_coef, parents_limit, penalize_coef, print_parents_info,  f_log):

    task_num = len(record_data_list)
    node_num = record_data_list[0].shape[1]

    old_graph_list = []
    mutual_info_list = []
    mutual_info_threshold_list = []
    mutual_info_graph_list = []
    left_scores = [[list() for i in range(node_num)] for t in range(task_num)]

    for t in range(task_num):
        use_record_data = record_data_list[t]
        mutual_infos, mutual_info_threshold, mutual_info_graph = construct_from_records(use_record_data, mutual_info_mode, more_zero_mutual_info, small_center_select, parents_limit)

        mutual_info_list.append(mutual_infos.copy())
        mutual_info_threshold_list.append(mutual_info_threshold)
        old_graph_list.append(mutual_info_graph.copy())
        mutual_info_graph_list.append(mutual_info_graph.copy())

        if print_parents_info:
            print('self_kmeans_threshold = ' + str(mutual_info_threshold))
            print('self_kmeans_threshold = ' + str(mutual_info_threshold), file=f_log, flush=True)
            print('mutual find edge sum = ' + str(mutual_info_graph.sum()))
            print('mutual find edge sum = ' + str(mutual_info_graph.sum()), file=f_log, flush=True)
            print('--------------------- mutual graph for ' + str(t) + ' done')
            print('--------------------- mutual graph for ' + str(t) + ' done', file=f_log, flush=True)
    if print_parents_info:
        for t in range(task_num):
            precision, recall, fscore = cal_precision_recall(graph_lable_list[t], mutual_info_graph_list[t])
            print('mutual_graph_' + str(t) + ': precision = ' + str(precision) + ', recall = ' + str(recall) + ', fscore = ' + str(fscore))
            print('mutual_graph_' + str(t) + ': precision = ' + str(precision) + ', recall = ' + str(recall) + ', fscore = ' + str(fscore), file=f_log, flush=True)
        print('--------------------------------------------------------------------')
        print('--------------------------------------------------------------------', file=f_log, flush=True)

    if initial_graph_use_AllDataMI:
        old_graph_list = []
        use_record_data = record_data_list[0].copy()
        for ri in range(1, len(record_data_list)):
            use_record_data = np.concatenate([use_record_data, record_data_list[ri]], axis=0)
        mutual_infos, mutual_info_threshold, mutual_info_graph = construct_from_records(use_record_data, mutual_info_mode, more_zero_mutual_info, small_center_select, parents_limit)

        for t in range(task_num):
            old_graph_list.append(mutual_info_graph.copy())
        if print_parents_info:
            print('mutual find edge sum = ' + str(mutual_info_graph.sum()))
            print('mutual find edge sum = ' + str(mutual_info_graph.sum()), file=f_log, flush=True)
            for t in range(task_num):
                precision, recall, fscore = cal_precision_recall(graph_lable_list[t], mutual_info_graph)
                print('mutual_graph_' + str(t) + ': precision = ' + str(precision) + ', recall = ' + str(recall) + ', fscore = ' + str(fscore))
                print('mutual_graph_' + str(t) + ': precision = ' + str(precision) + ', recall = ' + str(recall) + ', fscore = ' + str(fscore), file=f_log, flush=True)


    for ti in range(task_num):
        print("------------------------------ graph " + str(ti) + " ------------------------------")
        print("------------------------------ graph " + str(ti) + " ------------------------------", file=f_log, flush=True)
        mutual_info = mutual_info_list[ti]
        mutual_threshold = mutual_info_threshold_list[ti] * mutual_info_coef
        for i in range(node_num):
            i_mutual_info_index_mi = np.concatenate(
                [np.array([j for j in range(node_num)]).reshape([-1, 1]), mutual_info[:, i:i + 1]], axis=1)
            i_mutual_info_index_mi[i, 1] = -1
            i_mutual_info_index_mi = i_mutual_info_index_mi[(i_mutual_info_index_mi[:, 1] * -1).argsort()]
            i_big_mutual_info_index = i_mutual_info_index_mi[
                np.where(i_mutual_info_index_mi[:, 1] > mutual_threshold)[0], 0]
            i_big_mutual_info_index = i_big_mutual_info_index[:int(parents_limit_coef * parents_limit)].astype(int)
            i_big_mutual_info_index.sort()

            if len(i_big_mutual_info_index) > 0:
                max_parents_num = min(len(i_big_mutual_info_index), parents_limit)
                for pi in range(1, max_parents_num + 1):
                    all_pi_parents_combination = k_combine2(i_big_mutual_info_index, pi)
                    for ai_parents in all_pi_parents_combination:
                        ai_parents = np.sort(ai_parents).astype(int)
                        left_score = cal_parents_left_score(record_data_list[ti][:, i:i + 1], record_data_list[ti][:, ai_parents])
                        left_scores[ti][i].append([ai_parents, left_score, 0])

    new_graph_list = []
    iteration_num = 0
    while True:
        iteration_num = iteration_num + 1
        print("========================================== iter " + str(iteration_num) + " ==================================================")
        print("========================================== iter " + str(iteration_num) + " ==================================================", file=f_log, flush=True)
        for ti in range(task_num):
            use_record_data = record_data_list[ti]
            new_graph_ti = np.zeros([node_num, node_num])
            edge_prob = np.zeros([node_num, node_num])
            for tj in range(task_num):
                if tj != ti:
                    if iteration_num % 3 != -1:
                        edge_prob = edge_prob + old_graph_list[tj]
                    else:
                        edge_prob = edge_prob + graph_lable_list[tj]

            edge_prob = (edge_prob + alpha * 1.0) / (task_num - 1.0 + 2.0 * alpha)

            for i in range(node_num):
                for parents_left_right in left_scores[ti][i]:
                    ai_parents = parents_left_right[0]
                    left_score = parents_left_right[1]
                    right_score = cal_parents_right_score(i, ai_parents, edge_prob, task_num, alpha)
                    parents_left_right[2] = left_score + penalize_coef * right_score

                    continue
                left_scores[ti][i] = sorted(left_scores[ti][i], key=lambda x: -x[2])
                i_parents = set()
                for parents_left_right in left_scores[ti][i]:
                    i_parents = i_parents | set(parents_left_right[0])

                    if len(i_parents) >= parents_limit / 2:
                        break
                i_parents = np.sort(list(i_parents))
                if len(i_parents) > 0:
                    new_graph_ti[i_parents, i] = 1
            new_graph_list.append(new_graph_ti)
            if print_parents_info:
                precision, recall, fscore = cal_precision_recall(graph_lable_list[ti], new_graph_ti)
                print('mutual_graph_' + str(ti) + ': precision = ' + str(precision) + ', recall = ' + str(recall) + ', fscore = ' + str(fscore))
                print('mutual_graph_' + str(ti) + ': precision = ' + str(precision) + ', recall = ' + str(recall) + ', fscore = ' + str(fscore), file=f_log, flush=True)

        max_different_num = 0
        for ti in range(task_num):
            ti_different_num = (new_graph_list[ti] != old_graph_list[ti]).sum()
            max_different_num = max(max_different_num, ti_different_num)
        if print_parents_info:
            print('iteration_num = ' + str(iteration_num) + ', max_different_num = ' + str(max_different_num))
            print('iteration_num = ' + str(iteration_num) + ', max_different_num = ' + str(max_different_num), file=f_log, flush=True)
        old_graph_list = [new_graph.copy() for new_graph in new_graph_list]
        new_graph_list = []
        if max_different_num < 20 or iteration_num > 5:
            break

        print('===================================== Finished ==========================================')
        print('===================================== Finished ==========================================', file=f_log, flush=True)

    return mutual_info_graph_list, old_graph_list











# a running example
more_zero_mutual_info = True
small_center_select = 'mean'
mutual_info_coef = 1
parents_limit_coef = 3
print_parents_info = True
initial_graph_use_AllDataMI = False
initial_graph_use_topKMI = False
alpha = 0.0001
penalize_coef = 5

node_num = 2000
graph_record_data_path = './test'
record_num = 150
parents_limit = int(np.log2(record_num))
task_num = 5
mode = 2


# log路径
log_path = "./test_results.txt"

print(log_path)
f_log = open(log_path, "a+")

record_data_list = []
graph_label_list = []
for ti in range(task_num):
    graph_path = graph_record_data_path + '/test_network_' + str(ti) + '.txt'
    record_path = graph_record_data_path + '/test_record_' + str(ti) + '.txt'

    record_data, graph_label = read_records_and_graph(record_path, graph_path)
    record_data = record_data[:record_num, :]

    graph_label_list.append(graph_label)
    record_data_list.append(record_data)


begin = time.time()
mutual_info_graph_list, construct_graph_list = construct_multi_diffusion_network(
    record_data_list=record_data_list, graph_lable_list=graph_label_list,
    mutual_info_mode=mode, more_zero_mutual_info=more_zero_mutual_info, small_center_select=small_center_select,
    mutual_info_coef=mutual_info_coef,
    initial_graph_use_AllDataMI=initial_graph_use_AllDataMI, initial_graph_use_topKMI=initial_graph_use_topKMI, alpha=alpha,
    parents_limit_coef=parents_limit_coef, parents_limit=parents_limit, penalize_coef=penalize_coef, print_parents_info=print_parents_info,
     f_log=f_log
)
end = time.time()

precisions = np.zeros(task_num)
recalls = np.zeros(task_num)
f_scores = np.zeros(task_num)
for ti in range(len(graph_label_list)):

    mutual_precision, mutual_recall, mutual_fscore = cal_precision_recall(graph_label_list[ti], mutual_info_graph_list[ti])
    construct_precision, construct_recall, construct_fscore = cal_precision_recall(graph_label_list[ti], construct_graph_list[ti])
    precisions[ti] = construct_precision
    recalls[ti] = construct_recall
    f_scores[ti] = construct_fscore

    print('construct_graph_' + str(ti) + ': precision = ' + str(construct_precision) + ', recall = ' + str(construct_recall) + ', fscore = ' + str(construct_fscore))
    print('construct_graph_' + str(ti) + ': precision = ' + str(construct_precision) + ', recall = ' + str(construct_recall) + ', fscore = ' + str(construct_fscore), file=f_log, flush=True)

print('=========================================================================')
print('=========================================================================', file=f_log, flush=True)

print("{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(
    precisions.mean(), precisions.std(), recalls.mean(), recalls.std(), f_scores.mean(), f_scores.std(), end - begin),
)
print("{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(
    precisions.mean(), precisions.std(), recalls.mean(), recalls.std(), f_scores.mean(), f_scores.std(), end - begin),
    file=f_log, flush=True)

