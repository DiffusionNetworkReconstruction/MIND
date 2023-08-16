import time
import random
import numpy as np
import networkx as nx


class DataGenerator:
    def __init__(self, network_size, ori_degree, final_degree, type):
        self.network_size = network_size   
        self.ori_degree = ori_degree       
        self.final_degree = final_degree   
        self.type = type  

    
    def IC_model1(self, network, initial_nodes, infect_prob, f):
        nodes_num = self.network_size
        diffusion_result = np.zeros((1, nodes_num))  

        diffusion_result[0, initial_nodes] = 1
        last_node_infected = np.zeros(nodes_num)  

        time = 0
        inf_str = ""
        while True:
            new_infect = np.where((diffusion_result[0] - last_node_infected) == 1)[0]
            if new_infect.size == 0:
                break
            else:
                last_node_infected = diffusion_result[0].copy()
                for j in new_infect:
                    
                    inf_str = inf_str + str(j) + "," + str(time) + ","
                    j_neighbor = np.where(network[j] == 1)[0]
                    for k in j_neighbor:
                        if diffusion_result[0, k] == 0:
                            randP = random.random()
                            if randP < infect_prob:
                                diffusion_result[0, k] = 1
            time += 1
        f.write(inf_str[:-1])

        return diffusion_result

    def generate_records(self, network, initial_rate, infect_prob, beta, record_NR_save_path):
        f = open(record_NR_save_path, "a+")
        nodes_num = self.network_size
        sel_list = list(range(nodes_num))        
        initial_nodes_num = int(nodes_num * initial_rate)
        
        records_array = np.zeros((beta, nodes_num))
        for i in range(beta):
            initial_nodes = random.sample(sel_list, initial_nodes_num)
            records_array[i] = self.IC_model1(network, initial_nodes, infect_prob, f)
            f.write("\n")

        return records_array
   
    def generate_networks(self, task_num):
        node_num = self.network_size

        path = self.type
        if self.type == "Arti":
            path = str(node_num) + '_' + str(self.ori_degree) + '_to_' + str(self.final_degree)

        
        base_dir = './data/' + path

        
        graph_path = base_dir + '/' + "network.dat"
        G = nx.read_adjlist(graph_path, create_using=nx.DiGraph)
        ground_network = np.zeros((node_num, node_num))
        for (u, v) in G.edges():
            ground_network[int(u) - 1][int(v) - 1] = 1
            ground_network[int(v) - 1][int(u) - 1] = 1

        
        hold_rate = 2 / 3
        random.seed(int(time.time()))
        for t in range(task_num):
            new_network = ground_network.copy()
            
            for i in range(node_num):
                for j in range(i, node_num):
                    if new_network[i][j] == 1:
                        if random.random() > hold_rate:
                            new_network[i][j] = 0
                            new_network[j][i] = 0
            
            if self.type == "Arti":
                graph_save_path = base_dir + '/' + "network_" + str(node_num) + '_' + str(self.final_degree) + '_' + str(t) + ".dat"
                record_save_path = base_dir + '/' + "record_" + str(node_num) + '_' + str(self.final_degree) + '_' + str(t) + ".dat"
            else:
                graph_save_path = base_dir + '/' + "network_" + self.type + str(t) + ".dat"
                record_save_path = base_dir + '/' + "record_" + self.type + str(t) + ".dat"

            
            if self.type == "Arti":
                graph_NR_save_path = base_dir + '/' + "network_NR_" + str(node_num) + '_' + str(self.final_degree) + '_' + str(t) + ".txt"
                record_NR_save_path = base_dir + '/' + "record_NR_" + str(node_num) + '_' + str(self.final_degree) + '_' + str(t) + ".txt"
            else:
                graph_NR_save_path = base_dir + '/' + "network_NR_" + self.type + str(t) + ".txt"
                record_NR_save_path = base_dir + '/' + "record_NR_" + self.type + str(t) + ".txt"

            with open(graph_NR_save_path, "w+") as f:
                for i in range(node_num):
                    f.write("{},{}\n".format(i, i))
                f.write("\n")

            with open(record_NR_save_path, "w+") as f:
                for i in range(node_num):
                    f.write("{},{}\n".format(i, i))
                f.write("\n")
       
            record_array = self.generate_records(new_network, 0.15, 0.3, 300, record_NR_save_path).astype(int)

            f1 = open(graph_save_path, "w+")
            f2 = open(graph_NR_save_path, "a+")
            for i in range(node_num):
                for j in range(node_num):
                    if new_network[i][j] == 1:
                        f1.write("{}\t{}\n".format(i, j))
                        f2.write("{},{},{}\n".format(i, j, 0.3))
            with open(record_save_path, "w+") as f:
                for i in range(record_array.shape[0]):
                    f.write("\t".join(map(str, record_array[i])) + "\n")

class contro_sim_DataGenerator:
    def __init__(self, network_size, ori_degree, final_degree, type, similarity):
        self.network_size = network_size   
        self.ori_degree = ori_degree       
        self.final_degree = final_degree   
        self.type = type  
        self.similarity = similarity    

    def IC_model1(self, network, initial_nodes, infect_prob, f):
        nodes_num = self.network_size
        diffusion_result = np.zeros((1, nodes_num))  

        diffusion_result[0, initial_nodes] = 1
        last_node_infected = np.zeros(nodes_num)  

        time = 0
        inf_str = ""
        while True:
            new_infect = np.where((diffusion_result[0] - last_node_infected) == 1)[0]
            if new_infect.size == 0:
                break
            else:
                last_node_infected = diffusion_result[0].copy()
                for j in new_infect:
                    
                    inf_str = inf_str + str(j) + "," + str(time) + ","
                    j_neighbor = np.where(network[j] == 1)[0]
                    for k in j_neighbor:
                        if diffusion_result[0, k] == 0:
                            randP = random.random()
                            if randP < infect_prob:
                                diffusion_result[0, k] = 1
            time += 1
        f.write(inf_str[:-1])

        return diffusion_result

    def generate_records(self, network, initial_rate, infect_prob, beta, record_NR_save_path):
        f = open(record_NR_save_path, "a+")
        nodes_num = self.network_size
        
        sel_list = list(range(nodes_num))        
        initial_nodes_num = int(nodes_num * initial_rate)
       
        records_array = np.zeros((beta, nodes_num))
        for i in range(beta):
            initial_nodes = random.sample(sel_list, initial_nodes_num)
            records_array[i] = self.IC_model1(network, initial_nodes, infect_prob, f)
            f.write("\n")

        return records_array

    
    def generate_networks(self, task_num):
        node_num = self.network_size

        path = self.type
        if self.type == "Arti":
            path = str(node_num) + '_' + str(self.ori_degree) + '_to_' + str(self.final_degree)
        
        base_dir = 'data/' + path
        
        graph_path = base_dir + '/' + "network.dat"
        G = nx.read_adjlist(graph_path, create_using=nx.DiGraph)
        ground_network = np.zeros((node_num, node_num))
        for (u, v) in G.edges():
            ground_network[int(u) - 1][int(v) - 1] = 1
            ground_network[int(v) - 1][int(u) - 1] = 1

        network0 = np.zeros((node_num, node_num))
        random.seed(int(time.time()))
        # generate fixed edge according the similarity
        for x in range(node_num):
            for y in range(x,node_num):
                if ground_network[x][y] == 1:
                    if random.random() < (self.similarity/2):
                        network0[x][y] = 1
                        network0[y][x] = 1
        
        random.seed(int(time.time()))
        hold_rate = self.similarity/(2.0-self.similarity)
        for t in range(task_num):
            new_network = ground_network.copy()
            
            for i in range(node_num):
                for j in range(i, node_num):
                    if new_network[i][j] == 1 and network0[i][j] != 1:
                        if random.random() > hold_rate:
                            new_network[i][j] = 0
                            new_network[j][i] = 0

            
            if self.type == "Arti":
                graph_save_path = base_dir+'_contro_sim' + '/' + "network_" + str(node_num) + '_sim_' + str(self.similarity) + '_' + str(t) + ".dat"
                record_save_path = base_dir+'_contro_sim' + '/' + "record_" + str(node_num) + '_sim_' + str(self.similarity) + '_' + str(t) + ".dat"
            else:
                graph_save_path = base_dir+'_contro_sim' + '/' + "network_" + self.type + '_sim_' + str(self.similarity) + '_'+ str(t) + ".dat"
                record_save_path = base_dir+'_contro_sim' + '/' + "record_" + self.type + '_sim_' + str(self.similarity) + '_' + str(t) + ".dat"

           
            if self.type == "Arti":
                graph_NR_save_path = base_dir+ '/' + "network_NR_" + str(node_num) + '_sim_' + str(self.similarity) + '_' + str(t) + ".txt"
                record_NR_save_path = base_dir + '/' + "record_NR_" + str(node_num) + '_sim_' + str(self.similarity) + '_' + str(t) + ".txt"
            else:
                graph_NR_save_path = base_dir+ '/' + "network_NR_" + self.type + '_sim_' + str(self.similarity)+ '_' + str(t) + ".txt"
                record_NR_save_path = base_dir + '/' + "record_NR_" + self.type + '_sim_' + str(self.similarity)+ '_' + str(t) + ".txt"

            with open(graph_NR_save_path, "w+") as f:
                for i in range(node_num):
                    f.write("{},{}\n".format(i, i))
                f.write("\n")

            with open(record_NR_save_path, "w+") as f:
                for i in range(node_num):
                    f.write("{},{}\n".format(i, i))
                f.write("\n")

          
            record_array = self.generate_records(new_network, 0.15, 0.3, 150, record_NR_save_path).astype(int)

            f1 = open(graph_save_path, "w+")
            f2 = open(graph_NR_save_path, "a+")
            for i in range(node_num):
                for j in range(node_num):
                    if new_network[i][j] == 1:
                        f1.write("{}\t{}\n".format(i, j))
                        f2.write("{},{},{}\n".format(i, j, 0.3))
            with open(record_save_path, "w+") as f:
                for i in range(record_array.shape[0]):
                    f.write("\t".join(map(str, record_array[i])) + "\n")


#running example
generator = DataGenerator(750, 6, 4, "DUNF")
generator.generate_networks(5)

# generator = contro_sim_DataGenerator(2000, 6, 4, "Arti",0.6)
# generator.generate_networks(5)
