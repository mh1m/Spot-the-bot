import ray
import copy
import math
import scipy
import psutil
import pickle
import itertools
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.spatial import Delaunay
from joblib import Parallel, delayed


def create_vertices_with_sentences(corpus: list, k: int) -> dict:
    '''
    Create vertices from vectorized corpus splitted by sentences
    
    Parameters
    ----------
    corpus : list
    Vectorized corpus = [document_1, ..., document_i ..., document_n], where
    text_i = [{'document_index': i,
               'sentence_index': 0,
               'sentence_text': {word: vector, 
                                 word: vector, ...}},
              ...,
              {'document_index': i,
               'sentence_index': m_i,
               'sentence_text': {word: vector, 
                                 word: vector, ...}}]
    So, we get n documents and m_i sentences for each document
    
    k : int
    Vector length
    
    Returns
    -------
    vertices : dict
    Returns dict = {word: Node(vector,
                               empty set, 
                               set of words that occur in the same sentence throughout the corpus)}
    '''
    vertices = dict()
    for text in tqdm(corpus):
        for sentence in text:
            for word, vector in sentence['sentence_text'].items():
                word_neighbors = set(sentence['sentence_text'].keys()).difference(set([word]))
                if word in vertices.keys():
                    vertices[word].sentence_neighbors.update(word_neighbors)
                else:
                    vertices[word] = Node(vector=np.array(vector[:k]),
                                          neighbors=set(),
                                          sentence_neighbors=word_neighbors)
    return vertices


'''
Node and Graph
'''

class Node:
    def __init__(self, vector: np.ndarray, neighbors: set, sentence_neighbors: set):
        self.__vector = copy.deepcopy(vector)
        self.__neighbors = copy.deepcopy(neighbors)
        self.__sentence_neighbors = copy.deepcopy(sentence_neighbors)
    
    @property
    def vector(self):
        return self.__vector
    
    @vector.setter
    def vector(self, vector: np.ndarray):
        self.__vector = copy.deepcopy(vector)
            
    @property
    def neighbors(self):
        return self.__neighbors
    
    @neighbors.setter
    def neighbors(self, neighbors: set):
        self.__neighbors = copy.deepcopy(neighbors)
        
    @property
    def sentence_neighbors(self):
        return self.__sentence_neighbors
    
    @sentence_neighbors.setter
    def sentence_neighbors(self, sentence_neighbors: set):
        self.__sentence_neighbors = sentence_neighbors

class Graph:
    def __init__(self, vertices: dict):
        self.__vertices = copy.deepcopy(vertices)
    
    @property
    def vertices(self):
        return self.__vertices
    
    @vertices.setter
    def vertices(self, vertices: dict):
        self.__vertices = copy.deepcopy(vertices)
        
    def get_words(self) -> list:
        return list(self.vertices.keys())
    
    def get_vectors(self) -> list:
        vectors = list()
        for node in self.vertices.values():
            vectors.append(node.vector.tolist())
        return vectors
    
    def get_edges(self) -> set:
        edges = set()
        for word, node in self.vertices.items():
            for neighbor in node.neighbors:
                if (word, neighbor) in edges or (neighbor, word) in edges:
                    continue
                edges.add((word, neighbor))
        return edges
    
    def get_other_words(self, *words) -> set:
        return set(self.get_words()).difference(set(words))
    
    def add_edge(self, first_word: str, second_word: str):
        self.vertices[first_word].neighbors.add(second_word)
        self.vertices[second_word].neighbors.add(first_word)
    
    def delete_edge(self, first_word: str, second_word: str):
        self.vertices[first_word].neighbors.difference_update(set([second_word]))
        self.vertices[second_word].neighbors.difference_update(set([first_word]))
    
    def reset_graph_neighbors(self):
        for node in self.vertices.values():
            node.neighbors.clear()
    
    def euclid_distance(self, first_word: str, second_word: str) -> np.float64:
        first_vertex = self.vertices[first_word].vector
        second_vertex = self.vertices[second_word].vector
        return np.linalg.norm(first_vertex - second_vertex)
    
    def euclid_distance_to_vector(self, word: str, vector: np.ndarray) -> np.float64:
        word_vector = self.vertices[word].vector
        return np.linalg.norm(word_vector - vector)
    
    def get_knn(self, word: str, k: int) -> list:
        distance_dict = dict()
        for other_word in self.get_other_words(word):
            distance_dict[other_word] = self.euclid_distance(word, other_word)
        lambda_ = lambda item: item[1]
        words = [key for key, value in sorted(distance_dict.items(), key=lambda_)]
        return words[:k]
    
    def get_sphere_radius(self, word: str) -> float:
        nearest_neighbor = self.get_knn(word, 1)[0]
        return self.euclid_distance(word, nearest_neighbor)
    
    def in_sphere_check(self, first_word: str, second_word: str) -> bool:
        nearest_neighbor = self.get_knn(first_word, 1)
        radius = get_sphere_radius(first_word)
        distance = self.euclid_distance(first_word, second_word)
        if distance > radius:
            return False
        return True
    
    def update_graph_by_sentence_neighbors(self):
        for word in tqdm(self.vertices.keys()):
            self.vertices[word].neighbors.intersection_update(self.vertices[word].sentence_neighbors)

    def plot_graph(self,
               figsize=(60, 60),
               dpi=120,
               lines_color='royalblue',
               points_type='o',
               points_color='forestgreen',
               xytext=(10,10),
               textcoords='offset points'):
        plt.figure(figsize=figsize, dpi=dpi)
        for edge in self.get_edges():
            first_word, second_word = edge
            first_vector = self.vertices[first_word].vector
            second_vector = self.vertices[second_word].vector
            edge_vectors = np.concatenate([[first_vector], [second_vector]])
            plt.plot(edge_vectors[:, 0], edge_vectors[:, 1], color=lines_color)
            plt.annotate(first_word, (edge_vectors[0, 0], edge_vectors[0, 1]), xytext=xytext, textcoords=textcoords)
            plt.annotate(second_word, (edge_vectors[1, 0], edge_vectors[1, 1]), xytext=xytext, textcoords=textcoords)
            plt.plot(edge_vectors[:, 0], edge_vectors[:, 1], points_type, color=points_color)
        plt.show()


'''
Параллелизация графов
'''


@ray.remote
class Parallel(Graph):
    def __init__(self, vertices: dict):
        super().__init__(vertices)
        self.__parallel = set()
        
    @property
    def parallel(self):
        return self.__parallel
    
    @parallel.setter
    def parallel(self, parallel: set):
        self.__parallel = copy.deepcopy(parallel)
        
    def add_parallel_word(self, word: str):
        self.parallel.add(word)
        
    def add_parallel_triangle(self, triangle: set):
        self.parallel.add(tuple(triangle))

    def parallel_eball_graph(self, epsilon: float) -> dict:
        for first_word in tqdm(self.parallel):
            other_words = self.get_other_words(first_word)
            for second_word in other_words:
                distance = self.euclid_distance(first_word, second_word)
                if distance < epsilon:
                    self.add_edge(first_word, second_word)
        return self.vertices
    
    def is_neighbor_in_edge_sphere_gromov(self, edge: tuple, neighbor: str) -> bool:
        edge_length = self.euclid_distance(edge[0], edge[1])
        first_to_neighbor = self.euclid_distance(edge[0], neighbor)
        second_to_neighbor = self.euclid_distance(edge[1], neighbor)
        if edge_length < math.sqrt(first_to_neighbor ** 2 + second_to_neighbor ** 2):
            return False
        return True
    
    def is_neighbor_in_edge_sphere(self, edge: tuple, neighbor: str) -> bool:
        sphere_center = (self.vertices[edge[0]].vector + self.vertices[edge[1]].vector) / 2
        radius = self.euclid_distance_to_vector(edge[0], sphere_center)
        if self.euclid_distance_to_vector(neighbor, sphere_center) < radius:
            return True
        return False
    
    def parallel_gabriel_graph(self) -> dict:
        for triangle in tqdm(self.parallel):
            edges = list(itertools.combinations(triangle, 2))
            for edge in edges:
                neighbors = set(triangle).difference(set(edge))
                for neighbor in neighbors:
                    if self.is_neighbor_in_edge_sphere(edge, neighbor):
                        self.delete_edge(edge[0], edge[1])
        return self.vertices
    
    def parallel_rn_graph(self) -> dict:
        for word in tqdm(self.parallel):
            neighbors = copy.deepcopy(self.vertices[word].neighbors)
            for neighbor in neighbors:
                for other_word in self.get_other_words(word, neighbor):
                    word_to_neighbor_distance = self.euclid_distance(word, neighbor)
                    word_to_other_word_distance = self.euclid_distance(word, other_word)
                    neighbor_to_other_word_distance = self.euclid_distance(neighbor, other_word)
                    if max(word_to_other_word_distance, neighbor_to_other_word_distance) <= word_to_neighbor_distance:
                        self.delete_edge(word, neighbor)
        return self.vertices
    
    def parallel_influence_graph(self) -> dict:
        for first_word in tqdm(self.parallel):
            first_radius = self.get_sphere_radius(first_word)
            for second_word in self.get_other_words(first_word):
                second_radius = self.get_sphere_radius(second_word)
                distance = self.euclid_distance(first_word, second_word)
                if distance <= first_radius + second_radius:
                    self.add_edge(first_word, second_word)
        return self.vertices
    
    def parallel_nn_graph(self, k: int) -> dict:
        for word in tqdm(self.parallel):
            knn_list = self.get_knn(word, k)
            for neighbor in knn_list:
                self.add_edge(word, neighbor)
        return self.vertices


'''
Граф ε-окружности (ε-ball)
'''
    
    
class EBall(Graph):
    def __init__(self, vertices: dict):
        super().__init__(vertices)
        
    def create_eball_graph(self, epsilon: float):
        self.reset_graph_neighbors()
        for first_word in tqdm(self.get_words()):
            other_words = self.get_other_words(first_word)
            for second_word in other_words:
                distance = self.euclid_distance(first_word, second_word)
                if distance < epsilon:
                    self.add_edge(first_word, second_word)
    
    def create_parallel_eball_graph(self, epsilon: float):
        self.reset_graph_neighbors()
        num_cpus = psutil.cpu_count(logical=False)
        ray.init(num_cpus=num_cpus)
        streaming_actors = [Parallel.remote(self.vertices) for _ in range(num_cpus)]
        for index, word in enumerate(self.get_words()):
            streaming_actors[index % num_cpus].add_parallel_word.remote(word)
        results = ray.get([
            actor.parallel_eball_graph.remote(epsilon) for actor in streaming_actors
        ])
        ray.shutdown()
        for parallel_vertices in results:
            for word in self.get_words():
                self.vertices[word].neighbors.update(parallel_vertices[word].neighbors)


'''
Триангуляция Делоне (DT) и Граф Габриэля (GG)
'''


class GG(Graph):
    def __init__(self,
                 vertices: dict,
                 triangles: list = list(),
                 delaunay: scipy.spatial.qhull.Delaunay = None):
        super().__init__(vertices)
        self.__triangles = copy.deepcopy(triangles)
        self.__delaunay = copy.deepcopy(delaunay)
    
    @property
    def triangles(self):
        return self.__triangles
    
    @triangles.setter
    def triangles(self, triangles: list):
        self.__triangles = copy.deepcopy(triangles)
    
    @property
    def delaunay(self):
        return self.__delaunay
    
    @delaunay.setter
    def delaunay(self, delaunay: scipy.spatial.qhull.Delaunay):
        self.__delaunay = copy.deepcopy(delaunay)
    
    def create_delaunay_graph(self):
        self.reset_graph_neighbors()
        vectors = self.get_vectors()
        words = self.get_words()
        word_num_dict = {word: num for word, num in enumerate(words)}
        self.delaunay = Delaunay(np.array(vectors))
        delaunay_graph = self.delaunay.simplices.tolist()
        for triangle in tqdm(delaunay_graph):
            triangle_words = set(map(word_num_dict.get, triangle))
            self.triangles.append(triangle_words)
            for word in triangle_words:
                new_neighbors = triangle_words.difference(set([word]))
                self.vertices[word].neighbors.update(new_neighbors)
    
    def is_neighbor_in_edge_sphere_gromov(self, edge: tuple, neighbor: str) -> bool:
        edge_length = self.euclid_distance(edge[0], edge[1])
        first_to_neighbor = self.euclid_distance(edge[0], neighbor)
        second_to_neighbor = self.euclid_distance(edge[1], neighbor)
        if edge_length < math.sqrt(first_to_neighbor ** 2 + second_to_neighbor ** 2):
            return False
        return True
    
    def is_neighbor_in_edge_sphere(self, edge: tuple, neighbor: str) -> bool:
        sphere_center = (self.vertices[edge[0]].vector + self.vertices[edge[1]].vector) / 2
        radius = self.euclid_distance_to_vector(edge[0], sphere_center)
        if self.euclid_distance_to_vector(neighbor, sphere_center) < radius:
            return True
        return False
    
    def create_gabriel_graph(self):
        print("Delaunay start...")
        self.create_delaunay_graph()
        print("Delaunay done!")
        for triangle in tqdm(self.triangles):
            edges = list(itertools.combinations(triangle, 2))
            for edge in edges:
                neighbors = triangle.difference(set(edge))
                for neighbor in neighbors:
                    if self.is_neighbor_in_edge_sphere(edge, neighbor):
                        self.delete_edge(edge[0], edge[1])
    
    def create_parallel_gabriel_graph(self):
        print("Delaunay start...")
        self.create_delaunay_graph()
        print("Delaunay done!")
        num_cpus = psutil.cpu_count(logical=False)
        ray.init(num_cpus=num_cpus)
        streaming_actors = [Parallel.remote(self.vertices) for _ in range(num_cpus)]
        for index, triangle in enumerate(self.triangles):
            streaming_actors[index % num_cpus].add_parallel_triangle.remote(triangle)
        results = ray.get([
            actor.parallel_gabriel_graph.remote() for actor in streaming_actors
        ])
        ray.shutdown()
        for parallel_vertices in results:
            for word in self.get_words():
                self.vertices[word].neighbors.intersection_update(parallel_vertices[word].neighbors)


'''
Граф относительного соседства (RNG)
'''


class RNG(GG):
    def __init__(self,
                 vertices: dict,
                 triangles: list = list(),
                 delaunay: scipy.spatial.qhull.Delaunay = None):
        super().__init__(vertices, triangles, delaunay)
    
    def create_rn_graph(self):
        self.create_gabriel_graph()
        for word in tqdm(self.vertices.keys()):
            neighbors = copy.deepcopy(self.vertices[word].neighbors)
            for neighbor in neighbors:
                for other_word in self.get_other_words(word, neighbor):
                    word_to_neighbor_distance = self.euclid_distance(word, neighbor)
                    word_to_other_word_distance = self.euclid_distance(word, other_word)
                    neighbor_to_other_word_distance = self.euclid_distance(neighbor, other_word)
                    if max(word_to_other_word_distance, neighbor_to_other_word_distance) <= word_to_neighbor_distance:
                        self.delete_edge(word, neighbor)
                        
    def create_parallel_rn_graph(self):
        print("Gabriel start...")
        self.create_parallel_gabriel_graph()
        print("Gabriel done!")
        num_cpus = psutil.cpu_count(logical=False)
        ray.init(num_cpus=num_cpus)
        streaming_actors = [Parallel.remote(self.vertices) for _ in range(num_cpus)]
        for index, word in enumerate(self.vertices):
            streaming_actors[index % num_cpus].add_parallel_word.remote(word)
        results = ray.get([
            actor.parallel_rn_graph.remote() for actor in streaming_actors
        ])
        ray.shutdown()
        for parallel_vertices in results:
            for word in self.get_words():
                self.vertices[word].neighbors.intersection_update(parallel_vertices[word].neighbors)


'''
Граф влияния (IG)
'''


class IG(Graph):
    def __init__(self, vertices: dict):
        super().__init__(vertices)
    
    def create_influence_graph(self):
        self.reset_graph_neighbors()
        for first_word in tqdm(self.vertices.keys()):
            other_words = self.get_other_words(first_word)
            first_radius = self.get_sphere_radius(first_word)
            for second_word in other_words:
                second_radius = self.get_sphere_radius(second_word)
                distance = self.euclid_distance(first_word, second_word)
                if distance <= first_radius + second_radius:
                    self.add_edge(first_word, second_word)

    def create_parallel_influence_graph(self):
        self.reset_graph_neighbors()
        num_cpus = psutil.cpu_count(logical=False)
        ray.init(num_cpus=num_cpus)
        streaming_actors = [Parallel.remote(self.vertices) for _ in range(num_cpus)]
        for index, word in enumerate(self.vertices):
            streaming_actors[index % num_cpus].add_parallel_word.remote(word)
        results = ray.get([
            actor.parallel_influence_graph.remote() for actor in streaming_actors
        ])
        ray.shutdown()
        for parallel_vertices in results:
            for word in self.get_words():
                self.vertices[word].neighbors.update(parallel_vertices[word].neighbors)


'''
Граф k-ближайших соседей (NNG)
'''


class NNG(Graph):
    def __init__(self, vertices: dict):
        super().__init__(vertices)
    
    def create_nn_graph(self, k: int):
        self.reset_graph_neighbors()
        for word in tqdm(self.vertices.keys()):
            knn_list = self.get_knn(word, k)
            for neighbor in knn_list:
                self.add_edge(word, neighbor)
                
    def create_parallel_nn_graph(self, k: int):
        self.reset_graph_neighbors()
        num_cpus = psutil.cpu_count(logical=False)
        ray.init(num_cpus=num_cpus)
        streaming_actors = [Parallel.remote(self.vertices) for _ in range(num_cpus)]
        for index, word in enumerate(self.vertices):
            streaming_actors[index % num_cpus].add_parallel_word.remote(word)
        results = ray.get([
            actor.parallel_nn_graph.remote(k) for actor in streaming_actors
        ])
        ray.shutdown()
        for parallel_vertices in results:
            for word in self.get_words():
                self.vertices[word].neighbors.update(parallel_vertices[word].neighbors)
