import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
import numpy as np
import random
import math


class Plotter:

  def __init__(self, type, number_pairs):
    self.type = type
    self.number_pairs = number_pairs

  def plot(self, dataset1, dataset2):

    if self.type == "euclidian":
      distances = self.distances_euclidian(dataset1, dataset2, self.number_pairs)
      self.plot_util(distances)
    elif self.type == "angular":
      distances = self.distances_angular(dataset1, dataset2, self.number_pairs)
      self.plot_util(distances)
    elif self.type == "cross":
      distances1 = self.distances_euclidian(dataset1, dataset2, self.number_pairs)
      distances2 = self.distances_angular(dataset1, dataset2, self.number_pairs)
      self.plot_util2(distances1, distances2)


  def distances_euclidian(self, dataset1, dataset2, number_pairs):

    euclidian_distances = np.zeros(self.number_pairs)

    for i_pair in range(self.number_pairs):
      i = random.randint(0, dataset1[0].shape[1])
      j = random.randint(0, dataset1[0].shape[1])
      euclidian_distances[i_pair] = np.linalg.norm(dataset1[0][i]-dataset1[0][j])/np.linalg.norm(dataset2[0][i]-dataset2[0][j])

    return euclidian_distances

  def distances_angular(self, dataset1, dataset2, number_pairs):

    angular_distances = np.zeros(self.number_pairs)

    for i_pair in range(self.number_pairs):
      i = random.randint(0, dataset1[0].shape[1])
      j = random.randint(0, dataset1[0].shape[1])
      angular_distances[i_pair] = self.compute_angular_distance(dataset1[0][i] - dataset1[0][j]) / self.compute_angular_distance(
        dataset2[0][i] - dataset2[0][j])

    return angular_distances

  def compute_angular_distance(self, u, v):

    cosuv = np.dot(u,v) / np.norm(u) / np.norm(v)
    sinuv = np.cross(u,v) / np.norm(u) / np.norm(v)

    angle_degree = math.acos(cosuv) * 180 / math.pi

    if sinuv > 0:
      return angle_degree
    else:
      return 360 - angle_degree

  def plot_util(self, distances):

    distances = distances[~np.isnan(distances)]


    mean = round(np.mean(distances), 2)
    std = round(np.std(distances), 2)
    skewness = round(sp.stats.skew(distances), 2)
    kurtosis = round(sp.stats.kurtosis(distances), 2)

    print(mean)
    print(std)
    print(skewness)
    print(kurtosis)

    plt.hist(distances, color='b', bins=distances.shape[0]//10)
    plt.legend()
    # plt.title("rho = -1, nb_steps = 10000 \n mean X: {}, std X: {} \n mean Z: {}, std Z: {} \n correlation: {}".format())
    # plt.xlim(0, 10000)
    # plt.ylim(-200, 200)
    plt.show()

  def plot_util2(self, distances1, distances2):
    plt.figure(figsize=(30, 10))

    plt.subplot(1, 3, 1)

    mean = round(np.mean(distances), 2)
    std = round(np.std(distances), 2)
    skewness = round(sp.stats.kurttosis(distances), 2)
    kurtosis = round(sp.stats.skew(distances), 2)

    plt.hist(distances1, color='b', label="X")
    plt.plot(distances2, color='g', label="Z")
    plt.axhline(y=0, xmin=0, xmax=1, linewidth=1, color='r')
    plt.legend()
    plt.title(
      "rho = -1, nb_steps = 10000 \n mean X: {}, std X: {} \n mean Z: {}, std Z: {} \n correlation: {}".format(m_x,
                                                                                                               std_x,
                                                                                                               m_z,
                                                                                                               std_z,
                                                                                                               cor))
    # plt.xlim(0, 10000)
    # plt.ylim(-200, 200)