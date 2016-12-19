import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
import numpy as np
import random
import math

def generate_pairs_points(dataset, number_of_pairs):
  pairs_of_point = np.zeros(number_of_pairs, dtype=[('first_image','i4'), ('second_image','i4')])

  for i in range(number_of_pairs):
    pairs_of_point[i] = (random.randint(0, dataset[0].shape[1]), random.randint(0, dataset[0].shape[1]))
  return pairs_of_point

class Plotter:

  def __init__(self, type, pairs_of_point):
    self.type = type
    self.pairs_of_point = pairs_of_point

  def plot(self, dataset1, dataset2):

    if self.type == "euclidian":
      distances = self.distances_euclidian(dataset1, dataset2, self.pairs_of_point)
      return self.plot_util(distances)
    elif self.type == "angular":
      distances = self.distances_angular(dataset1, dataset2, self.pairs_of_point)
      return self.plot_util(distances)
    # elif self.type == "cross":
    #   distances1 = self.distances_euclidian(dataset1, dataset2, self.pairs_of_point)
    #   distances2 = self.distances_angular(dataset1, dataset2, self.pairs_of_point)
    #   self.plot_util2(distances1, distances2)


  def distances_euclidian(self, dataset1, dataset2, number_pairs):
    euclidian_distances = np.zeros(len(self.pairs_of_point))
    for i_pair in range(len(self.pairs_of_point)):
      image_i = self.pairs_of_point[i_pair][0]
      image_j = self.pairs_of_point[i_pair][1]
      euclidian_distances[i_pair] = np.linalg.norm(dataset1[0][image_i]-dataset1[0][image_j])/np.linalg.norm(dataset2[0][image_i]-dataset2[0][image_j])
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

    plt.hist(distances, color='b', bins=distances.shape[0]//10)
    plt.legend()

    return (mean, std, skewness, kurtosis)

  def plot_util2(self, distances1, distances2):
    return


  def superplot(self, dataset_transform_targetdim_function, dataset_original, index_transform, index_target_dim, index_function, t_dict, td_dict, f_dict):

      plt.close('all')
      fig = plt.figure()
      xlim = [0.5, 1.5]
      ylim = [0, 50]

      if index_transform is not -1:

          td_dim = dataset_transform_targetdim_function.shape[1]
          f_dim = dataset_transform_targetdim_function.shape[2]
          i_graph = 0

          for j in range(td_dim):
              for k in range(f_dim):

                  ax1 = fig.add_subplot(td_dim*100 + f_dim*10 + (i_graph+1))

                  if i_graph < f_dim:
                    ax1.set_title('function = ' + str(f_dict[k]))

                  if (((i_graph+1) % f_dim) is 1) or (f_dim is 1):
                    ax1.set_ylabel('target dimension = ' + str(td_dict[j]))

                  ax1.set_xlim(xlim)
                  ax1.set_ylim(ylim)
                  ax1_metrics = self.plot(dataset_transform_targetdim_function[index_transform][j][k], dataset_original)
                  ax1.annotate(r'$\mu = ' + str(ax1_metrics[0]) + '$\n' +
                               r'$\sigma^2 = ' + str(ax1_metrics[1]) + '$\n' +
                               r'$Skewness = ' + str(ax1_metrics[2]) + '$\n' +
                               r'$Kurtosis = ' + str(ax1_metrics[3]) + '$\n',
                               xy=(1, 1), xycoords='axes fraction', fontsize=10,
                               xytext=(-5, -5), textcoords='offset points', ha='right', va='top')

                  i_graph += 1

      elif index_target_dim is not -1:

          t_dim = dataset_transform_targetdim_function.shape[0]
          f_dim = dataset_transform_targetdim_function.shape[2]
          i_graph = 0

          for i in range(t_dim):
              for k in range(f_dim):

                  ax1 = fig.add_subplot(t_dim * 100 + f_dim * 10 + (i_graph + 1))

                  if i_graph < f_dim:
                      ax1.set_title('function = ' + str(f_dict[k]))

                  if (((i_graph+1) % f_dim) is 1) or (f_dim is 1):
                      ax1.set_ylabel('transform = ' + str(t_dict[i]))

                  ax1.set_xlim(xlim)
                  ax1.set_ylim(ylim)
                  ax1_metrics = self.plot(dataset_transform_targetdim_function[i][index_target_dim][k], dataset_original)
                  ax1.annotate(r'$\mu = ' + str(ax1_metrics[0]) + '$\n' +
                               r'$\sigma^2 = ' + str(ax1_metrics[1]) + '$\n' +
                               r'$Skewness = ' + str(ax1_metrics[2]) + '$\n' +
                               r'$Kurtosis = ' + str(ax1_metrics[3]) + '$\n',
                               xy=(1, 1), xycoords='axes fraction', fontsize=10,
                               xytext=(-5, -5), textcoords='offset points', ha='right', va='top')

                  i_graph += 1

      elif index_function is not -1:

          t_dim = dataset_transform_targetdim_function.shape[0]
          td_dim = dataset_transform_targetdim_function.shape[1]
          i_graph = 0

          for i in range(t_dim):
              for j in range(td_dim):

                  ax1 = fig.add_subplot(t_dim * 100 + td_dim * 10 + (i_graph + 1))

                  if i_graph < td_dim:
                      ax1.set_title('target dimension = ' + str(td_dict[j]))

                  if (((i_graph+1) % td_dim) is 1) or (td_dim is 1):
                      ax1.set_ylabel('transform = ' + str(t_dict[i]))

                  ax1.set_xlim(xlim)
                  ax1.set_ylim(ylim)
                  ax1_metrics = self.plot(dataset_transform_targetdim_function[i][j][index_function],
                                          dataset_original)
                  ax1.annotate(r'$\mu = ' + str(ax1_metrics[0]) + '$\n' +
                               r'$\sigma^2 = ' + str(ax1_metrics[1]) + '$\n' +
                               r'$Skewness = ' + str(ax1_metrics[2]) + '$\n' +
                               r'$Kurtosis = ' + str(ax1_metrics[3]) + '$\n',
                               xy=(1, 1), xycoords='axes fraction', fontsize=10,
                               xytext=(-5, -5), textcoords='offset points', ha='right', va='top')

                  i_graph += 1

      # identity by default
      else:

          t_dim = dataset_transform_targetdim_function.shape[0]
          td_dim = dataset_transform_targetdim_function.shape[1]
          i_graph = 0

          for i in range(t_dim):
              for j in range(td_dim):

                  ax1 = fig.add_subplot(t_dim * 100 + td_dim * 10 + (i_graph + 1))

                  if i_graph < td_dim:
                      ax1.set_title('target dimension = ' + str(td_dict[j]))

                  if (((i_graph+1) % td_dim) is 1) or (td_dim is 1):
                      ax1.set_ylabel('transform = ' + str(t_dict[i]))

                  ax1.set_xlim(xlim)
                  ax1.set_ylim(ylim)
                  ax1_metrics = self.plot(dataset_transform_targetdim_function[i][j][0],
                                          dataset_original)
                  ax1.annotate(r'$\mu = ' + str(ax1_metrics[0]) + '$\n' +
                               r'$\sigma^2 = ' + str(ax1_metrics[1]) + '$\n' +
                               r'$Skewness = ' + str(ax1_metrics[2]) + '$\n' +
                               r'$Kurtosis = ' + str(ax1_metrics[3]) + '$\n',
                               xy=(1, 1), xycoords='axes fraction', fontsize=10,
                               xytext=(-5, -5), textcoords='offset points', ha='right', va='top')

                  i_graph += 1

      plt.show()