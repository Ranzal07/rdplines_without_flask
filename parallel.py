import csv
import os
import time
import math
from rdp import rdp
import pandas as pd
import numpy as np
from typing import List
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import ttest_ind

executor = ThreadPoolExecutor(4)


def classic_rdp(points, eps):
    """
    Returns the classic rdp result
    """
    res = rdp(points, epsilon=eps)
    return res


def parallel_rdp(points, eps):
    """
    Returns the rdp result for every chunk
    """
    future = executor.submit(rdp, points, epsilon=eps)
    result = future.result()
    return result


def parallel_rdp_algorithm(data: List[List[float]], epsilon: float, chunk_size: int = None) -> List[List[float]]:
    """
    This is the function where the process of running all the chunks of the original line will happen in a parallel way through the use of multiprocessing's threadpoolexecutor
    """

    # Create a thread pool with four threads
    executor = ThreadPoolExecutor(4)

    # Divide the data into chunks of size chunk_size (if specified)
    if chunk_size:
        data_chunks = [data[i:i+chunk_size]
                       for i in range(0, len(data), chunk_size)]
    else:
        data_chunks = [data]

    # Submit each chunk to the thread pool
    futures = [executor.submit(parallel_rdp, chunk, epsilon)
               for chunk in data_chunks]

    # Wait for all threads to finish and collect the results
    results = [future.result() for future in futures]

    # Concatenate the results into a single list
    return [point for sublist in results for point in sublist]


def find_optimal_chunk_size(data):
    """
    Returns the number of chunks that the points will be divided to be processed in a parallel way
    """
    len_data = len(data)
    if len_data <= 100:
        return 2
    elif len_data > 100 and len_data <= 1000:
        return 4
    elif len_data > 1000 and len_data <= 10000:
        return 16
    elif len_data > 10000 and len_data <= 100000:
        return 32
    elif len_data > 100000:
        return 64


def calculate_epsilon(data):
    """
    Find an epsilon value for Ramer-Douglas-Peucker line simplification
    based on the median absolute deviation (MAD) of the data.
    """
    time_interval = 1  # determines the intensity of the change (1 = 100% maximum value for the best epsilon, 0.5 = 50%, 0.1 = 10%)
    mad = np.median(np.abs(data - np.median(data)))  # MAD
    data_range = np.max(data) - np.min(data)  # range of the data
    # multiplying the average of the MAD and range to the intensity of change to get the epsilon
    epsilon = (mad + data_range) / 2 * time_interval
    return epsilon


def get_file_size(directory):
    """
    Gets the file size of the csv files in the /originals and /simplified folders
    """
    # return the new file size in KB
    file_size = os.path.getsize(directory) / 1024
    return file_size


def save_points_to_csv(points, filename, columns):
    """
    Turns the new parallelized rdp points into a csv file and saves it in the /simplified directory
    """
    try:
        col_1 = [point[0] for point in points]
        col_2 = [point[1] for point in points]

        # save the parallelized points into a new csv file
        directory = 'simplified/' + filename.split('.')[0] + '(simplified).csv'
        df = pd.DataFrame({columns[0]: col_1, columns[1]: col_2})
        df.to_csv(directory, index=False)
        return 1
    except:
        print("\nAn error occured during file saving.\n")
        return 0


filename = input("Enter CSV file path (e.g. Datasets/Boeing.csv): ")

with open(filename, 'r') as file:
    # reads the directory of the file input of user
    reader = csv.reader(file)
    df = pd.read_csv(file, delimiter=',')

    # take the columns and rows
    cols = df.columns.values.tolist()
    first_row = df.iloc[:, 0]
    second_row = df.iloc[:, 1].astype(float)

    points = np.column_stack([range(len(first_row)), second_row])

    # get automatic epsilon value
    epsilon = calculate_epsilon([p[1] for p in points])
    # epsilon = 0

    # chunk size
    chunk = find_optimal_chunk_size(points)

    # parallel results
    parallelized_start_time = time.time()
    parallelized_points = parallel_rdp_algorithm(points, epsilon, chunk)
    parallelized_end_time = time.time()

    # file sizes
    save_file = save_points_to_csv(
        points=parallelized_points, filename=filename[14:], columns=cols)
    directory = 'simplified/' + \
        filename[14:].split('.')[0] + '(simplified).csv'

    original_file_size = 0
    parallel_file_size = 0

    if (save_file):
        original_file_size = get_file_size(filename)
        parallel_file_size = get_file_size(directory)

    # calculate mean
    original_mean = np.mean(np.mean(points, axis=0)[1])
    parallelized_mean = np.mean(np.mean(parallelized_points, axis=0)[1])

    # calculate standard deviation
    original_standard_deviation = np.std([p[1] for p in points])
    parallelized_standard_deviation = np.std(
        [p[1] for p in parallelized_points])

    # calculate the t statistic
    t_statistic, p_value = ttest_ind([point[1] for point in points], [
                                     point[1] for point in parallelized_points])

    # print out information
    print('\nEpsilon value = ' + str(epsilon))

    # print(classic_points)
    # print(parallelized_points)

    print('\nFile size of original line : ' +
          str(original_file_size) + ' KB')
    print('File size of simplified line : ' +
          str(parallel_file_size) + ' KB')

    print('Running time of parallelRDP : ' +
          str(parallelized_end_time - parallelized_start_time))

    print('\nNumber of points in original line : ' + str(len(points)))
    print('Number of points in parallel rdp line : ' +
          str(len(parallelized_points)))

    print('\nMean of original line : ' + str(original_mean))
    print('Mean of parallel rdp line : ' +
          str(parallelized_mean))

    print('\nStandard deviation of original line : ' +
          str(original_standard_deviation))
          
    print('Standard Deviation of parallel rdp line : ' +
          str(parallelized_standard_deviation))

    print('\nT statistic : ' +
          str(t_statistic))
    print('P value : ' +
          str(p_value))
    print('\n')

    # write comparison results
    tol = 0.05  # the tolerance value to compare how close to zero the t statistic

    if p_value >= tol:
        print('Result : There is no significant difference between the two lines\n')
    else:
        print(
            'Result : There is a significant difference between the two lines\n')