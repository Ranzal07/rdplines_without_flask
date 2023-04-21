import csv
import os
import time
from rdp import rdp
import pandas as pd
import numpy as np
from typing import List
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import ttest_ind

executor = ThreadPoolExecutor(2)


def classic_rdp(points, eps):
    """
    Returns the classic rdp result
    """
    classic_rdp = rdp(points, epsilon=eps)
    return classic_rdp


def parallel_rdp(points, eps):
    """
    Returns the rdp result for every chunk
    """
    future = executor.submit(classic_rdp, points, eps)
    result = future.result()
    return result


def rdp_threadpool(points, epsilon, chunk_size, max_workers=4):
    """ RDP algorithm implementation using threadpool """
    chunks = [points[i:i+chunk_size] for i in range(0, len(points), chunk_size)]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(classic_rdp, chunks, [epsilon]*len(chunks)))
    return np.vstack(results)


def parallel_rdp_algorithm(data: List[List[float]], epsilon: float, chunk_size: int = None) -> List[List[float]]:
    """
    This is the function where the process of running all the chunks of the original line will happen in a parallel way through the use of multiprocessing's threadpoolexecutor
    """

    # Create a thread pool with two threads
    executor = ThreadPoolExecutor(2)

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
        return 1
    elif len_data <= 1000:
        return 4
    elif len_data <= 5000:
        return 8
    elif len_data <= 30000:
        return 16
    else:
        return 20


def calculate_epsilon(points):
    num_points = len(points)
    x1, y1 = points[0]  # get the first point of points
    x2, y2 = points[-1]  # get the last point of points

    # Calculating the perpendicular distance of each point to the line segment
    U = []    # Store all the perpendicular distance for each points
    for i in range(num_points):
        xi, yi = points[i]
        ui = abs((y2 - y1) * xi - (x2 - x1) * yi + x2 * y1 -
                 y2 * x1) / ((y2 - y1) * 2 + (x2 - x1) * 2) ** 0.5
        U.append(ui)

    total_ui = sum(U)   # Sums all the perpendicular distance for each points
    time_interval = 0.001    # get the time_interval of points
    # calculating the dynamic epsilon value
    epsilon = (total_ui * time_interval) / (x2 - x1)

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


filename = input("Enter CSV file path (e.g. originals/Alcohol_Sales.csv): ")

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
    eps_start_time = time.time()
    epsilon = calculate_epsilon(points)
    eps_end_time = time.time()

    # chunk size
    chunk = find_optimal_chunk_size(points)

    # get running time for classic rdp
    classic_start_time = time.time()
    classic_points = classic_rdp(points, epsilon)
    classic_end_time = time.time()

    # parallel results
    parallelized_start_time = time.time()
    parallelized_points = rdp_threadpool(points, epsilon, chunk)
    parallelized_end_time = time.time()

    # file sizes
    save_file = save_points_to_csv(
        points=parallelized_points, filename=filename[10:], columns=cols)
    directory = 'simplified/' + \
        filename[10:].split('.')[0] + '(simplified).csv'

    original_file_size = 0
    parallel_file_size = 0

    if (save_file):
        original_file_size = get_file_size(filename)
        parallel_file_size = get_file_size(directory)

    # calculate mean
    classic_mean = np.mean(np.mean(classic_points, axis=0)[1])
    parallelized_mean = np.mean(np.mean(parallelized_points, axis=0)[1])

    # calculate standard deviation
    classic_standard_deviation = np.std(classic_points, ddof=1)
    parallelized_standard_deviation = np.std(parallelized_points, ddof=1)

    # calculate the t statistic
    t_statistic, p_value = ttest_ind([point[0] for point in classic_points], [
                                     point[0] for point in parallelized_points])

    # print out information
    print('\nEpsilon value = ' + str(epsilon))

    print('\nNumber of points in original line : ' + str(len(points)))
    print('Number of points in simplified line : ' +
          str(len(parallelized_points)))

    print('\nFile size of original line : ' +
          str(original_file_size) + ' KB')
    print('File size of simplified line : ' +
          str(parallel_file_size) + ' KB')

    print('\nRunning time of classicRDP : ' +
          str(classic_end_time - classic_start_time))
    print('Running time of parallelRDP : ' +
          str(parallelized_end_time - parallelized_start_time))

    print('\nMean of original line : ' + str(classic_mean))
    print('Mean of simplified line : ' +
          str(parallelized_mean))

    print('\nStandard deviation of original line : ' +
          str(classic_standard_deviation))
    print('Standard Deviation of simplified line : ' +
          str(parallelized_standard_deviation))

    print('\nT statistic : ' +
          str(t_statistic))
    print('P value : ' +
          str(p_value))
    print('\n')

    # write comparison results
    tol = 0.01  # the tolerance value to compare how close to zero the t statistic

    classic_time = classic_end_time - classic_start_time
    parallel_time = parallelized_end_time - parallelized_start_time

    if parallel_time < classic_time:
        print("Parallel RDP is faster than Classic RDP!")
    else:
        print("Classic RDP is faster than Parallel RDP!")

    if abs(t_statistic) < tol and p_value > 0.05:
        print('Result : There is no significant difference between the two lines\n')
    else:
        print(
            'Result : There is a significant difference between the two lines\n')
