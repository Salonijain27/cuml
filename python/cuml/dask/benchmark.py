from dask_cuda import LocalCUDACluster
from dask.distributed import Client, wait, futures_of, performance_report
import dask.array as da
import dask.dataframe as dd
from cuml.dask.ensemble import RandomForestRegressor
from cuml.dask.datasets.regression import make_regression
from cuml.dask.common.comms import CommsContext
from cuml.dask.common import to_dask_cudf
import cudf
import pandas as pd
import dask_cudf
from numba import cuda
import psutil

import numpy as np
import sys
from time import time, sleep
import warnings
import rmm
import cupy as cp
import dask
import numpy as np
from cuml.dask.common.dask_arr_utils import extract_arr_partitions
import os

base_n_points = 250_000_000
n_gb_data = np.asarray([2], dtype=int)
base_n_features = np.asarray([250], dtype=int)

def get_meta(df):
    ret = df.iloc[:0]
    return ret

# ideal_benchmark_f = open('/gpfs/fs1/dgala/b_outs/ideal_benchmark_f.csv', 'a')
def _combine_data(X, n_features, n_workers, n_partitions):
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("X in _combine  data : ", type(X))   #X_comb = np.concatenate((X[0],X[1]))
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(" SHAPE OF X before comb :", X)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    X = cp.asarray(X)
    #X_comb = cp.concatenate(X[0], X[1])
    print("X in _combine  data  AFTER CONVERSION : ", type(X))    #X_comb = np.concatenate((X[0],X[1]))
    print(" X_comb : ", X.shape)

    if n_features:
        print("convert datat to a cudf dataframe")
        X_cudf = cudf.DataFrame.from_gpu_matrix(X)
        print("convert datat to a DASK cudf dataframe")
        X_train_df = dask_cudf.from_cudf(X_cudf, npartitions=n_partitions)

    else:
        """
        X_np = cp.asnumpy(X_comb)
        X_cudf = cudf.Series(X_cudf)
        X_df = \
        dask_cudf.from_cudf(X_cudf, npartitions=n_partitions)
        """
    print(X_train_df.shape)
    print(X_train_df.strides)
    print(X_train_df.flags)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    return X
# ideal_benchmark_f = open('/gpfs/fs1/dgala/b_outs/ideal_benchmark_f.csv', 'a')

def _read_data(file_list, n_samples, n_features):
    print(file_list)
    if n_features:
        X = cp.zeros((n_samples * len(file_list), n_features), dtype='float32', order='F')
        for i in range(len(file_list)):
            X[i * n_samples: (i + 1) * n_samples, :] = cp.load(file_list[i])[:n_samples, :]
    else:
        X = cp.zeros((n_samples * len(file_list), ), dtype='float32', order='F')
        for i in range(len(file_list)):
            X[i * n_samples: (i + 1) * n_samples] = cp.load(file_list[i])[:n_samples]

    print(" SHAPE OF X BEFORE CUDF DATAFRAME : ", X.shape)
    print(" CONVERT CUPY ARRAY TO CUDF DATAFRAME ")
    if n_features:
        X_df = cudf.DataFrame.from_gpu_matrix(X)
        del X
    else:
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        print("Shape of X : ", X.shape[0])
        print("**********************************************")
        X_np = cp.asnumpy(X)
        del(X)
        print(" TYPE OF X AFTER CONVERSION TO NP : ", type(X_np))
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        X_pd = pd.Series(X_np)
        del(X_np)
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        X_df = cudf.Series(X_pd)
        del(X_pd)
        #X = cp.reshape(X, (X.shape[0],1), order='F')
        #print("Shape of X : ", X.shape)
        #X_df = cudf.DataFrame.from_gpu_matrix((X,X))
        #X_df = X_df.drop(0)
        #print(" HEAD OF X IN READ : ", X_df.head(5))
    print(" TYPE OF DATA AFTER CONVERSION : ", type(X_df))
    # X = cp.array(X, order='F')
    # del X
    return X_df
    
    #return X


def read_data(client, path, n_workers, workers, n_samples, n_features, n_gb, n_samples_per_gb, gb_partitions=None):
    total_file_list = os.listdir(path)
    total_file_list = [path + '/' + tfl for tfl in total_file_list]
    if gb_partitions:
        if len(gb_partitions) == n_workers - 1:
            file_list = total_file_list[:n_gb] if n_gb > n_workers else total_file_list[:n_workers]
            file_list = np.split(np.asarray(file_list), gb_partitions)
    elif n_gb:
        if n_gb < n_workers:
            file_list = total_file_list[:n_workers]
        elif n_gb % n_workers == 0:
            file_list = total_file_list[:n_gb]
        file_list = np.split(np.asarray(file_list), n_workers)
    else:
        file_list = total_file_list[:n_workers]
        file_list = np.split(np.asarray(file_list), n_workers)
    print(file_list)
    
    if n_gb < n_workers:
        n_samples_per_worker = int(n_samples / n_workers)
        X = [client.submit(_read_data, [file_list[i][0]], n_samples_per_worker, n_features, workers=[workers[i]]) for i in range(n_workers)]
    else:
        X = [client.submit(_read_data, file_list[i], n_samples_per_gb, n_features, workers=[workers[i]]) for i in range(n_workers)]

    wait([X])

    
    X = to_dask_cudf(X, client=client)
    print("############################################")
    print(" THE TDATAT TYPE OF x IS : ", type(X))
    print(" READ DATA IS NOW OVER : PHEW ")
    """
    if n_features:
        X = [da.from_delayed(dask.delayed(x), meta=cp.zeros(1, dtype=cp.float32),
            shape=(np.nan, n_features),
            dtype=cp.float32) for x in X]

    else:
        X = [da.from_delayed(dask.delayed(x), meta=cp.zeros(1, dtype=cp.float32),
            shape=(np.nan, ),
            dtype=cp.float32) for x in X] 

    #X = [x.to_dask_dataframe() for x in X]
    
    print(" FINAL TYPE OF X RETURNING FROM READ : ", type(X))
    X = da.concatenate(X, axis=0, allow_unknown_chunksizes=True)
    """
    #X = X.to_dask_dataframe()

    return X

def _read_data_test(file_list, n_samples, n_features):
    print(file_list)
    if n_features:
        X = cp.zeros((n_samples * len(file_list), n_features), dtype='float32', order='F')
        for i in range(len(file_list)):
            X[i * n_samples: (i + 1) * n_samples, :] = cp.load(file_list[i])[:n_samples, :]
    else:
        X = cp.zeros((n_samples * len(file_list), ), dtype='float32', order='F')
        for i in range(len(file_list)):
            X[i * n_samples: (i + 1) * n_samples] = cp.load(file_list[i])[:n_samples]

    print(X.shape)
    print(X.strides)
    print(X.flags)
    # X = cp.concatenate(X, axis=0)
    # X = cp.array(X, order='F')
    # del X
    return X


def read_data_test(client, path, n_workers, workers, n_samples, n_features, n_gb, n_samples_per_gb, gb_partitions=None):
    total_file_list = os.listdir(path)
    total_file_list = [path + '/' + tfl for tfl in total_file_list]
    print(" Number of features : ", n_features)
    if gb_partitions:
        if len(gb_partitions) == n_workers - 1:
            file_list = total_file_list[:n_gb] if n_gb > n_workers else total_file_list[:n_workers]
            file_list = np.split(np.asarray(file_list), gb_partitions)
    elif n_gb:
        if n_gb < n_workers:
            file_list = total_file_list[:n_workers]
        elif n_gb % n_workers == 0:
            file_list = total_file_list[:n_gb]
        file_list = np.split(np.asarray(file_list), n_workers)
    else:
        file_list = total_file_list[:n_workers]
        file_list = np.split(np.asarray(file_list), n_workers)
    print(file_list)
    
    if n_gb < n_workers:
        n_samples_per_worker = int(n_samples / n_workers)
        X = [client.submit(_read_data, [file_list[i][0]], n_samples_per_worker, n_features, workers=[workers[i]]) for i in range(n_workers)]
    else:
        X = [client.submit(_read_data, file_list[i], n_samples_per_gb, n_features, workers=[workers[i]]) for i in range(n_workers)]

    wait([X])

    if n_features:
        X = [da.from_delayed(dask.delayed(x), meta=cp.zeros(1, dtype=cp.float32),
            shape=(np.nan, n_features),
            dtype=cp.float32) for x in X]
    else:
        X = [da.from_delayed(dask.delayed(x), meta=cp.zeros(1, dtype=cp.float32),
            shape=(np.nan, ),
            dtype=cp.float32) for x in X] 
    X = da.concatenate(X, axis=0, allow_unknown_chunksizes=True)
    print(" shape of X : ", X.shape)

    return X


def _mse(ytest, yhat):
    if ytest.shape == yhat.shape:
        return (cp.mean((ytest - yhat) ** 2), ytest.shape[0])
    else:
        print("sorry")


def dask_mse(ytest, yhat, client, workers):
    print(" DASK MSE CALC FUNCTION ")
    print(" TYPE OF YTEST :  ", type(ytest))
    print(" TYPE OF Y HAT/ PREDS : ", type(yhat))
    ytest = ytest.to_dask_dataframe()
    yhat = yhat.to_dask_dataframe()
    ytest = ytest.to_dask_array()
    yhat = yhat.to_dask_array()
    ytest_parts = client.sync(extract_arr_partitions, ytest, client)
    yhat_parts = client.sync(extract_arr_partitions, yhat, client)
    #mse_parts = np.asarray([client.submit(_mse, ytest_parts[i][1], yhat_parts[i][1]).result() for i in range(len(ytest_parts))])
    #ytest_parts = client.sync(extract_arr_partitions, ytest, client)
    #print(" GOT THE Y_TEST PARTS ")
    #yhat_parts = client.sync(extract_arr_partitions, yhat, client)
    print(" GOT THE PREDS PARTS ")
    print(" CALLING THE INTERNAL _MSE FUNCTION ")
    mse_parts = np.asarray([client.submit(_mse, ytest_parts[i][1], yhat_parts[i][1]).result() for i in range(len(ytest_parts))])
    print(" CALC THE MSE OF EVERYTHING ")
    mse_parts[:, 0] = mse_parts[:, 0] * mse_parts[:, 1]
    return np.sum(mse_parts[:, 0]) / np.sum(mse_parts[:, 1])
    #mse_parts[:, 0] = mse_parts[:, 0] * mse_parts[:, 1]
    #return np.sum(mse_parts[:, 0]) / np.sum(mse_parts[:, 1])
    #return 0

def set_alloc():
    cp.cuda.set_allocator(rmm.rmm_cupy_allocator)

def run_ideal_benchmark(n_workers, X_filepath, y_filepath, n_gb, n_features,n_partitions, scheduler_file):

    # for n_gb_m in n_gb_data:
    #     for n_features in base_n_features:
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    if scheduler_file == 'None':
        cluster = LocalCUDACluster(n_workers=n_workers)
    fit_time = np.zeros(6)
    pred_time = np.zeros(6)
    mse = np.zeros(6)
    for i in range(6):
        try:
            n_points = int(base_n_points * n_gb)
            if scheduler_file != 'None':
                client = Client(scheduler_file=scheduler_file)
            else:
                client = Client(cluster)
            client.run(set_alloc)

            try:
                workers = list(client.has_what().keys())
                print(workers)
    
                n_samples = int(n_points / n_features)
                n_samples_per_gb = int(n_samples / n_gb)
                # X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_features / 10, n_parts=n_workers)

                # X = X.rechunk((n_samples / n_workers, n_features))
                # y = y.rechunk(n_samples / n_workers )

                X = read_data(client, X_filepath, n_workers, workers, n_samples, n_features, n_gb, n_samples_per_gb)
                #print(X.compute_chunk_sizes().chunks)
                wait(X)

                #print(" FINAL TYPE OF X RETURNED : ", type(X))
                y = read_data(client, y_filepath, n_workers, workers, n_samples, None, n_gb, n_samples_per_gb)
                #print(y.compute_chunk_sizes().chunks)
                #print(" FINAL TYPE OF y RETURNED : ", type(y))
                wait(y)

                rfr = RandomForestRegressor(max_depth=16)
                print(rfr)
                print(" ######################################################## ")
                print(" ######################################################## ")
                print(" ######################################################## ")
                free_mem = cuda.current_context().get_memory_info()[0]
                print(" FREE GPU MEMORY BEFORE FIT : ", free_mem)
                print(" CPU MEMORY BEFORE FIT : ", psutil.cpu_percent())
                print(" CPU MEMORY ALL INFO AS DICT : ", dict(psutil.virtual_memory()._asdict()))
                start_fit_time = time()
                print(" START FITTING THE  MODEL")
                rfr.fit(X, y)
                print(" FINISH FIT ")
                end_fit_time = time()
                print("nGPUS: ", n_workers, ", Shape: ", X.shape, ", Fit Time: ", end_fit_time - start_fit_time)
                fit_time[i] = end_fit_time - start_fit_time
                free_mem_2 = cuda.current_context().get_memory_info()[0]
                print(" FREE MEMORY AFTER FIT : ", free_mem_2)
                print(" CPU MEMORY AFTER FIT : ", psutil.cpu_percent())
                print(" CPU MEMORY ALL INFO AS DICT AFTER : ", dict(psutil.virtual_memory()._asdict()))

                start_pred_time = time()
                preds = rfr.predict(X, predict_model='GPU')
                wait(preds)
                end_pred_time = time()
                print("nGPUS: ", n_workers, ", Shape: ", X.shape, ", Predict Time: ", end_pred_time - start_pred_time)
                pred_time[i] = end_pred_time - start_pred_time
                print(" CHECK TH MSE VALUE OF PREDS RFR ")
                #print(cp.mean((ytest - yhat) ** 2), ytest.shape[0])
                mse[i] = dask_mse(y, preds, client, workers)
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(mse[i])

                del X, y, preds

            except Exception as e:
                print(e)
                continue

        finally:
            if 'X' in vars():
                del X
            if 'y' in vars():
                del y
            if 'preds' in vars():
                del preds

        client.close()

    
    if scheduler_file == 'None':
        cluster.close()
    print("starting write")
    fit_stats = [fit_time[0], np.mean(fit_time[1:]), np.min(fit_time[1:]), np.var(fit_time[1:])]
    pred_stats = [np.mean(pred_time[1:]), np.min(pred_time[1:]), np.var(pred_time[1:]), np.mean(mse[1:])]
    to_write = ','.join(map(str, [n_workers, n_samples, n_features] + fit_stats + pred_stats))
    print(to_write)
    with open('/gpfs/fs1/saljain/b_outs/benchmark_report.csv', 'a') as f:
        f.write(to_write)
        f.write('\n')
    print("ending write")
        #     break
        # break


if __name__ == '__main__':
    n_gpus = int(sys.argv[1])
    X_filepath = sys.argv[2]
    y_filepath = sys.argv[3]
    n_gb = int(sys.argv[4])
    n_features = int(sys.argv[5])
    scheduler_file = sys.argv[6]
    n_partitions = int(sys.argv[7])
    run_ideal_benchmark(n_gpus, X_filepath, y_filepath, n_gb, n_features, n_partitions, scheduler_file)