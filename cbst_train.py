import asyncio
import os
import re
from collections import defaultdict
from datetime import datetime

import catboost as cb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight


async def load_libsvm_file(f_path):
    X = defaultdict(list)
    target = np.array([])
    with open(f_path) as f:
        for line in f:
            data = line.split()
            target = np.append(target, np.uint8(data[0]))
            for (idx, value) in [item.split(':') for item in data[1:]]:
                X[idx].append(np.float16(value))

    return X, target


async def fill_up_matrix(X, y):
    data = {}
    for key in X.keys():
        data[key] = len(X[key])
    data = {k: v for k, v in data.items() if v > len(data) / 10}
    newX = {}
    for key in data.keys():
        if key == '3':
            newX[key] = np.array(X[key], dtype=np.float16)
            continue
        newX[key] = np.array(np.hstack([X[key], np.zeros(y - len(X[key]))]), dtype=np.uint8)
    return newX


async def learn_catboost_model(X_train, X_test, y_train, y_test, params):
    class_weights = list(class_weight.compute_class_weight('balanced',
                                                           np.unique(y_train),
                                                           y_train))
    params['class_weights'] = class_weights
    model = cb.CatBoost(params=params)

    model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=(X_test, y_test), verbose=True, use_best_model=True)

    return model


async def learn_model_on_offer(dir_name, save_path, params):
    files = [x for x in os.listdir(dir_name) if '.libsvm' in x]
    for file in files:
        offer_num = int(re.search(r'\d+', file).group())
        print(f'Start to learn on {offer_num}')
        file_name = f"{dir_name}/{file}"
        X, y = await load_libsvm_file(file_name)
        clicks = [x for x in y if x == 1]
        if len(clicks) < 500:
            print(f'This offer has not enough clicks {len(clicks)}')
            continue
        X = pd.DataFrame(await fill_up_matrix(X, len(y)))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=44400)
        model = await learn_catboost_model(X_train, X_test, y_train, y_test, params)
        metric = params['eval_metric']
        score = model.best_score_['validation'][metric]
        if score < 0.6:
            print(f'{metric} metric isnt good enough')
            continue
        dir_path = f'{save_path}/{datetime.now().strftime("%Y-%m-%d %H:00:00")}'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f'{dir_path}. Created')
        model.save_model(f'{dir_path}/{offer_num}.cbm')


async def concatenate_all_files_to_one_set(root_dir):
    files = [x for x in os.listdir(root_dir) if '.libsvm' in x]
    X = defaultdict(list)
    y = []
    for file in files:
        data, target = await load_libsvm_file(f"{root_dir}/{file}")
        for key in data.keys():
            X[key].append(data[key])
        y = y + target
    for key in X.keys():
        if key == '3':
            X[key] = np.array([item for sublist in X[key] for item in sublist], dtype=np.float16)
            continue
        X[key] = np.array([item for sublist in X[key] for item in sublist], dtype=np.uint8)

    return X, np.array(y, dtype=np.uint8)


async def make_global_model(dir_path, save_path, params):
    X, y = await concatenate_all_files_to_one_set(dir_path)
    X = pd.DataFrame(await fill_up_matrix(X, len(y)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=44400)
    model = await learn_catboost_model(X_train, X_test, y_train, y_test, params)
    metric = params['eval_metric']
    score = model.best_score_['validation'][metric]
    if score < 0.6:
        print(f'{metric} metric isnt good enough')
        return
    dir_path = f'{save_path}/{datetime.now().strftime("%Y-%m-%d %H:00:00")}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f'{dir_path}. Created')
    model.save_model(f'{dir_path}/global.cbm')


params = {
    'eval_metric': 'F1',
    'iterations': 50,
    'logging_level': 'Silent',
    'loss_function': 'Logloss',
    'l2_leaf_reg': 3,
    'od_pval': 0.001,
    'random_strength': 1,
    'bagging_temperature': 2,
    'feature_border_type': 'MaxLogSum',
    'random_seed': 44000,
    'learning_rate': 0.3}


async def main():
    try:
        task2 = asyncio.create_task(make_global_model(f'data', 'data/Models', params))
        task1 = asyncio.create_task(learn_model_on_offer(f'data', 'data/Models', params))
    except Exception as e:
        print(e.__repr__())

    await task1, task2
    # await learn_model_on_offer(f'data', 'data/Models', params)


# await make_global_model(f'data', 'data/Models', params)
#  with PoolExecutor(max_workers=64) as executor:
#      f = executor.submit(learn_model_on_offer, f'data', 'data/Models', params)
#      f2 = executor.submit(make_global_model, f'data', 'data/Models', params)
#      print(f.result(), f2.result())


asyncio.run(main())
# offer_thread = threading.Thread(target=learn_model_on_offer,
#                                 args=(f'data', 'data/Models', params, learn_catboost_model),
#                                 name='Offer Learning Thread')
#
# offer_thread.start()
#
# global_thread = threading.Thread()
#
# print('Total number of threads', threading.activeCount())
#
# print('List of threads: ', threading.enumerate())
