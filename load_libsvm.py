import re

import numpy as np
import scipy.sparse as sp


# def load_libsvm_file(f_path):
#     X = defaultdict(list)
#     target = np.array([])
#     columns = 0
#     with open(f_path) as f:
#         for line in f:
#             columns += 1
#             data = line.split()
#             target = np.append(target, np.uint8(data[0]))
#             for (idx, value) in [item.split(':') for item in data[1:]]:
#                 X[idx].append(np.float16(value))
#
#     rows = len(X.keys())
#     shape = (columns, rows)
#     a = np.array([])
#     for key in X.keys():
#         a = np.append(a, X[key])
#
#     a = a.resize(shape)
#     return X, target

def load_libsvm_file(f_path):
    with open(f_path) as f:
        X_data = []
        X_row, X_col = [], []
        target = np.array([])
        for line in f:
            data = line.split()
            target = np.append(target, np.uint8(data[0]))
            for (idx, value) in [item.split(':') for item in data[1:]]:
                row = np.array([value])
                col_inds = np.nonzero(row)
                X_col.extend(col_inds)
                X_row.extend([idx] * len(col_inds))
                X_data.extend(row[col_inds])

        X = sp.coo_matrix((X_data, (X_row, X_col)), dtype=int)

    return X, target


test = 'Offer_523.libsvm'
test = int(re.search(r'\d+', test).group())
y = np.array([0, 1])
print(np.where(y == 1)[0])
print(test)
# clf = cb.CatBoost().load_model('data/Models/global_8_depth.cbm')
# clf = cb.to_regressor(clf)
# print(clf.tree_count_)
# clf.plot_tree(1).render('round-table', format='jpg')


# X, y = load_libsvm_file('data/data.libsvm')
# print(X.dtypes)
#
# print(X)
# d = pd.Series(X.columns).unique()
# print(len(d))
