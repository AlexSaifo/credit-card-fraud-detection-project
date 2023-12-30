import numpy as np
def acc_conf_matrix(y_pred, out):
    out = np.array(out).flatten()
    res = out.copy()
    print (f"min = {res.min()}")
    print (f"max = {res.max()}")
    for index, value in enumerate(out):
        if value > 0:
            res[index] = 1
        else: res[index] = 0

    tp , tn, fp, fn=0,0,0,0
    for index in range(len(y_pred)):
        if(res[index] == y_pred[index] and res[index]==0):
            tn = tn + 1
        if(res[index] == y_pred[index] and res[index]==1):
            tp = tp + 1
        if(res[index] != y_pred[index] and res[index]==0):
            fn = fn + 1
        if(res[index] != y_pred[index] and res[index]==1):
            fp = fp + 1
    print (f"acc = {(tn+tp)/len(res)}")
    print (f" tn: {tn}   tp: {tp}   fn: {fn}   fp: {fp} ")
    