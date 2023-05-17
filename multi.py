import mph
from scikitopt.sko.GA import GA
from scikitopt.sko.tools import set_run_mode
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import os
import sys

matplotlib.use('Agg')

os.makedirs('images', exist_ok=True)
os.makedirs('models', exist_ok=True)

# k_matcustom0 = Al
# k_matcustom1 = Grease

cpu_core = 52
pop_mult = 4
iter_num = 120


def save_history(y, x):
    content = np.append(y, x)
    content = content.reshape(1, -1)
    data = pd.DataFrame(content)
    data.to_csv('Xy_history.csv', mode='a', header=False)


def opt(model, matrix_mat):
    model = model.java

    model.component("comp1").geom("geom1").runPre("sel1")
    model.component("comp1").geom("geom1").feature("sel1").selection("selection").clear()

    model.component("comp1").geom("geom1").runPre("sel2")
    model.component("comp1").geom("geom1").feature("sel2").selection("selection").clear()

    for i in range(0, 40):
        for j in range(0, 40):
            if (matrix_mat[i, j].item() == 0):
                model.component("comp1").geom("geom1").feature("sel1").selection("selection").set(
                    "arr1(" + str(i + 1) + "," + str(j + 1) + ")", 1)
            else:
                model.component("comp1").geom("geom1").feature("sel2").selection("selection").set(
                    "arr1(" + str(i + 1) + "," + str(j + 1) + ")", 1)

    model.component("comp1").physics("ht2").feature("solid2").selection().named("geom1_sel1");

    model.sol("sol1").runAll();
    model.sol("sol2").runAll();

    probe_t = model.result().table("tbl1").getRealRow(0)[0] * 100

    return probe_t


def worker(matrix_mat):
    mph.option('session', 'stand-alone')
    client = mph.start(cores=1)
    client.caching(False)
    model = client.load('al-grease_outer_40.mph')
    temp = matrix_mat.reshape(40, 40)
    probe_t = opt(model, temp)
    print("probe_t:", probe_t)
    sys.stdout.flush()
    save_history(probe_t, matrix_mat)

    # to free up memory
    model.clear()
    model.reset()

    return probe_t


def run_ga(ga, iter_num):
    """Run the GA for iter_num times after initiating GA.

    :param ga: GA instance
    :param iter_num: number of iterations
    """
    start_time = time.time()
    for i in range(iter_num):
        temp_time = time.time()
        print("Iter {} begins...".format(i + 1))
        sys.stdout.flush()
        best_x, best_y = ga.run(1)
        print("best probe_t:", best_y)
        print("Iter {} end used {:2.2f} sec(s), total used: {:2.2f} sec(s) saving...".format(i + 1,
                                                                                             time.time() - temp_time,
                                                                                             time.time() - start_time))
        sys.stdout.flush()

        Y_history = pd.DataFrame(ga.all_history_Y)
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
        Y_history.min(axis=1).cummin().plot(kind='line')

        # save generation best
        np.savetxt('Y_history_gen.csv', np.array(ga.generation_best_Y), delimiter=',')
        np.savetxt('X_history_gen.csv', np.array(ga.generation_best_X), delimiter=',')
        np.savetxt('Y_history.csv', np.array(ga.all_history_Y), delimiter=',')

        # save chromosome and corresponding fitness (for resuming this computation in the future)
        df_Chrom = pd.DataFrame(data=ga.Chrom)
        df_FitV = pd.DataFrame(data=ga.FitV)
        df = pd.concat([df_Chrom, df_FitV], axis=1)
        df.to_csv('Chrom_fitV.csv', index=False, header=False)

        plt.savefig('./history.png', format='png')

    print('best_x:', best_x, '\n', 'best_y:', best_y)


def multi():
    # initiate GA
    start_time = time.time()
    print("Initiating...")
    set_run_mode(worker, 'multiprocessing')
    ga = GA(func=worker, n_dim=1600, size_pop=cpu_core * pop_mult, max_iter=1, prob_mut=0.01, lb=0, ub=1,
            precision=1)
    ga.record_mode = False
    print("Initiation completed used {:2.2f} sec(s)".format(time.time() - start_time))

    # run
    run_ga(ga, iter_num)


def remulti(previous_res_dir):
    """Remulti run the GA with previous result as initial point.

    :param previous_res_dir:
    :return:
    """
    # make sure the previous result exists
    if not os.path.exists(previous_res_dir):
        raise NotADirectoryError(previous_res_dir + "not found")
    if not os.path.exists(previous_res_dir + "/Y_history_gen.csv"):
        raise FileNotFoundError(previous_res_dir + "/Y_history_gen.csv not found")
    if not os.path.exists(previous_res_dir + "/X_history_gen.csv"):
        raise FileNotFoundError(previous_res_dir + "/X_history_gen.csv not found")
    if not os.path.exists(previous_res_dir + "/Y_history.csv"):
        raise FileNotFoundError(previous_res_dir + "/Y_history.csv not found")
    if not os.path.exists(previous_res_dir + "/Chrom_fitV.csv"):
        raise FileNotFoundError(previous_res_dir + "/Chrom_fitV.csv not found")

    # load previous result
    x_history_gen = np.loadtxt(previous_res_dir + "/X_history_gen.csv", delimiter=',')
    y_history_gen = np.loadtxt(previous_res_dir + "/Y_history_gen.csv", delimiter=',')
    y_history = np.loadtxt(previous_res_dir + "/Y_history.csv", delimiter=',')
    # Chrom and FitV is the key data to resume the previous computation
    df = pd.read_csv(previous_res_dir + "/Chrom_fitV.csv", header=None)
    Chrom = np.array(df.iloc[:, :-1])
    FitV = np.array(df.iloc[:, -1])

    # initialize the GA
    start_time = time.time()
    print("Remulti initiating...")
    set_run_mode(worker, 'multiprocessing')
    ga = GA(func=worker, n_dim=1600, size_pop=cpu_core * pop_mult, max_iter=1, prob_mut=0.01, lb=0, ub=1,
            precision=1)
    # load previous result into ga
    ga.generation_best_X = x_history_gen.tolist()
    ga.generation_best_Y = y_history_gen.tolist()
    ga.all_history_Y = y_history.tolist()
    ga.Chrom = Chrom
    ga.FitV = FitV
    print("Initiation completed used {:2.2f} sec(s)".format(time.time() - start_time))

    # run
    run_ga(ga, iter_num)


if __name__ == '__main__':
    multi()
