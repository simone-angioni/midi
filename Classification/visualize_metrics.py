import numpy as np

from functools import reduce

import pandas as pd

from matplotlib import pyplot as plt

from IPython.display import display

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


#def report_average(*args):
def report_average(reports):
    report_list = list()
    for report in reports:
        splited = [' '.join(x.split()) for x in report.split('\n\n')]
        header = [x for x in splited[0].split(' ')]
        data = np.array(splited[1].split(' ')).reshape(-1, len(header) + 1)
        data = np.delete(data, 0, 1).astype(float)
        avg_total = np.array([x for x in splited[2].split(' ')][3:]).astype(float).reshape(-1, len(header))
        df = pd.DataFrame(np.concatenate((data, avg_total)), columns=header)
        report_list.append(df)
    res = reduce(lambda x, y: x.add(y, fill_value=0), report_list) / len(report_list)
    return res.rename(index={res.index[-1]: 'avg / total'})


def display_report(report, string, n_target):

    display(pd.DataFrame(report))
    df = pd.DataFrame(report).T
    del df["support"]
    #Parametro iloc a 5 per il numero di classi di database nes mentre a 4 per db rock
    df.iloc[:n_target, :n_target].plot(kind='bar')
    #df.iloc[:4, :4].plot(kind='bar')
    plt.savefig(string + '_metrics_chart', bbox_inches="tight")
    #plt.show()
    plt.clf()
    plt.cla()
    plt.close()

    return
