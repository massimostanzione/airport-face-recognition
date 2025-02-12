import math
import multiprocessing
import os

from matplotlib import pyplot as plt

from DES_Python.rvms import idfStudent
from globs import hours_to_secs
from inout import exists_in_json, append_to_json, load_from_json, extract_from_json, extract_key_values
from model.simulation import Simulation, ExecutionMode, split

arrivals_ctr = 0
arrivals_ctr_live = 0

lambdaa = 15 # richieste/s
expparam_base = 1 / lambdaa
expparam_fasciaoraria = -1

t = None

totresptimes = {}

from sortedcontainers import SortedList

eventlist = SortedList(key=lambda x: x.arrival_time)
nodes = []

event_balance_ctr = {"arrivals": 0, "completions": 0, "cloud_processings": 0}

NPROC = 2 # per il parallelismo


# welford
def estimate(valuesArray):
    if len(valuesArray) == 0:
        mean = 0.0
        stdev = 0.0
        w = 0.0
        return mean, stdev, w

    LOC = 0.95
    n = 0
    sum = 0.0
    mean = 0.0

    data = valuesArray[n]

    for i in range(1, len(valuesArray)):
        n += 1
        diff = float(data) - mean
        sum += diff * diff * (n - 1.0) / n
        mean += diff / n
        data = valuesArray[i]

    stdev = math.sqrt(sum / n)

    if n > 1:
        u = 1.0 - 0.5 * (1.0 - LOC)
        t = idfStudent(n - 1, u)
        w = t * stdev / math.sqrt(n - 1)

        return mean, stdev, w
    else:
        print("ERROR - insufficient data\n")


def run_comb_inf(terna):
    edge = terna[1]
    coord = terna[2]
    start = terna[0]
    # seeds = [123456789, 987654321, 246810121, 135792468]
    seeds = [123456789, 987654321, 246814421, 135792468]
    metrics = ["pop", "rho_edge", "rho_cloud", "rho_coord", "rt_prio", "rt_all"]
    avgs = []
    simname = f"INF-e{edge}-c{coord}-ore{start}"
    for seed in seeds:
        if not exists_in_json(edge, coord, seed, 0, simname + ".json"):
            sim = Simulation(simname, ExecutionMode.INFINITE_HORIZON, lambda_base, seed,
                             hours_to_secs(start), hours_to_secs(2), edge, coord, replicas_no=1)
            avgs.append(sim.run())

        else:
            print("skipped already run simulation")


def run_comb_fin(coppia):
    edge = coppia[0]
    coord = coppia[1]
    seeds = [123456789]  # , 987654321, 246814421, 135792468]
    metrics = ["pop", "rho_edge", "rho_cloud", "rho_coord", "rt_prio", "rt_all"]
    avgs = []
    simname = f"FIN-e{edge}-c{coord}-GENERATORE"
    for seed in seeds:
        if not exists_in_json(edge, coord, seed, 0, simname + ".json"):
            sim = Simulation(simname, ExecutionMode.FINITE_HORIZON, lambda_base, seed,
                             hours_to_secs(6), hours_to_secs(24), edge, coord, replicas_no=1)
            sim.run()
        else:
            print("skipped already run simulation")


if __name__ == '__main__':
    lambda_base = 15  # reqs/s

    # orizzonte infinito
    seeds = [123456789, 987654321, 246810121, 135792468]

    # ora, edge, cloud
    terne = [[18, 18, 5], [18, 23, 5], [12, 16, 4], [18, 16, 4], [18, 25, 4], [0, 25, 4], [6, 15, 2], [6, 20, 5],
             [6, 19, 4], [12, 19, 4], [12, 16, 3], [18, 16, 3], [18, 23, 3], [18, 23, 4], [0, 23, 4], [0, 5, 3],
             [0, 7, 2]]

    with multiprocessing.Pool(NPROC) as pool:
        pool.map(run_comb_inf, terne)

    # orizzonte finito
    # edge, cloud
    coppie = [[23, 5], [25, 7], [16, 4]]

    with multiprocessing.Pool(NPROC) as pool:
        pool.map(run_comb_fin, coppie)

    # grafici
    metrics = ["pop", "rho_edge", "rho_cloud", "rho_coord", "rt_prio", "rt_all"]
    for terna in terne:
        edge = terna[1]
        coord = terna[2]
        start = terna[0]
        # prendi il file
        simname = f"INF-e{edge}-c{coord}-ore{start}"

        avgs = []
        for metric in metrics:
            imgfilename = os.getcwd() + "/" + simname + "-" + metric + ".svg"
            if not os.path.exists(imgfilename):
                plt.figure()
                for seed in seeds:
                    item = extract_from_json(edge, coord, seed, 0, os.getcwd() + "/" + simname + ".json")

                    plt.plot(item[metric], ls='', marker='o', mfc='none', label=seed)
                    plt.xlabel("Batch index")
                if metric in ["rho_edge"]:
                    plt.axhline(.5, color='red', linestyle='dashed', linewidth=2, label='QoS')
                    plt.ylabel("Utilizzazione (Edge nodes)")
                if metric in ["rho_cloud"]:
                    plt.axhline(.5, color='red', linestyle='dashed', linewidth=2, label='QoS')
                    plt.ylabel("Utilizzazione (Cloud nodes)")
                if metric in ["rho_coord"]:
                    plt.axhline(.5, color='red', linestyle='dashed', linewidth=2, label='QoS')
                    plt.ylabel("Utilizzazione (Coordinator nodes)")
                if metric in ["rt_prio"]:
                    plt.axhline(1.5, color='red', linestyle='dashed', linewidth=2, label='QoS')
                    plt.ylabel("Tempo di risposta (richieste prioritarie)")
                if metric in ["rt_all"]:
                    plt.axhline(3, color='red', linestyle='dashed', linewidth=2, label='QoS')
                    plt.ylabel("Tempo di risposta (tutte le richieste)")
                plt.title(f"Configurazione {{{edge}, {coord}}}")
                plt.legend()
                print(os.getcwd())
                plt.savefig(imgfilename)
    seeds_coppie = [123456789]
    for coppia in coppie:
        print("1")
        edge = coppia[0]
        coord = coppia[1]
        simname = f"FIN-e{edge}-c{coord}"
        avgs = []
        for metric in metrics:
            imgfilename = os.getcwd() + "/" + simname + "-" + metric + ".svg"
            if not os.path.exists(imgfilename):
                plt.figure()
                for seed in seeds:
                    item = extract_from_json(edge, coord, seed, 0, os.getcwd() + "/" + simname + ".json")
                    plt.plot(item[metric], ls='', marker='o', mfc='none', label=seed)
                    plt.xlabel("Batch index")
                if metric in ["rho_edge"]:
                    plt.axhline(.5, color='red', linestyle='dashed', linewidth=2, label='QoS')
                    plt.ylabel("Utilizzazione (Edge nodes)")
                if metric in ["rho_cloud"]:
                    plt.axhline(.5, color='red', linestyle='dashed', linewidth=2, label='QoS')
                    plt.ylabel("Utilizzazione (Cloud nodes)")
                if metric in ["rho_coord"]:
                    plt.axhline(.5, color='red', linestyle='dashed', linewidth=2, label='QoS')
                    plt.ylabel("Utilizzazione (Coordinator nodes)")
                if metric in ["rt_prio"]:
                    plt.axhline(1.5, color='red', linestyle='dashed', linewidth=2, label='QoS')
                    plt.ylabel("Tempo di risposta (richieste prioritarie)")
                if metric in ["rt_all"]:
                    plt.axhline(3, color='red', linestyle='dashed', linewidth=2, label='QoS')
                    plt.ylabel("Tempo di risposta (tutte le richieste)")

                plt.title(f"Configurazione {{{edge}, {coord}}}")
                plt.legend()
                print(os.getcwd())
                plt.savefig(imgfilename)

    # intervalli di confidenza
    metrics = ["pop", "rho_edge", "rho_cloud", "rho_coord", "rt_prio", "rt_all"]
    filen = os.getcwd() + "/IC95.txt"
    if not os.path.exists(filen):
        os.mknod(filen)
    for terna in terne:
        edge = terna[1]
        coord = terna[2]
        start = terna[0]
        # prendi il file
        simname = f"INF-e{edge}-c{coord}-ore{start}"

        avgs = []
        for metric in metrics:
            for seed in seeds:
                item = extract_from_json(edge, coord, seed, 0, os.getcwd() + "/" + simname + ".json")
                sublist = []
                if type(item[metric][0]) is list:
                    for i in item[metric]:
                        sublist.append(i[0])
                else:
                    sublist = item[metric]
                media, stdd, w = estimate(sublist)
                stringa = simname + "\t" + metric + "\t" + str(media) + "\t" + str(stdd) + "\t" + str(w)
                with open(filen, "a") as f:
                    print(stringa, file=f)
import math
import multiprocessing
import os

from matplotlib import pyplot as plt

from DES_Python.rvms import idfStudent
from globs import hours_to_secs
from inout import exists_in_json, append_to_json, load_from_json, extract_from_json, extract_key_values
from model.simulation import Simulation, ExecutionMode, split

arrivals_ctr = 0
arrivals_ctr_live = 0

lambdaa = 15 # richieste/s
expparam_base = 1 / lambdaa
expparam_fasciaoraria = -1

t = None

totresptimes = {}

from sortedcontainers import SortedList

eventlist = SortedList(key=lambda x: x.arrival_time)
nodes = []

event_balance_ctr = {"arrivals": 0, "completions": 0, "cloud_processings": 0}

NPROC = 2 # per il parallelismo


# welford
def estimate(valuesArray):
    if len(valuesArray) == 0:
        mean = 0.0
        stdev = 0.0
        w = 0.0
        return mean, stdev, w

    LOC = 0.95
    n = 0
    sum = 0.0
    mean = 0.0

    data = valuesArray[n]

    for i in range(1, len(valuesArray)):
        n += 1
        diff = float(data) - mean
        sum += diff * diff * (n - 1.0) / n
        mean += diff / n
        data = valuesArray[i]

    stdev = math.sqrt(sum / n)

    if n > 1:
        u = 1.0 - 0.5 * (1.0 - LOC)
        t = idfStudent(n - 1, u)
        w = t * stdev / math.sqrt(n - 1)

        return mean, stdev, w
    else:
        print("ERROR - insufficient data\n")


def run_comb_inf(terna):
    edge = terna[1]
    coord = terna[2]
    start = terna[0]
    # seeds = [123456789, 987654321, 246810121, 135792468]
    seeds = [123456789, 987654321, 246814421, 135792468]
    metrics = ["pop", "rho_edge", "rho_cloud", "rho_coord", "rt_prio", "rt_all"]
    avgs = []
    simname = f"INF-e{edge}-c{coord}-ore{start}"
    for seed in seeds:
        if not exists_in_json(edge, coord, seed, 0, simname + ".json"):
            sim = Simulation(simname, ExecutionMode.INFINITE_HORIZON, lambda_base, seed,
                             hours_to_secs(start), hours_to_secs(2), edge, coord, replicas_no=1)
            avgs.append(sim.run())

        else:
            print("skipped already run simulation")


def run_comb_fin(coppia):
    edge = coppia[0]
    coord = coppia[1]
    seeds = [123456789]  # , 987654321, 246814421, 135792468]
    metrics = ["pop", "rho_edge", "rho_cloud", "rho_coord", "rt_prio", "rt_all"]
    avgs = []
    simname = f"FIN-e{edge}-c{coord}-GENERATORE"
    for seed in seeds:
        if not exists_in_json(edge, coord, seed, 0, simname + ".json"):
            sim = Simulation(simname, ExecutionMode.FINITE_HORIZON, lambda_base, seed,
                             hours_to_secs(6), hours_to_secs(24), edge, coord, replicas_no=1)
            sim.run()
        else:
            print("skipped already run simulation")


if __name__ == '__main__':
    lambda_base = 15  # reqs/s

    # orizzonte infinito
    seeds = [123456789, 987654321, 246810121, 135792468]

    # ora, edge, cloud
    terne = [[18, 18, 5], [18, 23, 5], [12, 16, 4], [18, 16, 4], [18, 25, 4], [0, 25, 4], [6, 15, 2], [6, 20, 5],
             [6, 19, 4], [12, 19, 4], [12, 16, 3], [18, 16, 3], [18, 23, 3], [18, 23, 4], [0, 23, 4], [0, 5, 3],
             [0, 7, 2]]

    with multiprocessing.Pool(NPROC) as pool:
        pool.map(run_comb_inf, terne)

    # orizzonte finito
    # edge, cloud
    coppie = [[23, 5], [25, 7], [16, 4]]

    with multiprocessing.Pool(NPROC) as pool:
        pool.map(run_comb_fin, coppie)

    # grafici
    metrics = ["pop", "rho_edge", "rho_cloud", "rho_coord", "rt_prio", "rt_all"]
    for terna in terne:
        edge = terna[1]
        coord = terna[2]
        start = terna[0]
        # prendi il file
        simname = f"INF-e{edge}-c{coord}-ore{start}"

        avgs = []
        for metric in metrics:
            imgfilename = os.getcwd() + "/" + simname + "-" + metric + ".svg"
            if not os.path.exists(imgfilename):
                plt.figure()
                for seed in seeds:
                    item = extract_from_json(edge, coord, seed, 0, os.getcwd() + "/" + simname + ".json")

                    plt.plot(item[metric], ls='', marker='o', mfc='none', label=seed)
                    plt.xlabel("Batch index")
                if metric in ["rho_edge"]:
                    plt.axhline(.5, color='red', linestyle='dashed', linewidth=2, label='QoS')
                    plt.ylabel("Utilizzazione (Edge nodes)")
                if metric in ["rho_cloud"]:
                    plt.axhline(.5, color='red', linestyle='dashed', linewidth=2, label='QoS')
                    plt.ylabel("Utilizzazione (Cloud nodes)")
                if metric in ["rho_coord"]:
                    plt.axhline(.5, color='red', linestyle='dashed', linewidth=2, label='QoS')
                    plt.ylabel("Utilizzazione (Coordinator nodes)")
                if metric in ["rt_prio"]:
                    plt.axhline(1.5, color='red', linestyle='dashed', linewidth=2, label='QoS')
                    plt.ylabel("Tempo di risposta (richieste prioritarie)")
                if metric in ["rt_all"]:
                    plt.axhline(3, color='red', linestyle='dashed', linewidth=2, label='QoS')
                    plt.ylabel("Tempo di risposta (tutte le richieste)")
                plt.title(f"Configurazione {{{edge}, {coord}}}")
                plt.legend()
                print(os.getcwd())
                plt.savefig(imgfilename)
    seeds_coppie = [123456789]
    for coppia in coppie:
        print("1")
        edge = coppia[0]
        coord = coppia[1]
        simname = f"FIN-e{edge}-c{coord}"
        avgs = []
        for metric in metrics:
            imgfilename = os.getcwd() + "/" + simname + "-" + metric + ".svg"
            if not os.path.exists(imgfilename):
                plt.figure()
                for seed in seeds:
                    item = extract_from_json(edge, coord, seed, 0, os.getcwd() + "/" + simname + ".json")
                    plt.plot(item[metric], ls='', marker='o', mfc='none', label=seed)
                    plt.xlabel("Batch index")
                if metric in ["rho_edge"]:
                    plt.axhline(.5, color='red', linestyle='dashed', linewidth=2, label='QoS')
                    plt.ylabel("Utilizzazione (Edge nodes)")
                if metric in ["rho_cloud"]:
                    plt.axhline(.5, color='red', linestyle='dashed', linewidth=2, label='QoS')
                    plt.ylabel("Utilizzazione (Cloud nodes)")
                if metric in ["rho_coord"]:
                    plt.axhline(.5, color='red', linestyle='dashed', linewidth=2, label='QoS')
                    plt.ylabel("Utilizzazione (Coordinator nodes)")
                if metric in ["rt_prio"]:
                    plt.axhline(1.5, color='red', linestyle='dashed', linewidth=2, label='QoS')
                    plt.ylabel("Tempo di risposta (richieste prioritarie)")
                if metric in ["rt_all"]:
                    plt.axhline(3, color='red', linestyle='dashed', linewidth=2, label='QoS')
                    plt.ylabel("Tempo di risposta (tutte le richieste)")

                plt.title(f"Configurazione {{{edge}, {coord}}}")
                plt.legend()
                print(os.getcwd())
                plt.savefig(imgfilename)

    # intervalli di confidenza
    metrics = ["pop", "rho_edge", "rho_cloud", "rho_coord", "rt_prio", "rt_all"]
    filen = os.getcwd() + "/IC95.txt"
    if not os.path.exists(filen):
        os.mknod(filen)
    for terna in terne:
        edge = terna[1]
        coord = terna[2]
        start = terna[0]
        # prendi il file
        simname = f"INF-e{edge}-c{coord}-ore{start}"

        avgs = []
        for metric in metrics:
            for seed in seeds:
                item = extract_from_json(edge, coord, seed, 0, os.getcwd() + "/" + simname + ".json")
                sublist = []
                if type(item[metric][0]) is list:
                    for i in item[metric]:
                        sublist.append(i[0])
                else:
                    sublist = item[metric]
                media, stdd, w = estimate(sublist)
                stringa = simname + "\t" + metric + "\t" + str(media) + "\t" + str(stdd) + "\t" + str(w)
                with open(filen, "a") as f:
                    print(stringa, file=f)

