import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import t
from stats import estimate
from inout import extract_from_json, extract_key_values, print_ic95_fin, print_ic95_inf

metrics = ["pop", "rho_edge", "rho_cloud", "rho_coord", "rt_prio", "rt_all"]
def plot_with_confidence(metric,edge, coord, data, confidence=0.95):
    data_array = np.array(data)
    # calcolo delle statistiche
    mean_values = np.mean(data_array, axis=0)
    std_dev = np.std(data_array, axis=0, ddof=1)
    n_rep = data_array.shape[0]  # numero di repliche

    # calcolo del valore t (intv. di confidenza)
    t_value = t.ppf((1 + confidence) / 2, df=n_rep - 1)
    stderr = std_dev / np.sqrt(n_rep)
    margin_of_error = t_value * stderr

    # Intervallo di confidenza superiore e inferiore
    lower_bound = mean_values - margin_of_error
    upper_bound = mean_values + margin_of_error

    time_points = np.arange(data_array.shape[1])

    # Plot
    plt.figure()
    plt.plot(time_points, mean_values, color='blue')
    plt.fill_between(time_points, lower_bound, upper_bound, color='skyblue', alpha=0.4)
    plt.xlabel("Tempo")
    if metric in ["pop"]:
        plt.ylabel("Popolazione nel sistema")
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

    labels = ["6:00","12:00","18:00","24:00","6:00"]
    xticks = np.arange(0, 145, 36)
    plt.xticks(xticks, labels)

    for t_mark in range(0, len(time_points), 36):
        plt.axvline(x=t_mark, color='grey', linestyle='--', alpha=0.5)

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    return plt, mean_values


def plot_infinite(terna, seeds):
    # grafici
    metrics = ["pop", "rho_edge", "rho_cloud", "rho_coord", "rt_prio", "rt_all"]
    edge = terna[1]
    coord = terna[2]
    start = terna[0]
    # prendi il file
    simname = f"INF-e{edge}-c{coord}-ore{start}"
    for metric in metrics:
        imgfilename = os.getcwd() + "/" + simname + "-" + metric + ".svg"
        plt.figure()
        for seed in seeds:
            item = extract_from_json(edge, coord, seed, 0, os.getcwd() + "/" + simname + ".json")
            print_ic95_inf(terna,seed)
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
        print("saving graph for", simname,"...")
        plt.savefig(imgfilename)

def plot_finite(coppia):
    edge = coppia[0]
    coord = coppia[1]
    simname = f"FIN-e{edge}-c{coord}"
    for metric in metrics:
        imgfilename = os.getcwd() + "/" + simname + "-" + metric + ".svg"
        items=extract_key_values(simname+".json", metric)
        plt,series = plot_with_confidence(metric, edge, coord, items)
        print("saving graph for", simname,"...")
        plt.savefig(imgfilename)
        avg, stddev, w=estimate(series)
        print_ic95_fin(coppia, metric, avg, stddev, w)
