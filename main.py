import multiprocessing

from DES_Python.rngs import plantSeeds, selectStream, getSeed
from globs import hours_to_secs
from inout import exists_in_json, append_to_json, load_from_json, extract_from_json, extract_key_values, print_ic95_fin
from model.simulation import Simulation, ExecutionMode, split
from plots import plot_infinite, plot_finite
from stats import estimate

arrivals_ctr = 0
arrivals_ctr_live = 0

lambdaa = 15  # richieste/s
expparam_base = 1 / lambdaa
expparam_fasciaoraria = -1

t = None

totresptimes = {}

from sortedcontainers import SortedList

eventlist = SortedList(key=lambda x: x.arrival_time)
nodes = []

event_balance_ctr = {"arrivals": 0, "completions": 0, "cloud_processings": 0}

NPROC = 1  # per il parallelismo

metrics = ["pop", "rho_edge", "rho_cloud", "rho_coord", "rt_prio", "rt_all"]


def run_comb_inf(combination, seeds):
    start = combination[0]
    edge = combination[1]
    coord = combination[2]
    metrics = ["pop", "rho_edge", "rho_cloud", "rho_coord", "rt_prio", "rt_all"]
    simname = f"INF-e{edge}-c{coord}-ore{start}"
    for seed in seeds:
        if not exists_in_json(edge, coord, seed, 0, simname + ".json"):
            sim = Simulation(simname, ExecutionMode.INFINITE_HORIZON, lambda_base, seed,
                             hours_to_secs(start), hours_to_secs(2), edge, coord, replicas_no=1)
            sim.run()

        else:
            print("skipped already run simulation", simname)


def run_comb_fin(coppia):
    edge = coppia[0]
    coord = coppia[1]
    metrics = ["pop", "rho_edge", "rho_cloud", "rho_coord", "rt_prio", "rt_all"]
    simname = f"FIN-e{edge}-c{coord}"
    if not exists_in_json(edge, coord, 123456789, 0, simname + ".json"):
        sim = Simulation(simname, ExecutionMode.FINITE_HORIZON, lambda_base, 123456789,
                         hours_to_secs(6), hours_to_secs(24), edge, coord, replicas_no=64)
        sim.run()
    else:
        print("skipped already run simulation", simname)


def infinite_horizon_sim(comblist):
    print("Starting infinite horizon simulations...")
    seeds_inf = [123456789, 987654321, 246814421, 135792468]

    for terna in comblist:
        run_comb_inf(terna, seeds_inf)
        plot_infinite(terna, seeds_inf)

    print("... done.")


def finite_horizon_sim(comblist):
    print("Starting finite horizon simulations...")
    for coppia in comblist:
        run_comb_fin(coppia)
        plot_finite(coppia)
    print("... done")


if __name__ == '__main__':
    lambda_base = 15  # reqs/s

    # each sublist is [starttime, edge_no, cloud_no]
    infinite_horizon_comblist = [[6, 15, 2], [6, 19, 4], [6, 19, 8], [12, 16, 4], [12, 16, 7], [18, 19, 5], [18, 23, 9],
                                 [0, 5, 3], [0, 7, 3]]

    # each sublist is [edge_no, cloud_no]
    finite_horizon_comblist = [[23, 9]]

    infinite_horizon_sim(infinite_horizon_comblist)
    finite_horizon_sim(finite_horizon_comblist)
