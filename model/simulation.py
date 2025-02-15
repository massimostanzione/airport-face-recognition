import math
import os
from copy import deepcopy
from enum import Enum

import numpy as np
from sortedcontainers import SortedList

from DES_Python.rngs import plantSeeds, selectStream, random, getSeed
from DES_Python.rvgs import Exponential, TruncatedNormal, Lognormal
from globs import hours_to_secs, GenerationMode, SAMPLING_TIME_FIN, seconds_to_hhmm, INFINITY, N, K, SAMPLING_TIME_INF
from inout import append_to_json, exists_in_json
from model.events import EventType, Event
from model.nodes import NodeMM1, NodePrio2, Node
from model.requests import categorize_request, RequestCategory, pty, Request
from model.time import Time, Track, lookup_timeslot, TimeSlot, lookup_timeslot_next


class NodeType(Enum):
    EDGE = 0,
    CLOUD = 1,
    COORDINATOR = 2


class ExecutionMode(Enum):
    COMPUTATIONAL_MODEL = 0,
    VERIFY = 1,
    FINITE_HORIZON = 2,
    INFINITE_HORIZON = 3


fasce_orarie = [
    TimeSlot("Notte (0-6)", hours_to_secs(0), 0.1),
    TimeSlot("Mattino (6-12)", hours_to_secs(6), 0.3),
    TimeSlot("Pomeriggio (12-18)", hours_to_secs(12), 0.25),
    TimeSlot("Sera (18-24)", hours_to_secs(18), 0.35),
]

NUM_CLOUD = 1

arrivalTemp = 0


class EdgeNode(NodeMM1):
    def __init__(self):
        super().__init__()
        self.type = NodeType.EDGE


class CoordinatorNode(NodePrio2):
    def __init__(self):
        super().__init__()
        self.type = NodeType.COORDINATOR


class NodeMultiServer(Node):
    def __init__(self, available_servers: int):
        super().__init__(available_servers)
        # i tempi di completamento
        self.subcompletions = [INFINITY] * available_servers
        self.reqeusts = []
        self.reqeusts_processing_now = []

    def get_available_server(self, time):
        sid = next(i for i in range(len(self.subcompletions)) if
                   self.subcompletions[i] <= time or self.subcompletions[i] == INFINITY)
        return sid if sid is not None else None


class CloudNode(NodeMultiServer):
    def __init__(self, available_servers: int):
        super().__init__(available_servers)
        self.type = NodeType.CLOUD


event_balance_ctr = {"arrivals": 0, "completions": 0, "cloud_processings": 0}


class Simulation:
    def __init__(self,
                 name: str,
                 exec_mode: ExecutionMode,
                 lambda_in: float,
                 seeds_no: int,
                 start: int,
                 duration: int,
                 edge_no: int,
                 coord_no: int,
                 replicas_no: int = -1,
                 k: int = -1,
                 b: int = -1):
        self.name = name
        self.exec_mode = exec_mode
        self.lambda_in = lambda_in
        self.seed = seeds_no
        self.start = start
        self.duration = duration
        self.edge_no = edge_no
        self.coord_no = coord_no
        self.replicas_no = replicas_no
        self.k = k
        self.b = b

        self.t = Time(self.edge_no + NUM_CLOUD + self.coord_no)
        self.event_list = SortedList(key=lambda x: x.arrival_time)
        self.areas = None
        self.nodes = []
        self.STOP = self.start + self.duration
        self.INFINITY = INFINITY
        self.generationmode = GenerationMode.GENERATION
        self.arrivals_ctr = 0

        self.current_timeslot = None
        self.current_lambda = -1

        # diagnostics
        self.totresptimes = {}
        self.cronometro = 0
        self.arrivicron = 0
        self.samplepoints = []

    def schedule_new_event(self, event_type: EventType, time: float = None, request: Request = None,
                           node_id: int = None,
                           fascia: TimeSlot = None, server_id=1):
        e = None

        if event_type == EventType.ARRIVAL_EDGE:
            e = self.get_system_arrival(time)
            self.arrivicron += 1
            if e is not None:
                event_balance_ctr["arrivals"] += 1

        elif event_type == EventType.COMPLETION_EDGE:
            e = Event(time, event_type, request, node_id)
            self.t.completion[node_id] = time
            event_balance_ctr["completions"] += 1

        elif event_type == EventType.ARRIVAL_CLOUD:
            e = Event(self.t.current, event_type, request)

        elif event_type == EventType.COMPLETION_CLOUD:
            e = Event(time, event_type, request, node_id)
            self.nodes[node_id].subcompletions[server_id] = time
            self.t.completion[node_id] = min(self.nodes[node_id].subcompletions)

        elif event_type == EventType.FEEDBACK_FROM_CLOUD:
            e = Event(self.t.current, event_type, request)

        elif event_type == EventType.ARRIVAL_COORDINATOR:
            e = Event(self.t.current, event_type, request)

        elif event_type == EventType.COMPLETION_COORDINATOR:
            e = Event(time, event_type, request, node_id)
            self.t.completion[node_id] = time

        elif event_type == EventType.CAMBIO_FASCIA_ORARIA:
            if fascia is None:
                print("error - time slot")
                exit(1)
            e = Event(time, event_type)

        else:
            print("error - unknown type")
            exit(1)

        if e is not None:
            self.event_list.add(e)
            pass
        return e

    def update_ts(self, time_slot: TimeSlot):
        avgperc = np.average([fasce_orarie[i].traffic_perc for i in range(len(fasce_orarie))])
        if avgperc != 0.25:
            print("error - avgperc consistency check")
        self.current_timeslot = time_slot
        self.current_lambda = self.lambda_in * time_slot.traffic_perc / avgperc

    def init_eventlist(self):
        self.event_list.clear()
        print("Initializing event list. This might take a while, please wait...")
        local_time = self.start
        fascia_successiva = None
        if self.exec_mode in [ExecutionMode.VERIFY, ExecutionMode.INFINITE_HORIZON]:
            timeslot = lookup_timeslot(self.start, fasce_orarie)
            self.update_ts(timeslot)
            self.schedule_new_event(EventType.CAMBIO_FASCIA_ORARIA, local_time, fascia=timeslot)
            fascia_successiva = TimeSlot("Infinity", self.INFINITY, -1)
        while (local_time < self.STOP):
            fascia_updated = lookup_timeslot(local_time, fasce_orarie)
            if self.exec_mode not in [ExecutionMode.VERIFY, ExecutionMode.INFINITE_HORIZON] and \
                    (self.current_timeslot is None or fascia_updated.name != self.current_timeslot.name):
                self.update_ts(fascia_updated)
                fascia_successiva = lookup_timeslot_next(local_time, fasce_orarie)
                self.schedule_new_event(EventType.CAMBIO_FASCIA_ORARIA, local_time, fascia=fascia_updated)
            e = self.schedule_new_event(EventType.ARRIVAL_EDGE, time=local_time)
            local_time = min(e.arrival_time if e is not None else self.INFINITY, fascia_successiva.start)

        print("... done!")

    def select_node_dest_type(self, event_type: EventType) -> NodeType:
        if event_type == EventType.ARRIVAL_EDGE:
            return NodeType.EDGE
        elif event_type == EventType.COMPLETION_EDGE:
            return NodeType.EDGE
        elif event_type == EventType.ARRIVAL_CLOUD:
            return NodeType.CLOUD
        elif event_type == EventType.COMPLETION_CLOUD:
            return NodeType.EDGE
        if event_type == EventType.FEEDBACK_FROM_CLOUD:
            return NodeType.EDGE
        elif event_type == EventType.ARRIVAL_COORDINATOR:
            return NodeType.COORDINATOR
        elif event_type == EventType.COMPLETION_COORDINATOR:
            return NodeType.COORDINATOR
        print("error - type unknown")
        exit(1)

    def init_trt(self):
        self.totresptimes = {}
        for k in RequestCategory:
            self.totresptimes[k] = []
            self.totresptimes["double_processing=False" + k.__str__()] = []
            self.totresptimes["double_processing=True" + k.__str__()] = []
        self.totresptimes["double_processing=False"] = []
        self.totresptimes["double_processing=True"] = []
        self.totresptimes["prio"] = {"value": 0, "index": 0}
        self.totresptimes["all"] = {"value": 0, "index": 0}

    def choose_minimum_load_node(self, node_type: NodeType, is_prio: bool = None):
        filtered = [n for n in self.nodes if n.type == node_type]

        if not filtered:
            raise ValueError(f"No nodes of type {node_type} found.")

        if node_type == NodeType.COORDINATOR:
            if is_prio:
                min_node = min(filtered, key=lambda n: (len(n.reqs_high), n.number))

            elif not is_prio:
                min_node = min(filtered, key=lambda n: n.number)
            else:
                print("error - minimum load node")
                exit(1)
        else:
            min_node = min(filtered, key=lambda n: n.number)

        return self.nodes.index(min_node)

    def is_node_idle(self, node_id):
        # True se:
        # - l'ultimo completamento è avvenuto prima di adesso (o al più adesso)
        #   < oppure >
        # - inizializzazione ad INFINITY
        return self.t.completion[node_id] <= self.t.current \
            or \
            self.t.completion[node_id] == self.INFINITY

    def esistono_richieste_nei_nodi(self) -> bool:
        return any(n.number > 0 for n in self.nodes)

    def get_system_arrival(self, arrivalTemp):
        self.arrivals_ctr += 1
        arrivalTemp += (
            Exponential(1 / self.current_lambda)) if self.generationmode == GenerationMode.GENERATION \
            else 1
        if arrivalTemp < self.STOP:
            e = Event(arrivalTemp, EventType.ARRIVAL_EDGE)
            e.request = Request(arrivalTemp)
            return e
        return None

    def GetService(self, node_type: NodeType, is_catC: bool = None):
        if node_type == NodeType.EDGE:
            selectStream(1)
            if is_catC is None:
                print("error - catC check")
                exit(1)
            if self.exec_mode != ExecutionMode.VERIFY:
                stddev = 0.025
                if not is_catC:
                    return TruncatedNormal(0.5, stddev, 0.48, 3 * stddev)
                else:
                    return TruncatedNormal(0.1, stddev, 0.08, 0.15)
            else:
                return Exponential(0.5)
        if node_type == NodeType.CLOUD:
            selectStream(2)
            if self.exec_mode != ExecutionMode.VERIFY:
                stddev = 0.1
                return TruncatedNormal(0.8, stddev, 0.75, 1.0)
            else:
                return Exponential(0.8)
        if node_type == NodeType.COORDINATOR:
            selectStream(3)
            if self.exec_mode != ExecutionMode.VERIFY:
                return TruncatedNormal(0.2, 0.05, 0.1, 0.5)
            else:
                return Exponential(0.2)

    def run(self):
        print(f"Starting simulation \"{self.name}\"")
        next_replica_seed = -1
        for replica in range(self.replicas_no):
            for seed_iter in range(1):
                if exists_in_json(self.edge_no, self.coord_no, self.seed, replica, self.name + ".json"):
                    print("skipped already run simulation")
                    continue
                self.t = Time(self.edge_no + NUM_CLOUD + self.coord_no)
                # scegli seed
                if replica > 0:
                    current_seed = next_replica_seed
                else:
                    current_seed = self.seed
                    plantSeeds(current_seed)
                selectStream(0)
                self.current_timeslot = None
                self.current_lambda = -1

                self.samplepoints.clear()

                self.init_trt()
                index = 0
                arrivals_ctr = 0
                arrivals_ctr_live = 0
                out = 0
                sampleno = 0

                self.areas = [Track() for _ in range(self.edge_no + NUM_CLOUD + self.coord_no)]
                self.nodes = [EdgeNode() for _ in range(self.edge_no)] + [CloudNode(self.edge_no) for _ in
                                                                          range(NUM_CLOUD)] + [
                                 CoordinatorNode() for _ in range(self.coord_no)]

                print("seed:", current_seed)

                self.t.current = self.start

                self.current_timeslot = None
                self.current_lambda = -1

                self.init_eventlist()
                print("scheduled n.", len(self.event_list), "events")

                self.current_timeslot = None
                self.current_lambda = -1

                event = self.event_list[0]

                print("Starting system simulation. It might take a while, please wait...")
                while (self.exec_mode in [ExecutionMode.VERIFY, ExecutionMode.INFINITE_HORIZON] and len(
                        self.samplepoints) < N) or \
                        (self.exec_mode not in [ExecutionMode.VERIFY, ExecutionMode.INFINITE_HORIZON] and len(
                            self.event_list) > 0 and self.t.current < self.STOP):  # or esistono_richieste_nei_nodi(nodes):

                    self.t.next = event.arrival_time

                    for i, node in enumerate(self.nodes):
                        if node.number > 0:
                            self.areas[i].node += (self.t.next - self.t.current) * node.number
                            self.areas[i].queue += (self.t.next - self.t.current) * (node.number - 1)

                            self.areas[i].service += self.t.next - self.t.current

                    self.t.current = self.t.next

                    if event.event_type in [EventType.ARRIVAL_EDGE, EventType.FEEDBACK_FROM_CLOUD]:
                        if not (
                                event.event_type == EventType.FEEDBACK_FROM_CLOUD and self.exec_mode == ExecutionMode.VERIFY):
                            if event.event_type == EventType.ARRIVAL_EDGE:
                                arrivals_ctr_live += 1
                            nodetype_filter = self.select_node_dest_type(event.event_type)
                            selected_node = self.choose_minimum_load_node(nodetype_filter)

                            # accoda il job
                            self.nodes[selected_node].number += 1
                            self.nodes[selected_node].reqeusts.append(event.request)

                            # se l'evento è oltre STOP, sia marcato come l'ultimo da eseguire (escluso)
                            if event.arrival_time > self.STOP:
                                self.t.last[selected_node] = self.t.current  # marcato come ultimo da eseguire
                                event.arrival_time = self.INFINITY  # Stop the event if it's beyond STOP

                            if self.nodes[selected_node].number == 1:
                                # schedula completamento (== esegui)

                                # categorizzazione
                                if event.request.category is None:
                                    event.request = categorize_request(event.request)
                                a = True
                                ts = 0 if a == False else \
                                    self.GetService(nodetype_filter, event.request.is_classC())
                                to_be_scheduled = self.schedule_new_event(EventType.COMPLETION_EDGE,
                                                                          self.t.current + ts, event.request,
                                                                          selected_node)

                    # processa un completamento ai nodi edge
                    elif event.event_type == EventType.COMPLETION_EDGE:
                        completed_node = event.server_id  # t.completion.index(t.current)

                        # aggiorna le statistiche per il nodo
                        self.nodes[completed_node].index += 1  # completamenti
                        self.nodes[completed_node].number -= 1  # popolazione

                        # categorizzazione
                        if event.request.category is None:
                            event.request = categorize_request(event.request)

                        # invio a cloud se unknown
                        if event.request.category is None:
                            print("Errore!!!")
                            exit(1)
                        elif event.request.category == RequestCategory.UNKNOWN:
                            to_be_scheduled = self.schedule_new_event(EventType.ARRIVAL_CLOUD, request=event.request)
                        else:
                            to_be_scheduled = self.schedule_new_event(EventType.ARRIVAL_COORDINATOR,
                                                                      request=event.request)
                            pass
                        self.nodes[completed_node].reqeusts.remove(event.request)

                        if self.nodes[completed_node].number > 0:
                            # schedula completamento
                            richiesta_successiva_in_coda = self.nodes[completed_node].reqeusts[0]
                            ts = 0 if event.request.category == RequestCategory.UNKNOWN else self.GetService(
                                NodeType.EDGE,
                                event.request.is_classC())
                            to_be_scheduled = self.schedule_new_event(EventType.COMPLETION_EDGE, self.t.current + ts,
                                                                      richiesta_successiva_in_coda,
                                                                      completed_node)

                    elif event.event_type == EventType.ARRIVAL_CLOUD:
                        event.request.double_processing = True
                        if event.request.category is not RequestCategory.UNKNOWN:
                            print(f"arrivata a cloud una richiesta non-unknown! ({event.request.category})")
                            exit(1)

                        nodetype_filter = self.select_node_dest_type(event.event_type)
                        selected_node = self.choose_minimum_load_node(nodetype_filter)
                        # accoda il job
                        self.nodes[selected_node].number += 1
                        self.nodes[selected_node].reqeusts.append(event.request)

                        # come per altrove, se .number è 1 (es. MM1), vuol dire che c'è solo lui nel nodo => lo processo subito
                        # l'equivalente multiserver è se .number<numero_servers
                        if self.nodes[selected_node].number <= self.nodes[selected_node].available_servers:
                            # schedula completamento (== esegui)
                            ts = self.GetService(nodetype_filter)
                            sid = self.nodes[selected_node].get_available_server(self.t.current)
                            if sid is None:
                                print("errore sid")
                                exit(1)
                            to_be_scheduled = self.schedule_new_event(EventType.COMPLETION_CLOUD, self.t.current + ts,
                                                                      event.request,
                                                                      selected_node,
                                                                      server_id=sid)
                            self.nodes[selected_node].reqeusts.remove(event.request)
                            self.nodes[selected_node].reqeusts_processing_now.append(event.request)
                            # l'evento di arrivo è eliminato di seguito, dopo gli if/elif

                    # processa un completamento ai nodi cloud
                    elif event.event_type == EventType.COMPLETION_CLOUD:
                        completed_node = event.server_id
                        if completed_node != self.edge_no and NUM_CLOUD == 1:  # perché il nodo cloud è, nell'ordine, appena dopo (self.edge_no+1)-1
                            print("errore in verifica cloud", completed_node)
                            exit(1)
                        # aggiorna le statistiche per il nodo
                        self.nodes[completed_node].index += 1  # completamenti
                        self.nodes[completed_node].number -= 1  # popolazione

                        # give the request a definitive category
                        while event.request.category == RequestCategory.UNKNOWN:
                            event.request = categorize_request(event.request)

                        self.nodes[completed_node].reqeusts_processing_now.remove(event.request)
                        to_be_scheduled = self.schedule_new_event(EventType.FEEDBACK_FROM_CLOUD, request=event.request)
                        event_balance_ctr["cloud_processings"] += 1
                        if len(self.nodes[completed_node].reqeusts) > 0:
                            # schedula completamento
                            richiesta_successiva_in_coda = self.nodes[completed_node].reqeusts[0]
                            ts = self.GetService(NodeType.CLOUD)
                            sid = self.nodes[completed_node].get_available_server(self.t.current)
                            if sid is None:
                                print("errore sid/2")
                                exit(1)

                            to_be_scheduled = self.schedule_new_event(EventType.COMPLETION_CLOUD, self.t.current + ts,
                                                                      richiesta_successiva_in_coda,
                                                                      completed_node, server_id=sid)

                            self.nodes[completed_node].reqeusts.remove(richiesta_successiva_in_coda)
                            self.nodes[completed_node].reqeusts_processing_now.append(richiesta_successiva_in_coda)

                    elif event.event_type == EventType.ARRIVAL_COORDINATOR:
                        if event.request.category is RequestCategory.UNKNOWN:
                            print(f"arrivata a coord una richiesta unknown! ({event.request.category})")
                            exit(1)

                        # scegli il nodo meno carico (via attributo number, i.e. popolazione in coda)
                        nodetype_filter = self.select_node_dest_type(event.event_type)
                        selected_node = self.choose_minimum_load_node(nodetype_filter,
                                                                      event.request.category.is_priority())
                        # accoda il job
                        self.nodes[selected_node].number += 1
                        if event.request.category.is_priority():
                            self.nodes[selected_node].reqs_high.append(event.request)
                        else:
                            self.nodes[selected_node].reqs_low.append(event.request)

                        if self.is_node_idle(selected_node):
                            if len(self.nodes[selected_node].reqs_high) == 1:
                                # schedula completamento (== esegui)
                                ts = self.GetService(nodetype_filter)
                                to_be_scheduled = self.schedule_new_event(EventType.COMPLETION_COORDINATOR,
                                                                          self.t.current + ts,
                                                                          event.request,
                                                                          selected_node)
                            elif len(self.nodes[selected_node].reqs_low) == 1:
                                if len(self.nodes[selected_node].reqs_high) > 0:
                                    print("inversione di priorità!")
                                    exit(1)
                                ts = self.GetService(nodetype_filter)
                                to_be_scheduled = self.schedule_new_event(EventType.COMPLETION_COORDINATOR,
                                                                          self.t.current + ts,
                                                                          event.request,
                                                                          selected_node)
                        else:
                            pass
                    # processa un completamento ai nodi cloud
                    elif event.event_type == EventType.COMPLETION_COORDINATOR:
                        completed_node = event.server_id  # t.completion.index(t.current)
                        if self.nodes[completed_node].type is not NodeType.COORDINATOR:
                            print("non coord!")
                            exit(1)
                        # aggiorna le statistiche per il nodo
                        self.nodes[completed_node].index += 1  # completamenti
                        self.nodes[completed_node].number -= 1  # popolazione

                        if event.request.category.is_priority():
                            self.nodes[completed_node].reqs_high.popleft()
                            tot_resp_time = self.t.current - event.request.generation_time
                            self.totresptimes["prio"]["value"] = \
                                (self.totresptimes["prio"]["value"] * self.totresptimes["prio"][
                                    "index"] + tot_resp_time) / (self.totresptimes["prio"]["index"] + 1)
                            self.totresptimes["prio"]["index"] += 1
                        else:
                            self.nodes[completed_node].reqs_low.popleft()
                        tot_resp_time = self.t.current - event.request.generation_time
                        self.totresptimes["all"]["value"] = \
                            (self.totresptimes["all"]["value"] * self.totresptimes["all"]["index"] + tot_resp_time) / (
                                    self.totresptimes["all"]["index"] + 1)
                        self.totresptimes["all"]["index"] += 1
                        self.totresptimes[event.request.category].append(tot_resp_time)

                        self.totresptimes["double_processing=" + str(event.request.double_processing)].append(
                            tot_resp_time)
                        self.totresptimes[
                            "double_processing=" + str(
                                event.request.double_processing) + event.request.category.__str__()].append(
                            tot_resp_time)
                        out += tot_resp_time
                        sampleno += 1

                        if not self.is_node_idle(completed_node):
                            print("error: node should be idle (before re-assigning it), a completion just occurred!")
                            exit(1)
                        if len(self.nodes[completed_node].reqs_high) > 0:
                            # schedula completamento
                            richiesta_successiva_in_coda = self.nodes[completed_node].reqs_high[0]
                            ts = self.GetService(NodeType.COORDINATOR)
                            to_be_scheduled = self.schedule_new_event(EventType.COMPLETION_COORDINATOR,
                                                                      self.t.current + ts,
                                                                      richiesta_successiva_in_coda, completed_node)

                        elif len(self.nodes[completed_node].reqs_low) > 0:
                            if len(self.nodes[completed_node].reqs_high) > 0:
                                print("inversione di priorità!")
                                exit(1)
                            # schedula completamento
                            richiesta_successiva_in_coda = self.nodes[completed_node].reqs_low[0]
                            ts = self.GetService(NodeType.COORDINATOR)
                            to_be_scheduled = self.schedule_new_event(EventType.COMPLETION_COORDINATOR,
                                                                      self.t.current + ts,
                                                                      richiesta_successiva_in_coda, completed_node)

                    elif event.event_type == EventType.CAMBIO_FASCIA_ORARIA:
                        fascia = lookup_timeslot(self.t.current, fasce_orarie)

                        avgperc = np.average([fasce_orarie[i].traffic_perc for i in range(len(fasce_orarie))])
                        lambda_fascia = self.lambda_in * fascia.traffic_perc / avgperc
                        global expparam_fasciaoraria
                        expparam_fasciaoraria = 1 / lambda_fascia

                    else:
                        print("EVENTO NON RICONOSCIUTO:", event.event_type)
                        exit(1)

                    to_be_deleted = next(e for e in self.event_list if e.id == event.id)
                    self.event_list.remove(to_be_deleted)
                    if (len(self.event_list) > 0):
                        event = self.event_list[0]

                    # campionamento
                    stime = SAMPLING_TIME_INF if self.exec_mode in [ExecutionMode.VERIFY,
                                                                    ExecutionMode.INFINITE_HORIZON] else SAMPLING_TIME_FIN
                    sample_slot = math.floor((self.t.current - self.start) / stime)

                    if len(self.samplepoints) <= sample_slot:
                        elapsed = self.t.current - self.start if self.t.current != self.start else 1

                        dividendo = {NodeType.EDGE: 1, NodeType.CLOUD: self.edge_no, NodeType.COORDINATOR: 1}
                        if self.exec_mode == ExecutionMode.VERIFY:
                            point = StatsPointVer(
                                {nodetype: np.average([(self.areas[i].node / (
                                        (self.nodes[i].index if self.nodes[i].index != 0 else 1) * dividendo[
                                    nodetype]))
                                                       for i, node in enumerate(self.nodes) if node.type == nodetype])
                                 for nodetype in NodeType},
                                {nodetype: np.average([(self.areas[i].queue / (
                                        (self.nodes[i].index if self.nodes[i].index != 0 else 1) * dividendo[
                                    nodetype]))
                                                       for i, node in enumerate(self.nodes) if node.type == nodetype])
                                 for nodetype in NodeType},
                                {nodetype: np.average([(self.areas[i].service / (
                                        (self.nodes[i].index if self.nodes[i].index != 0 else 1) * dividendo[
                                    nodetype]))
                                                       for i, node in enumerate(self.nodes) if node.type == nodetype])
                                 for nodetype in NodeType},
                                {nodetype: np.average([(self.areas[i].node / (elapsed * dividendo[nodetype]))
                                                       for i, node in enumerate(self.nodes) if node.type == nodetype])
                                 for nodetype in NodeType},
                                {nodetype: np.average([(self.areas[i].queue / (elapsed * dividendo[nodetype]))
                                                       for i, node in enumerate(self.nodes) if node.type == nodetype])
                                 for nodetype in NodeType},
                                {nodetype: np.average([(self.areas[i].service / (elapsed * dividendo[nodetype]))
                                                       for i, node in enumerate(self.nodes) if node.type == nodetype])
                                 for nodetype in NodeType},

                            )
                        else:
                            point = StatsPoint(
                                sum((self.areas[i].node / (elapsed)) for i in range(len(self.nodes))),
                                {"prio": self.totresptimes["prio"]["value"], "all": self.totresptimes["all"]["value"]},
                                {nodetype: np.average([(self.areas[i].service / (elapsed * dividendo[nodetype]))
                                                       for i, node in enumerate(self.nodes) if node.type == nodetype])
                                 for nodetype in NodeType}
                            )

                        self.samplepoints.append(point)
                        out = arrivals_ctr_live / elapsed
                        if sample_slot % 1000 == 0:
                            print("(still proceesing, please wait...)")
                        out = 0
                        arrivals_ctr_live = 0
                        sampleno = 0

                print("... done!")
                trascorso = self.t.current - self.start

                if self.exec_mode == ExecutionMode.VERIFY:
                    for i in range(self.edge_no + NUM_CLOUD + self.coord_no):
                        index = self.nodes[i].index
                        print(f"\nFor {self.nodes[i].type} node {i + 1}:")
                        print(f"   Jobs completed ........... = {index}")
                        if index > 0:
                            print(f"   Average interarrival time = {self.t.last[i] / index:.2f}")
                            print(f"   Average wait ............ = {self.areas[i].node / index:.2f}")
                            print(f"   Average delay ........... = {self.areas[i].queue / index:.2f}")
                            print(f"   Average service time .... = {self.areas[i].service / index:.2f}")
                            print(f"   Average # in the node ... = {self.areas[i].node / trascorso:.2f}")
                            print(f"   Average # in the queue .. = {self.areas[i].queue / trascorso:.2f}")
                            print(f"   Utilization ............. = {self.areas[i].service / trascorso:.2f}")
                        else:
                            print(f"   Average interarrival time = 0.00")
                            print(f"   Average wait ............ = 0.00")
                            print(f"   Average delay ........... = 0.00")
                            print(f"   Average service time .... = 0.00")
                            print(f"   Average # in the node ... = 0.00")
                            print(f"   Average # in the queue .. = 0.00")
                            print(f"   Utilization ............. = 0.00")

                    # per CENTRO

                    a = {}
                    a["intertime"] = 0  # t.last[i] / index
                    a["wait"] = 0  # areas[i].node / index
                    a["delay"] = 0  # areas[i].queue / index
                    a["servicetime"] = 0  # areas[i].service / index
                    a["popN"] = 0  # areas[i].node / t.current
                    a["popQ"] = 0  # areas[i].queue / t.current
                    a["rho"] = 0  # areas[i].service / t.current

                    d = {NodeType.EDGE: deepcopy(a), NodeType.CLOUD: deepcopy(a), NodeType.COORDINATOR: deepcopy(a)}
                    for i in range(self.edge_no + NUM_CLOUD + self.coord_no):
                        index = self.nodes[i].index

                        if index > 0:
                            d[self.nodes[i].type]["intertime"] += self.t.last[i] / index
                            d[self.nodes[i].type]["wait"] += self.areas[i].node / index
                            d[self.nodes[i].type]["delay"] += self.areas[i].queue / index
                            d[self.nodes[i].type]["servicetime"] += self.areas[i].service / index
                            d[self.nodes[i].type]["popN"] += self.areas[i].node / trascorso
                            d[self.nodes[i].type]["popQ"] += self.areas[i].queue / trascorso
                            d[self.nodes[i].type]["rho"] += self.areas[i].service / trascorso

                    for tt in NodeType:
                        if tt is NodeType.EDGE: quanti = self.edge_no
                        if tt is NodeType.CLOUD: quanti = NUM_CLOUD
                        if tt is NodeType.COORDINATOR: quanti = self.coord_no
                        print(f"\nFor type {tt}:")
                        if index > 0:
                            print(f"   Average interarrival time = ", d[tt]["intertime"] / quanti)
                            print(f"   Average wait ............ = ", d[tt]["wait"] / quanti)
                            print(f"   Average delay ........... = ", d[tt]["delay"] / quanti)
                            print(f"   Average service time .... = ", d[tt]["servicetime"] / quanti)
                            print(f"   Average # in the node ... = ", d[tt]["popN"] / quanti)
                            print(f"   Average # in the queue .. = ", d[tt]["popQ"] / quanti)
                            print(f"   Utilization ............. = ", d[tt]["rho"] / quanti)
                        else:
                            print(f"   Average interarrival time = 0.00")
                            print(f"   Average wait ............ = 0.00")
                            print(f"   Average delay ........... = 0.00")
                            print(f"   Average service time .... = 0.00")
                            print(f"   Average # in the node ... = 0.00")
                            print(f"   Average # in the queue .. = 0.00")
                            print(f"   Utilization ............. = 0.00")
                    print("CATEGORIZZAZIONI TOTALI:", sum(pty), "ARRIVI TOTALI:",
                          arrivals_ctr)  # , "EVENTCTR:", event_ctr)
                    print(self.arrivicron / self.t.current)
                else:
                    print("----------------------------------------")
                    print("VERIFICA degli OBIETTIVI")
                    print("")
                    print("ob. 1: numero di EN t.c. rho (EN o globale?) sia < 0.5")
                    sum_rho_edge = np.average([point.rhos[0][NodeType.EDGE] for point in self.samplepoints])
                    rho_cloud = np.average([point.rhos[0][NodeType.CLOUD] for point in self.samplepoints])
                    rho_coord = np.average([point.rhos[0][NodeType.COORDINATOR] for point in self.samplepoints])
                    print("\tnumero di EN:", self.edge_no, "numero di COORD:", self.coord_no)
                    print("\trho_edge:\t", sum_rho_edge, "\t", sum_rho_edge <= 0.5)
                    print("\trho_cloud:\t", rho_cloud, "\t", rho_cloud <= 0.5)
                    print("\trho_coord:\t", rho_coord, "\t", rho_coord <= 0.5)

                    print("")
                    print(
                        "ob. 2: TdR per richieste dangerous/suspect sia < 1.5 s, indipendentemente da ev. doppio passaggio")

                    tdr_prio = np.average([point.resptime[0]["prio"] for point in self.samplepoints])
                    tdr_all = np.average([point.resptime[0]["all"] for point in self.samplepoints])
                    print("\tTdR prio:\t", tdr_prio, tdr_prio <= 1.5)

                    print("")
                    print("ob. 3: TdR per TUTTE le richieste sia < 3 s")
                    tutte = []
                    for k in RequestCategory:
                        tutte += self.totresptimes[k]
                    print("\tTdR tutte:\t", tdr_all, tdr_all <= 3)

                if self.exec_mode in [ExecutionMode.VERIFY, ExecutionMode.INFINITE_HORIZON]:
                    splitted = split(self.samplepoints, K)[0]
                else:
                    splitted = self.samplepoints
                if self.exec_mode == ExecutionMode.VERIFY:
                    avgs = {"edge_no": -1, "coord_no": -1, "seed": -1, "replica_no": -1,
                            "wait_edge": [], "wait_cloud": [], "wait_coord": [],
                            "delay_edge": [], "delay_cloud": [], "delay_coord": [],
                            "service_edge": [], "service_cloud": [], "service_coord": [],
                            "nnode_edge": [], "nnode_cloud": [], "nnode_coord": [],
                            "nqueue_edge": [], "nqueue_cloud": [], "nqueue_coord": [],
                            "rho_edge": [], "rho_cloud": [], "rho_coord": [],
                            }
                else:
                    avgs = {"edge_no": -1, "coord_no": -1, "seed": -1, "replica_no": -1,
                            "pop": [], "rho_edge": [], "rho_cloud": [], "rho_coord": [], "rt_prio": [], "rt_all": []}
                cum = 0
                i = 0
                for point in splitted:
                    if self.exec_mode == ExecutionMode.VERIFY:
                        avgs["wait_edge"].append(point.wait[NodeType.EDGE])
                        avgs["wait_cloud"].append(point.wait[NodeType.CLOUD])
                        avgs["wait_coord"].append(point.wait[NodeType.COORDINATOR])
                        avgs["delay_edge"].append(point.delay[NodeType.EDGE])
                        avgs["delay_cloud"].append(point.delay[NodeType.CLOUD])
                        avgs["delay_coord"].append(point.delay[NodeType.COORDINATOR])
                        avgs["service_edge"].append(point.service[NodeType.EDGE])
                        avgs["service_cloud"].append(point.service[NodeType.CLOUD])
                        avgs["service_coord"].append(point.service[NodeType.COORDINATOR])
                        avgs["nnode_edge"].append(point.nnode[NodeType.EDGE])
                        avgs["nnode_cloud"].append(point.nnode[NodeType.CLOUD])
                        avgs["nnode_coord"].append(point.nnode[NodeType.COORDINATOR])
                        avgs["nqueue_edge"].append(point.nqueue[NodeType.EDGE])
                        avgs["nqueue_cloud"].append(point.nqueue[NodeType.CLOUD])
                        avgs["nqueue_coord"].append(point.nqueue[NodeType.COORDINATOR])
                        avgs["rho_edge"].append(point.rho[NodeType.EDGE])
                        avgs["rho_cloud"].append(point.rho[NodeType.CLOUD])
                        avgs["rho_coord"].append(point.rho[NodeType.COORDINATOR])

                    else:
                        avgs["pop"].append(point.population)
                        avgs["rho_edge"].append(point.rhos[0][NodeType.EDGE])
                        avgs["rho_cloud"].append(point.rhos[0][NodeType.CLOUD])
                        avgs["rho_coord"].append(point.rhos[0][NodeType.COORDINATOR])
                        avgs["rt_prio"].append(point.resptime[0]["prio"])
                        avgs["rt_all"].append(point.resptime[0]["all"])
                avgs["edge_no"] = self.edge_no
                avgs["coord_no"] = self.coord_no
                avgs["seed"] = current_seed
                avgs["replica_no"] = replica

                append_to_json([avgs], self.name + ".json")

                # seme per la replica successiva
                selectStream(255)
                next_replica_seed = getSeed()


class StatsPoint:
    def __init__(self, population, resptime, rhos):
        self.population = population,
        self.resptime = resptime,
        self.rhos = rhos,

        # diagnostics
        self.classifications = {i: 0 for i in RequestCategory},
        self.lambda_sampl = 0
        self.arrivals = 0
        self.completions = 0


class StatsPointVer:
    def __init__(self, wait=None, delay=None, service=None, nnode=None, nqueue=None, rho=None):
        # verify
        self.wait = wait
        self.delay = delay
        self.service = service
        self.nnode = nnode
        self.nqueue = nqueue
        self.rho = rho

    def flush(self):
        self.population = 0
        self.resptime = 0
        self.rhos = {i: 0 for i in NodeType},

        # diagnostics
        self.classifications = {i: 0 for i in RequestCategory},
        self.lambda_sampl = 0
        self.arrivals = 0
        self.completions = 0


def split(lista, n):
    return [lista[i:i + n] for i in range(0, len(lista), n)]
