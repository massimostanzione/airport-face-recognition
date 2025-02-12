import math
from typing import List

from globs import INFINITY, hours_to_secs


class Track:
    def __init__(self):
        self.node = 0.0
        self.queue = 0.0
        self.service = 0.0


class TimeSlot:
    def __init__(self, name, start, traffic_perc):
        self.name = name
        self.start = start
        self.traffic_perc = traffic_perc


def _ts_lookup(time, slots, lookup_next):
    norm = time % hours_to_secs(24)
    cycle_number = math.floor(time / hours_to_secs(24))

    for i, ts in enumerate(slots):
        next_start = slots[i + 1].start if i + 1 < len(slots) else float('inf')
        if ts.start <= norm < next_start:
            selected_slot = ts if not lookup_next else slots[(i + 1) % len(slots)]
            return TimeSlot(
                selected_slot.name,
                selected_slot.start + ((cycle_number + 1) * hours_to_secs(24)),
                # Qui aggiorniamo correttamente lo start
                selected_slot.traffic_perc
            )

    return None


def lookup_timeslot(time, slots: List[TimeSlot]):
    return _ts_lookup(time, slots, False)


def lookup_timeslot_next(time, slots: List[TimeSlot]):
    return _ts_lookup(time, slots, True)


class Time:
    def __init__(self, cardty: int):
        self.current = 0.0
        global globalclock
        globalclock = self.current
        self.next = 0.0
        self.last = [0.0] * (cardty)
        self.arrival = 0.0
        self.completion = [INFINITY] * (cardty)
