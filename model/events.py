from enum import Enum

from model.requests import Request
from globs import event_ctr


class EventType(Enum):
    ARRIVAL_EDGE = 0,
    COMPLETION_EDGE = 1,
    ARRIVAL_CLOUD = 2,
    COMPLETION_CLOUD = 3,
    FEEDBACK_FROM_CLOUD = 4,
    ARRIVAL_COORDINATOR = 5,
    COMPLETION_COORDINATOR = 6,
    CAMBIO_FASCIA_ORARIA = 7


class Event:
    def __init__(self, arrival_time, type: EventType, request: Request = None, server_id: int = None):
        global event_ctr
        event_ctr += 1
        self.id = event_ctr
        self.arrival_time = arrival_time
        self.event_type = type
        self.request = request
        self.server_id = server_id
