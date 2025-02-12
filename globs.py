from enum import Enum

event_ctr = 0


def hours_to_secs(hours: int, mins: int = 0):
    return hours * 60 * 60 + mins * 60


def seconds_to_hhmm(seconds: int) -> str:
    seconds = seconds % 86400
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)

    return f"{h:02}:{m:02}:{s:02}.{ms:03}"


def seconds_to_extract_h(seconds: int) -> int:
    seconds = seconds % 86400
    h = int(seconds // 3600)
    return h


SAMPLING_TIME_INF = .1
SAMPLING_TIME_FIN = hours_to_secs(0, 10)

K = 512
b = 128
N = K * b


class GenerationMode(Enum):
    GENERATION = 0,
    TRACE = 1


generationmode = GenerationMode.GENERATION
START = hours_to_secs(6, 0)
STOP = START + hours_to_secs(24)
INFINITY = (100.0 * STOP)
globalclock = START

REPLICATIONS = 64
SEEDS_NO = 4

NUM_EDGE = 11
NUM_CLOUD = 1
NUM_CLOUD_SERVERS = NUM_EDGE
NUM_COORDINATOR = 7
