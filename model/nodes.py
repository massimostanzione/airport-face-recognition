from collections import deque


class Node():
    def __init__(self, available_servers=1):
        self.type = None
        self.available_servers = available_servers
        self.index = 0
        self.number = 0

    def schedule_next(self):
        pass


class NodeMM1(Node):
    def __init__(self):
        super().__init__()
        self.reqeusts = []

    def schedule_next(self):
        pass


class NodePrio2(Node):
    def __init__(self):
        super().__init__()
        self.reqs_high = deque()
        self.reqs_low = deque()
