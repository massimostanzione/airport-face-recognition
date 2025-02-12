from enum import Enum

probabilities = [0.05, 0.10, 0.40, 0.05, 0.40]


class RequestCategory(Enum):
    DANGEROUS = 'dangerous',
    SUSPECT = 'suspect',
    REGULAR = 'regular',
    POOR_QUALITY = 'poor-quality',
    UNKNOWN = 'unknown'

    def is_priority(self):
        priority_cats = [RequestCategory.DANGEROUS, RequestCategory.SUSPECT]
        return self in priority_cats


class Request:
    def __init__(self, generation_time: float, category: RequestCategory = None):
        self.category = category  # La categoria inizialmente è None
        self.generation_time = generation_time
        self.double_processing = False

    def is_classC(self) -> bool:
        return self.double_processing


def categorize_request(request: Request) -> Request:
    global pty
    old_cat = request.category
    import random
    selected_category = random.choices(categories, probabilities)[0]
    request.category = selected_category

    if old_cat is None:
        index = categories.index(selected_category)
        pty[index] += 1
    return request


pty = [0.0] * len([c for c in RequestCategory])
categories = [
    RequestCategory.DANGEROUS,
    RequestCategory.SUSPECT,
    RequestCategory.REGULAR,
    RequestCategory.POOR_QUALITY,
    RequestCategory.UNKNOWN
]
