from enum import Enum


class ContextType(str, Enum):
    SLICED = "sliced"
    FULL = "full"
