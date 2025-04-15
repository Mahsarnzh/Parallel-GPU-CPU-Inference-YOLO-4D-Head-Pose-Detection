from collections import defaultdict


REGISTRY = defaultdict(list)

def register_module(group_name:str):
    def class_decorator(cls):
        REGISTRY[group_name].append(cls)
        return cls
    return class_decorator