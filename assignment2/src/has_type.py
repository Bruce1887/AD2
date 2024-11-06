from typing import Any, Dict, List, Optional, Set, Tuple, Union, get_origin


__all__ = ['get_return_type', 'has_type']


def has_args(annotation: Any, min_len: int) -> bool:
    if not hasattr(annotation, '__args__'):
        return False
    if not hasattr(annotation.__args__, '__len__'):
        return False
    return len(annotation.__args__) >= min_len


def has_type(annotation: Any, value: Any) -> bool:
    if get_origin(annotation) in {Dict, dict}:
        return (isinstance(value, dict) and
                has_args(annotation, 2) and
                all((has_type(annotation.__args__[0], k) and
                     has_type(annotation.__args__[1], v)
                     for k, v in value.items())))
    elif get_origin(annotation) in {List, list}:
        return (isinstance(value, list) and
                has_args(annotation, 1) and
                all((has_type(annotation.__args__[0], v)
                     for v in value)))
    elif get_origin(annotation) == Optional:
        return value is None or has_type(annotation.__args__[0], value)
    elif get_origin(annotation) in {Set, set}:
        return (isinstance(value, set) and
                has_args(annotation, 1) and
                all((has_type(annotation.__args__[0], v)
                     for v in value)))
    elif get_origin(annotation) in {Tuple, tuple}:
        return (isinstance(value, tuple) and
                has_args(annotation, 0) and
                len(annotation.__args__) == len(value) and
                all((has_type(a, v)
                     for a, v in zip(annotation.__args__, value))))
    elif get_origin(annotation) == Union:
        return (has_args(annotation, 0) and
                any((has_type(u, value)
                     for u in annotation.__args__)))
    elif annotation is None:
        return value is None
    else:
        return isinstance(value, annotation)


def get_return_type(fun: callable) -> Any:
    '''
    Return the return type of the annotations of fun
    '''
    if not hasattr(fun, '__annotations__'):
        return None
    if not isinstance(fun.__annotations__, dict):
        return None
    return fun.__annotations__.get('return', None)
