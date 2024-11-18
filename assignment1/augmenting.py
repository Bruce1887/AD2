#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import math
import unittest
from collections import deque
from src.augmenting_data import data
from src.graph import Graph
from src.has_type import get_return_type, has_type
from typing import *
'''
Assignment 1, Problem 2: Augmenting Path Detection in Network Graphs

Student Name: Edvin Bruce
'''

'''
Copyright: justin.pearson@it.uu.se and his teaching assistants, 2024.

This file is part of course 1DL231 at Uppsala University, Sweden.

Permission is hereby granted only to the registered students of that
course to use this file, for a homework assignment.

The copyright notice and permission notice above shall be included in
all copies and extensions of this file, and those are not allowed to
appear publicly on the internet, both during a course instance and
forever after.
'''
# If your solution needs a queue, then you can use deque.

# If you need to log information during tests, execution, or both,
# then you can use this library:
# Basic example:
#   logger = logging.getLogger('put name here')
#   a = 5
#   logger.debug(f'a = {a}')


__all__ = ['augmenting', 'augmenting_extended']


def augmenting(G: Graph, s: str, t: str) -> bool:
    '''
    Pre:
    Post:
    Ex:
    '''
        
    def aux(G: Graph, visited: List[str], current: str) -> bool:
        if current == t:
            return True

        visited.append(current)
        for neighbour in G._edges[current]:
            if neighbour not in visited and G.flow(current, neighbour) < G.capacity(current, neighbour):
                if aux(G, visited, neighbour):
                    return True
        return False
    
    return aux(G, [], s)

def augmenting_extended(G: Graph, s: str, t: str) -> Tuple[bool, List[Tuple[str, str]]]:
    stack = [(s, [])]  # used to store the current node and the path to it, we add to this when we visit a new node
    visited = set() # used to store the visited nodes

    while stack:
        current, path_edges = stack.pop()
        if current in visited:
            continue
        visited.add(current)

        if current == t:
            return (True, path_edges)

        for neighbor in G._edges[current]:
            if neighbor not in visited and G.flow(current, neighbor) < G.capacity(current, neighbor):
                stack.append((neighbor, path_edges + [(current, neighbor)]))

    return (False, [])

class AugmentingTest(unittest.TestCase):
    '''
    Test Suite for augmenting path dectection problem

    Any method named 'test_something' will be run when this file is executed.
    Use the sanity check as a template for adding your own test cases if you
    wish. (You may delete this class from your submitted solution.)
    '''
    logger = logging.getLogger('AugmentingTest')
    data = data
    augmenting = augmenting
    augmenting_ret_type = get_return_type(augmenting)
    augmenting_extended = augmenting_extended
    augmenting_extended_ret_type = get_return_type(augmenting_extended)

    def assertIsAugmentingPath(self, graph: Graph, s: str, t: str,
                               path: List[Tuple[str, str]]) -> None:
        if len(path) == 0:
            self.fail('The path should be non-empty.')

        self.assertEqual(path[0][0], s,
                         f'The path does not start at the source {s}.')
        self.assertEqual(path[-1][1], t,
                         f'The path does not end at the sink {t}.')
        for u, v in path:
            self.assertIn((u, v), graph,
                          f'The edge {(u, v)} of the path does not exist in '
                          'the graph.')

        for i in range(1, len(path)):
            self.assertEqual(path[i - 1][1], path[i][0],
                             f'The end of edge {path[i - 1]} does not match '
                             f'the start of the next edge {path[i]}.')

        self.assertEqual(len(path), len(set(path)),
                         'The path contains duplicates of one or more edges.')

        for u, v in path:
            self.assertLess(graph.flow(u, v), graph.capacity(u, v),
                            f'The flow is not less than the capacity for '
                            f'edge {(u, v)}.')

    def test_augmenting(self) -> None:
        for i, instance in enumerate(AugmentingTest.data):
            with self.subTest(instance=i):
                graph = instance['digraph'].copy()
                found = AugmentingTest.augmenting(
                    graph, instance['source'], instance['sink'])
                m = '' if instance['expected'] else ' not'
                self.assertTrue(
                  has_type(self.augmenting_ret_type, found),
                  f'expected type: {self.augmenting_ret_type} '
                  f'but {type(found)} (value: {found}) was returned.')
                self.assertEqual(found, instance['expected'],
                                 f'The network should{m} contain an '
                                 f'augmenting path.')

    def test_augmenting_extended(self) -> None:
        instance = AugmentingTest.data[0]
        if AugmentingTest.augmenting_extended(instance['digraph'].copy(),
                                              instance['source'],
                                              instance['sink']) is None:
            self.skipTest('augmenting_extended not implemented.')

        for i, instance in enumerate(AugmentingTest.data):
            with self.subTest(instance=i):
                graph = instance['digraph'].copy()
                t = AugmentingTest.augmenting_extended(graph,
                                                       instance['source'],
                                                       instance['sink'])
                self.assertTrue(
                  has_type(self.augmenting_extended_ret_type, t),
                  f'expected type: {self.augmenting_extended_ret_type} '
                  f'but {type(t)} (value: {t}) was returned.')
                found, path = t

                m = '' if instance['expected'] else ' not'
                self.assertEqual(found, instance['expected'],
                                 f'The network should{m} contain an '
                                 f'augmenting path.')
                if instance['expected']:
                    self.assertIsAugmentingPath(instance['digraph'].copy(),
                                                instance['source'],
                                                instance['sink'],
                                                path)
                else:
                    self.assertListEqual(path, [])


if __name__ == '__main__':
    # Set logging config to show debug messages:
    logging.basicConfig(level=logging.DEBUG)
    # run unit tests (failfast=True stops testing after the first failed test):
    unittest.main(failfast=True)
