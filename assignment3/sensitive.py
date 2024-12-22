#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import math
import unittest
from collections import deque
from src.graph import Graph
from src.has_type import get_return_type, has_type
from src.sensitive_data import data
from typing import *
'''
Assignment 3, Problem 1: Controlling the Maximum Flow

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


__all__ = ['sensitive']


def sensitive(G: Graph, s: str, t: str) -> Union[Tuple[str, str],
                                                 Tuple[None, None]]:
    '''
    Pre:
    Post:
    Ex:   sensitive(g1, 'a', 'f') = ('b', 'd')
    '''
    
    residual_graph: Graph = Graph(True)
    
    for u, v in G.edges:
        flow = G.flow(u, v) or 0
        residual = (G.capacity(u, v) or 0) - flow
        if residual > 0:
            residual_graph.add_edge(u, v, capacity = residual)
        if flow > 0:
            residual_graph.add_edge(v, u, capacity = flow)
    
    if not residual_graph.edges:
        return None, None        

    def find_visited(G: Graph, s: str, t: str) -> set:    
        stack = [s]
        visited = {s}

        while stack:
            u = stack.pop()                          
            for v in G.neighbors(u): 
                capacity = G.capacity(u, v) or 0
                flow = G.flow(u, v) or 0
                if v not in visited and capacity > flow:
                    visited.add(v)
                    stack.append(v)
        return visited     
    
    visited = find_visited(residual_graph, s, t)        
    
    for node in visited:        
        for neighbor in G.neighbors(node):            
            if neighbor not in visited:                
                return node, neighbor # sensitive edge found as per the max-cut min-flow theorem

    return None, None


class SensitiveTest(unittest.TestCase):
    '''
    Test suite for the sensitive edge problem
    '''
    logger = logging.getLogger('SensitiveTest')
    data = data
    sensitive = sensitive
    sensitive_ret_type = get_return_type(sensitive)

    def test_sensitive(self) -> None:
        for i, instance in enumerate(SensitiveTest.data):
            with self.subTest(instance=i):
                graph = instance['digraph'].copy()
                t = SensitiveTest.sensitive(graph, instance['source'],
                                            instance['sink'])
                self.assertTrue(
                  has_type(self.sensitive_ret_type, t),
                  f"expected type: {self.sensitive_ret_type} "
                  f"but {type(t)} (value: {t}) was returned.")
                u, v = t
                if len(instance['sensitive_edges']) == 0:
                    self.assertEqual((u, v), (None, None),
                                     'Network contains no sensitive edges.')
                    continue

                self.assertIn(u, graph, f'Node "{u}" not in network.')
                self.assertIn(v, graph, f'Node "{v}" not in network.')
                self.assertIn((u, v), graph,
                              f'Edge ("{u}", "{v}") not in network.')
                self.assertIn((u, v), instance['sensitive_edges'],
                              f'Edge ("{u}", "{v}") is not sensitive.')


if __name__ == '__main__':
    # Set logging config to show debug messages:
    logging.basicConfig(level=logging.DEBUG)
    # run unit tests (failfast=True stops testing after the first failed test):
    unittest.main(failfast=True)
