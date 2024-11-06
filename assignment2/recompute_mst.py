#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import math
import unittest
from collections import deque
from src.graph import Graph
from src.has_type import get_return_type, has_type
from src.recompute_mst_data import data
from typing import *
'''
Assignment 2, Problem 2: Recomputing a Minimum Spanning Tree

Student Name:
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


__all__ = ['update_MST_1', 'update_MST_2', 'update_MST_3', 'update_MST_4']


def update_MST_1(G: Graph, T: Graph, e: Tuple[str, str],
                 weight: int) -> Union[Tuple[None, None],
                                       Tuple[Tuple[str, str],
                                             Tuple[str, str]]]:
    '''
    Pre:  T is a spanning tree of G.
          Edge e is in G.
          Edge e is not in T.
          weight > G.weight(u, v)
    Post: T is a minimum spanning tree of G.
    '''
    (u, v) = e
    assert (e in G and e not in T and weight > G.weight(u, v))


def update_MST_2(G: Graph, T: Graph, e: Tuple[str, str],
                 weight: int) -> Union[Tuple[None, None],
                                       Tuple[Tuple[str, str],
                                             Tuple[str, str]]]:
    '''
    Pre:  T is a spanning tree of G.
          Edge e is in G.
          Edge e is not in T.
          weight < G.weight(u, v)
    Post: T is a minimum spanning tree of G.
    '''
    (u, v) = e
    assert (e in G and e not in T and weight < G.weight(u, v))


def update_MST_3(G: Graph, T: Graph, e: Tuple[str, str],
                 weight: int) -> Union[Tuple[None, None],
                                       Tuple[Tuple[str, str],
                                             Tuple[str, str]]]:
    '''
    Pre:  T is a spanning tree of G.
          Edge e is in G.
          Edge e is in T.
          weight < G.weight(u, v)
    Post: T is a minimum spanning tree of G.
    '''
    (u, v) = e
    assert (e in G and e in T and weight < G.weight(u, v))


def update_MST_4(G: Graph, T: Graph, e: Tuple[str, str],
                 weight: int) -> Union[Tuple[None, None],
                                       Tuple[Tuple[str, str],
                                             Tuple[str, str]]]:
    '''
    Pre:  T is a spanning tree of G.
          Edge e is in G.
          Edge e is in T.
          weight > G.weight(u, v)
    Post:
    '''
    (u, v) = e
    assert (e in G and e in T and weight > G.weight(u, v))


class RecomputeMstTest(unittest.TestCase):
    '''
    Test Suite for minimum spanning tree problem

    Any method named 'test_something' will be run when this file is
    executed. You may add your own test cases if you wish.
    (You may delete this class from your submitted solution.)
    '''
    logger = logging.getLogger('RecomputeMstTest')
    data = data
    update_MST = [update_MST_1, update_MST_2, update_MST_3, update_MST_4]
    update_MST_ret_type = [get_return_type(f) for f in update_MST]

    def assertReturnedValue(self, unmodified_g: Graph, modified_g: Graph,
                            unmodified_t: Graph, modified_t: Graph,
                            t: Union[Tuple[None, None],
                                     Tuple[Tuple[str, str],
                                           Tuple[str, str]]]) -> None:
        e_removed, e_added = t
        if e_removed is None or e_added is None:
            return
        if e_removed == e_added:
            return
        self.assertLessEqual(unmodified_g.weight(*e_removed),
                             unmodified_g.weight(*e_added),
                             'The weight of the added edge cannot be less '
                             'than the weight of the removed edge in the'
                             'unmodified graph.')
        self.assertGreaterEqual(modified_g.weight(*e_removed),
                                modified_g.weight(*e_added),
                                'The weight of the added edge cannot be '
                                'greater than the weight of the removed edge '
                                'in the modified graph.')
        self.assertIn(e_removed, unmodified_t,
                      'The removed edge is not in unmodified MST.')
        self.assertNotIn(e_removed, modified_t,
                         'The removed edge is in the modified MST.')
        self.assertNotIn(e_added, unmodified_t,
                         'The added edge is in the unmodified MST.')
        self.assertIn(e_added, modified_t,
                      'The added edge is not in the modified MST.')

    def assertUndirectedEdgesEqual(self, actual: List[Tuple[str, str]],
                                   expected: List[Tuple[str, str]]) -> None:
        self.assertListEqual(
            sorted(((min(u, v), max(u, v)) for u, v in actual)),
            sorted(((min(u, v), max(u, v)) for u, v in expected)),
            'The edges in the MST does not match the expected edges.')

    def assertEdgesInExpectedGraph(self, expected_graph: List[Tuple[str, str]],
                                   graph: Graph) -> None:
        for e in expected_graph:
            self.assertIn(e, graph,
                          f'The edge {e} is not in the expected graph.')

    def assertTreeIsConnected(self, tree: Graph) -> None:
        if len(tree.nodes) == 0:
            return
        visited = set()
        queue = deque([tree.nodes[0]])
        while len(queue) > 0:
            u = queue.popleft()
            visited.add(u)
            for v in tree.neighbors(u):
                if v not in visited:
                    queue.append(v)
        self.assertSetEqual(visited, set(tree.nodes),
                            'The tree is not connected.')

    def assertGraphsEqual(self, actual: Graph, expected: Graph) -> None:
        self.assertEqual(len(actual.edges), len(expected.edges),
                         'The graphs have different number of edges.')
        for u, v in actual.edges:
            self.assertEqual(actual.weight(u, v), expected.weight(u, v),
                             f'The weight of edge ({u}, {v}) is not equal in '
                             'the actual and expected graphs.')

    def test_mst(self) -> None:
        is_implemented = [True] * len(self.update_MST)

        for i, update_MST in enumerate(self.update_MST, start=1):
            instance = data[0]
            graph = instance['graph'].copy()
            tree = instance['mst'].copy()
            u, v = instance['solutions'][i - 1]['edge']
            weight = instance['solutions'][i - 1]['weight']
            if update_MST(graph, tree, (u, v), weight) is None:
                self.logger.warning(f'update_MST_{i} not implemented.')
                is_implemented[i - 1] = False
                continue

            # test update_MST_{i}:
            for j, instance in enumerate(self.data):
                # test update_MST_{i} using instance {j}:
                with self.subTest(case=i, instance=j):
                    graph: Graph = instance['graph'].copy()
                    tree: Graph = instance['mst'].copy()
                    u, v = instance['solutions'][i - 1]['edge']
                    weight = instance['solutions'][i - 1]['weight']
                    expected = instance['solutions'][i - 1]['expected']
                    expected_graph: Graph = instance['graph'].copy()
                    expected_graph.set_weight(u, v, weight)
                    t = update_MST(graph, tree, (u, v), weight)
                    self.assertTrue(
                      has_type(self.update_MST_ret_type[i - 1], t),
                      "expected type: "
                      f"{get_return_type(self.update_MST_ret_type[i - 1])} "
                      f"but {type(t)} (value: {t}) was returned.")
                    self.assertReturnedValue(instance['graph'], graph,
                                             instance['mst'], tree,
                                             t)
                    self.assertUndirectedEdgesEqual(tree.edges, expected)
                    self.assertEdgesInExpectedGraph(tree.edges, expected_graph)
                    self.assertTreeIsConnected(tree)
                    self.assertGraphsEqual(graph, expected_graph)

        num_implemented_funs = sum((1 if b else 0 for b in is_implemented))
        self.assertGreaterEqual(
          num_implemented_funs, 2,
          'At least two of the four update_MST functions must be ' +
          'implemented.')


if __name__ == '__main__':
    # Set logging config to show debug messages:
    logging.basicConfig(level=logging.DEBUG)
    # run unit tests (failfast=True stops testing after the first failed test):
    unittest.main(failfast=True)
