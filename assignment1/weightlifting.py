#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import math
import unittest
from collections import deque
from src.has_type import get_return_type, has_type
from src.weightlifting_data import data
from sys import gettrace, settrace
from typing import *
'''
Assignment 1, Problem 1: Weightlifting

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


__all__ = ['weightlifting_recursive', 'weightlifting_top_down',
           'weightlifting_bottom_up', 'weightlifting_list']

    
# recursion variant: The pair (w, p) strictly decreases in at each recursive call. "p" is always decremented by 1, and "w" is decremented in one of the two recursive calls.
def weightlifting_recursive(P: List[int], w: int, p: int) -> bool:
    '''
    Pre:  for 0 <= i < len(P): P[i] >= 0
    Post:
    Ex:   P = [2, 32, 234, 35, 12332, 1, 7, 56]
          weightlifting_recursive(P, 299, 8) returns True
          weightlifting_recursive(P, 11, 8) returns False
    '''
    # 1. Add base case(s)
    if w == 0: # if the weight is 0, we have found a solution
        return True
    if w < 0: # if the weight is negative, we have not found a solution and never will
        return False
    if p == 0 and w != 0: # if we have no more plates to choose from, we have not found a solution (the last w!=0 is redundant but included for clarity)
        return False
    # 2. add recursive case(s)
    return weightlifting_recursive(P, w - P[p-1], p - 1) or weightlifting_recursive(P, w, p - 1) # Either we choose the weight or we don't

# recursion variant: The pair (w, P) strictly decreases in size at each recursive call. The size of "P" is decremented by 1, and "w" is decremented in one of the two recursive calls. 
def weightlifting_top_down(P: List[int], w: int,
                           dp_matrix: List[List[None]]) -> bool:
    '''
    Pre:  for 0 <= i < len(P): P[i] >= 0
    Post: no element in dp_matrix is None
    Ex:   dp_matrix  [[None, ..., None], ..., [None, ..., None]]]
          P = [2, 32, 234, 35, 12332, 1, 7, 56]
          weightlifting_top_down(P, 299, dp_matrix) returns True
          weightlifting_top_down(P, 11, dp_matrix) returns False
    ''' 
    if w < 0: # if the weight is negative, we have not found a solution and never will
        return False
    
    P_size = len(P)
    if dp_matrix[P_size][w] is not None: # if the value is already calculated, return it
        return dp_matrix[P_size][w]
    
    if w == 0: # if the weight is 0, we have found a solution
        dp_matrix[P_size][w] = True
        return True
    
    if P_size == 0 and w != 0: # if we have no more plates to choose from, we have not found a solution (the last w!=0 is redundant but included for clarity)
        dp_matrix[P_size][w] = False
        return False    
    
    dp_matrix[P_size][w] = weightlifting_top_down(P[:-1], w - P[-1], dp_matrix) or weightlifting_top_down(P[:-1], w, dp_matrix) # Either we choose the weight or we don't    
    return dp_matrix[P_size][w]

def weightlifting_bottom_up(P: List[int], w: int,
                            dp_matrix: List[List[None]]) -> bool:
    '''
    Pre:  for 0 <= i < len(P): P[i] >= 0
    Post:
    Ex:   dp_matrix  [[None, ..., None], ..., [None, ..., None]]]
          P = [2, 32, 234, 35, 12332, 1, 7, 56]
          weightlifting_bottom_up(P, 299, dp_matrix) returns True
          weightlifting_bottom_up(P, 11, dp_matrix) returns False
    '''    

    # 1. Fill first column and row of dp_matrix
    for r in range(len(dp_matrix)):  
        dp_matrix[r][0] = True 
    for c in range(1, len(dp_matrix[0])):
        dp_matrix[0][c] = False
        
    # 2. iteratively fill rest of dp_matrix
    for r in range(1, len(dp_matrix)):
        for c in range(1, len(dp_matrix[0])):
            # if the cell above is true, the current cell is true
            if dp_matrix[r-1][c]:
                dp_matrix[r][c] = True
            # check if we could make a sub-sum of the other elements not in this subset, such that the sub-sums value is equal to (the current column) - (the last element in the current subset).
            else:
                # check if any cell on the row above has a solution for c - P[r-1]
                other_c = c - P[r-1]
                # all cells on the row above with a column-index less than this cell will have been calculated at this point, thus other_c will index a boolean value or negative 
                # check if the other_c is indexes a valid cell
                if other_c >= 0:
                    dp_matrix[r][c] = dp_matrix[r-1][other_c]
                else:
                    dp_matrix[r][c] = False
                
    # 3. return the result from the dp_matrix
    return dp_matrix[-1][-1]


def weightlifting_list(P: List[int], w: int,
                       dp_matrix: List[List[None]]) -> List[int]:
    '''
    Pre:  0 <= w
          for 0 <= i < len(P): P[i] >= 0
    Post:
    Ex:   P = [2, 32, 234, 35, 12332, 1, 7, 56]
          weightlifting_list(P, 299) returns a permutation of [2, 7, 56, 234]
          weightlifting_list(P, 11) returns []
    '''
    
    # maps each cell in the dp_matrix to a list of the plates used to reach that cell
    components_map = {}
    # 1. Fill first column and row of dp_matrix
    for r in range(len(dp_matrix)):  
        dp_matrix[r][0] = True 
    for c in range(1, len(dp_matrix[0])):
        dp_matrix[0][c] = False
    
    # 2. iteratively fill rest of dp_matrix
    for r in range(1, len(dp_matrix)):
        for c in range(1, len(dp_matrix[0])):
            # if the cell above is true, the current cell is true
            if dp_matrix[r-1][c]:
                dp_matrix[r][c] = True
                # the weights used to reach this cell is the same as the weights used to reach the cell above
                components_map[(r,c)] = components_map[(r-1,c)]
            else:
                # check if any cell on the row above has a solution for c - P[r-1]. If that is the case, the current cell is also a solution as we can add the current plate to the solution for c - P[r-1]
                other_c = c - P[r-1]
                # all cells on the row above with a column-index less than this cell will have been calculated at this point, thus other_c will index a boolean value or negative 
                # check if the other_c is indexes a cell in the dp_matrix (in this case not negative)
                if other_c >= 0:                    
                    dp_matrix[r][c] = dp_matrix[r-1][other_c]
                    # if the solution cell is on the far left of the matrix, the solution is the current plate (P[r-1])
                    if other_c == 0:
                        components_map[(r,c)] = [P[r-1]]
                    # if the solution cell is not on the far left, the solution is the solution cells weights + the current plate (P[r-1])
                    elif (r-1,other_c) in components_map:
                            components_map[(r,c)] = components_map[(r-1,other_c)] + [P[r-1]]
                else:
                    dp_matrix[r][c] = False
    
    # 3. return the result from the dp_matrix
    last_idx = (len(dp_matrix)-1,len(dp_matrix[0])-1)
    if (last_idx) in components_map:
        return components_map[last_idx]
    else:
        return []    
    


class WeightliftingTest(unittest.TestCase):
    logger = logging.getLogger('WeightLiftingTest')
    data = data
    weightlifting_recursive = weightlifting_recursive
    weightlifting_recursive_ret_type = get_return_type(weightlifting_recursive)
    weightlifting_top_down = weightlifting_top_down
    weightlifting_top_down_ret_type = get_return_type(weightlifting_top_down)
    weightlifting_bottom_up = weightlifting_bottom_up
    weightlifting_bottom_up_ret_type = get_return_type(weightlifting_bottom_up)
    weightlifting_list = weightlifting_list
    weightlifting_list_ret_type = get_return_type(weightlifting_list)

    def create_tracer() -> Tuple[Dict[str, int], Any]:
        func_calls: Dict[str, int] = dict()

        def tracer(frame, event, arg):
            f_name = frame.f_code.co_name
            if f_name not in func_calls:
                func_calls[f_name] = 0
            func_calls[f_name] += 1
        return func_calls, tracer

    def assertDpMatrix(self, dp_matrix: List[List[Any]]) -> None:
        for p in range(len(dp_matrix)):
            for w in range(len(dp_matrix[p])):
                self.assertIsNotNone(dp_matrix[p][w],
                                     f'Expected bool at dp_matrix[{p}][{w}], '
                                     'but found '
                                     f'"{type(dp_matrix[p][w]).__name__}".')

    def dp_matrix(self, P: List[int], w: int) -> List[List[None]]:
        return [[None for _ in range(w + 1)]
                for _ in range(len(P) + 1)]

    def trace_exec(self, f: Callable, *args) -> Tuple[int, Any]:
        '''
        executes the callable f with args as arguments.
        the tuple (n, res) is returned, where n is the maximum number of
        calls to any single function during the execution
        '''
        func_calls, tracer = WeightliftingTest.create_tracer()
        prev_tracer = gettrace()
        settrace(tracer)
        res = f(*args)
        settrace(prev_tracer)
        return func_calls, res

    def test_recursive(self) -> None:
        for i, instance in enumerate(self.data):
            with self.subTest(instance=i):
                P: List[int] = instance['plates'].copy()
                if len(P) > 20:
                    continue
                w: int = instance['weight']
                min_recursions: int = instance['min_recursions']
                func_calls, res = self.trace_exec(
                  WeightliftingTest.weightlifting_recursive,
                  P.copy(), w, len(P))
                self.assertTrue(
                  has_type(self.weightlifting_recursive_ret_type, res),
                  f"expected type: {self.weightlifting_recursive_ret_type} "
                  f"but {type(res)} (value: {res}) was returned.")
                func_name = WeightliftingTest.weightlifting_recursive.__name__
                self.assertEqual(len(func_calls),
                                 1,
                                 'weightlifting_recursive should only call '
                                 'itself recursively.')
                self.assertIn(func_name, func_calls)
                # The first call is not a recursive call:
                self.assertGreaterEqual(func_calls[func_name],
                                        min_recursions + 1,
                                        'weightlifting_recursive must be '
                                        'recursive ')

                self.assertEqual(res, instance['expected'])

    def test_bottom_up(self) -> None:
        for i, instance in enumerate(self.data):
            with self.subTest(instance=i):
                P: List[int] = instance['plates'].copy()
                w: int = instance['weight']
                dp_matrix = self.dp_matrix(P, w)
                func_name = (
                  WeightliftingTest.weightlifting_bottom_up.__name__)
                func_calls, res = self.trace_exec(
                  WeightliftingTest.weightlifting_bottom_up,
                  P.copy(), w, dp_matrix)
                self.assertTrue(
                  has_type(self.weightlifting_bottom_up_ret_type, res),
                  f"expected type: {self.weightlifting_bottom_up_ret_type} "
                  f"but {type(res)} (value: {res}) was returned.")
                self.assertEqual(
                  len(func_calls),
                  1,
                  'weightlifting_bottom_up should make no function calls. ' +
                  'But calls to the function(s) ' +
                  ', '.join((f'"{f}"' for f in func_calls.keys())) +
                  ' were made.')
                self.assertIn(func_name, func_calls)
                self.assertEqual(func_calls[func_name], 1)

                self.assertDpMatrix(dp_matrix)
                self.assertEqual(res, instance['expected'])

    def test_top_down(self) -> None:
        for i, instance in enumerate(self.data):
            with self.subTest(instance=i):
                P: List[int] = instance['plates'].copy()
                w: int = instance['weight']
                dp_matrix = self.dp_matrix(P, w)
                res = WeightliftingTest.weightlifting_top_down(
                  P.copy(), w, dp_matrix)
                self.assertTrue(
                  has_type(self.weightlifting_top_down_ret_type, res),
                  f"expected type: {self.weightlifting_top_down_ret_type} "
                  f"but {type(res)} (value: {res}) was returned.")
                self.assertEqual(res, instance['expected'])
                self.assertIsNotNone(dp_matrix[-1][-1],
                                     'weightlifting_top_down must use '
                                     'dp_matrix for memoisation.')
                contains_none = any(x is None
                                    for array in dp_matrix for x in array)
                self.assertTrue(contains_none,
                                'weightlifting_top_down must use the '
                                'top-down approach.')

    def test_list(self) -> None:
        if WeightliftingTest.weightlifting_list([], 0, [[None]]) is None:
            self.skipTest('weightlifting_list not implemented.')

        for i, instance in enumerate(self.data):
            with self.subTest(instance=i):
                P: List[int] = instance['plates'].copy()
                w: int = instance['weight']
                res = WeightliftingTest.weightlifting_list(
                  P.copy(), w, self.dp_matrix(P, w))
                self.assertTrue(
                  has_type(self.weightlifting_list_ret_type, res),
                  f"expected type: {self.weightlifting_list_ret_type} "
                  f"but {type(res)} (value: {res}) was returned.")
                plate_counts = {p: P.count(p) for p in set(P)}
                used_plates = {p: res.count(p) for p in set(res)}
                for p in used_plates:
                    self.assertLessEqual(used_plates[p],
                                         plate_counts.get(p, 0),
                                         f'plate {p} occurs {used_plates[p]} '
                                         'times in the solution, but only '
                                         f'{plate_counts[p]} times in P')

                if instance['expected']:
                    self.assertEqual(sum(res), instance['weight'],
                                     'The sum of the returned list of plates '
                                     'does not equal the expected weight.')
                else:
                    self.assertListEqual(res, list())


if __name__ == '__main__':
    # Set logging config to show debug messages:
    logging.basicConfig(level=logging.DEBUG)
    # run unit tests (failfast=True stops testing after the first failed test):
    unittest.main(failfast=True)
