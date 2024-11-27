#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import math
import unittest
from collections import defaultdict, deque
from src.difference_data import data
from src.has_type import get_return_type, has_type
from string import ascii_lowercase
from typing import *
'''
Assignment 2, Problem 1: Search String Replacement

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


__all__ = ['min_difference', 'min_difference_align']


def min_difference(s: str, r: str, R: Dict[str, Dict[str, int]]) -> int:
    '''
    Pre:  For all characters c in s and k in r,
          elements R[c][k], R[c]['-'], R['-'][k], and R['-']['-'] exists.
    Post:
    Ex:   Let R be the resemblance matrix where every change and skip
          costs 1
          min_difference('dinamck', 'dynamic', R) --> 3
    '''
    # To get the resemblance between two letters, use code like this:
    # difference = R['a']['b']
    
    dp_dict = {}
    def min_diff_aux(s: str, r: str) -> int:
        # if one of the strings is empty, the cost is the sum of the differences between the "remaining" characters and the '-' character    
        if not s or not r:
            cost = 0
            for char in s + r:
                if char in s:
                    cost += R[char]['-']
                else:
                    cost += R['-'][char]
            return cost

        if (s, r) in dp_dict:
            return dp_dict[(s, r)]

        #recursive case:
        if s[0] == r[0]:
            return min_diff_aux(s[1:], r[1:])
        else:
            # the cost of changing the character
            change = R[s[0]][r[0]] + min_diff_aux(s[1:], r[1:])
            #the cost of skipping a character in s
            skip_s = R[s[0]]['-'] + min_diff_aux(s[1:], r)
            #the cost of skipping a character in r
            skip_r = R['-'][r[0]] + min_diff_aux(s, r[1:])
            # the minimum cost of the three options
            min_diff = min(skip_s, skip_r, change)
            dp_dict[(s, r)] = min_diff
            return min_diff
    
    return min_diff_aux(s, r)


def min_difference_align(s: str, r: str,
                         R: Dict[str, Dict[str, int]]) -> Tuple[int, str, str]:
    '''
    Pre:  For all characters c in s and k in r,
          elements R[c][k], R[c]['-'], R['-'][k], and R['-']['-'] exists.
    Post:
    Ex:   Let R be the resemblance matrix where every change and skip
          costs 1
          min_difference_align('dinamck', 'dynamic', R) -->
                                    3, 'dinam-ck', 'dynamic-'
                                 or 3, 'dinamck', 'dynamic'
    '''
    print(f"s: {s}, r: {r}")
    
    dp_dict = {}
    def min_diff_align_aux(s: str, r: str) -> Tuple[int,str,str]:
        # if one of the strings is empty, the cost is the sum of the differences between the "remaining" characters and the '-' character    
        if not s or not r:
            cost = 0
            for char in s + r:
                if char in s:
                    cost += R[char]['-']
                else:
                    cost += R['-'][char]
            return (cost, s, r)

        if (s, r) in dp_dict:
            return dp_dict[(s, r)]

        #recursive case:
        if s[0] == r[0]:
            return min_diff_align_aux(s[1:], r[1:])
        else:
            # TUPLES ARE IMMUTABLE IN PYTHON APPARENTLY
            
            # the cost of changing the character
            change: Tuple[int,str,str] = min_diff_align_aux(s[1:], r[1:])
            new_change = (R[s[0]][r[0]] + change[0], change[1], change[2])
            
            #the cost of skipping a character in s
            skip_s: Tuple[int,str,str] = min_diff_align_aux(s[1:], r)
            new_skip_s = (R[s[0]]['-'] + skip_s[0], skip_s[1], skip_s[2])
            
            #the cost of skipping a character in r
            skip_r: Tuple[int,str,str] = min_diff_align_aux(s, r[1:])
            new_skip_r = (R['-'][r[0]] + skip_r[0], skip_r[1], skip_r[2])
            
            print(f"new_change: {new_change}")
            print(f"new_skip_s: {new_skip_s}")
            print(f"new_skip_r: {new_skip_r}")
            
            # the minimum cost of the three options
            min_diff = min(new_skip_s, new_skip_r, new_change, key=lambda x: x[0])
            dp_dict[(s, r)] = min_diff
            return min_diff
    
    rval = min_diff_align_aux(s, r)
    print(f"rval: {rval}\n")
    return rval

def qwerty_distance() -> Dict[str, Dict[str, int]]:
    '''
    Generates a QWERTY Manhattan distance resemblance matrix

    Costs for letter pairs are based on the Manhattan distance of the
    corresponding keys on a standard QWERTY keyboard.
    Costs for skipping a character depends on its placement on the keyboard:
    adding a character has a higher cost for keys on the outer edges,
    deleting a character has a higher cost for keys near the middle.

    Usage:
        R = qwerty_distance()
        R['a']['b']  # result: 5
    '''
    R: Dict[str, Dict[str, int]] = defaultdict(dict)
    R['-']['-'] = 0
    zones = ['dfghjk', 'ertyuislcvbnm', 'qwazxpo']
    keyboard = ['qwertyuiop', 'asdfghjkl', 'zxcvbnm']
    for row, content in enumerate(zones):
        for char in content:
            R['-'][char] = row + 1
            R[char]['-'] = 3 - row
    for a, b in ((a, b) for b in ascii_lowercase for a in ascii_lowercase):
        row_a, pos_a = next(
            (row, content.index(a))
            for row, content in enumerate(keyboard) if a in content
        )
        row_b, pos_b = next(
            (row, content.index(b))
            for row, content in enumerate(keyboard) if b in content
        )
        R[a][b] = abs(row_b - row_a) + abs(pos_a - pos_b)
    return R


def simple_distance() -> Dict[str, Dict[str, int]]:
    alphabet = ascii_lowercase + '-'
    return {a: {b: (0 if a == b else 1) for b in alphabet} for a in alphabet}


class MinDifferenceTest(unittest.TestCase):
    '''
    Test Suite for search string replacement problem

    Any method named 'test_something' will be run when this file is
    executed. Use the sanity check as a template for adding your own test
    cases if you wish.
    (You may delete this class from your submitted solution.)
    '''
    logger = logging.getLogger('MinDifferenceTest')
    data = data
    min_difference = min_difference
    min_difference_ret_type = get_return_type(min_difference)
    min_difference_align = min_difference_align
    min_difference_align_ret_type = get_return_type(min_difference_align)

    def assertMinDifference(self, s: str, r: str, difference: int,
                            R: Dict[str, Dict[str, int]]) -> None:
        res_difference = MinDifferenceTest.min_difference(s, r, R)
        self.assertTrue(
                  has_type(self.min_difference_ret_type, res_difference),
                  f"expected type: {self.min_difference_ret_type} "
                  f"but {type(res_difference)} (value: {res_difference}) was "
                  "returned.")
        self.assertEqual(res_difference, difference,
                         f'Difference between s="{s}" and r="{r}" was '
                         f'{res_difference}, {difference} expected.')

    def assertMinDifferenceAlign(self, s: str, r: str, difference: int,
                                 R: int,
                                 solutions: Union[Set[Tuple[str, str]],
                                                  None] = None) -> None:
        t = MinDifferenceTest.min_difference_align(s, r, R)
        self.assertTrue(
                  has_type(self.min_difference_align_ret_type, t),
                  f"expected type: {self.min_difference_align_ret_type} "
                  f"but {type(t)} (value: {t}) was returned.")
        res_difference, res_s, res_r = t
        self.assertEqual(res_difference, difference,
                         f'Difference between s="{s}" and r="{r}" was '
                         f'{res_difference}, {difference} expected.')

        self.assertEqual(len(res_s), len(res_r), f'len("{s}") != len("{r}")')

        res_sum = sum((R[res_s[i]][res_r[i]] for i in range(len(res_s))))

        self.assertEqual(res_sum, difference,
                         f'Difference for s="{s}", r={r}, res_r={res_s}, and '
                         f'r="{res_r}" was summed to {res_sum}, {difference}'
                         ' expected.')
        if solutions is not None:
            self.assertIn((res_s, res_r), solutions)

    def test_min_difference(self) -> None:
        for i, instance in enumerate(MinDifferenceTest.data):
            with self.subTest(instance=i):
                R = (qwerty_distance() if instance['resemblance'] == 'qwerty'
                     else simple_distance())
                self.assertMinDifference(instance['s'], instance['r'],
                                         instance['difference'], R)

    def test_min_difference_align(self) -> None:
        if MinDifferenceTest.min_difference_align('a', 'a',
                                                  qwerty_distance()) is None:
            self.skipTest('min_difference_align not implemented.')

        for i, instance in enumerate(MinDifferenceTest.data):
            with self.subTest(instance=i):
                R = (qwerty_distance() if instance['resemblance'] == 'qwerty'
                     else simple_distance())
                self.assertMinDifferenceAlign(instance['s'], instance['r'],
                                              instance['difference'], R,
                                              instance['solutions'])


if __name__ == '__main__':
    # Set logging config to show debug messages:
    logging.basicConfig(level=logging.DEBUG)
    # run unit tests (failfast=True stops testing after the first failed test):
    unittest.main(failfast=True)
