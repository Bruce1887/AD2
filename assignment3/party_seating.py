#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import math
import unittest
from collections import deque
from src.has_type import get_return_type, has_type
from src.party_seating_data import data
from typing import *
'''
Assignment 3, Problem 2: Party Seating

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


__all__ = ['party']


def party(known: List[Set[int]]) -> Tuple[bool, Set[int], Set[int]]:
    '''
    Pre:  for all i in 0 <= len(known), j in known[i],
          value i is in set known[j]
    Post:
    Ex:   party([{1, 2}, {0}, {0}]) = True, {0}, {1, 2}
    '''
    return False, set(), set()


class PartySeatingTest(unittest.TestCase):
    '''
    Test suite for party seating problem
    '''
    logger = logging.getLogger('PartySeatingTest')
    data = data
    party = party
    party_ret_value = get_return_type(party)

    def assertKnown(self, known: List[Set[int]], A: Set[int],
                    B: Set[int]) -> None:
        self.assertEqual(len(A) + len(B), len(known),
                         f"wrong number of guests: expected {len(known)} "
                         f"guests, tables hold {len(A)} and {len(B)} guests "
                         "respectively")
        for g in range(len(known)):
            self.assertTrue(g in A or g in B, f"Guest {g} not seated anywhere")

        for a1, a2 in ((a1, a2) for a2 in A for a1 in A):
            self.assertNotIn(a2, known[a1],
                             f"Guests {a1} and {a2} seated together, and "
                             "know each other")

        for b1, b2 in ((b1, b2) for b2 in B for b1 in B):
            self.assertNotIn(b2, known[b1],
                             f"Guests {b1} and {b2} seated together, and "
                             "know each other")

    def test_party(self) -> None:
        for i, instance in enumerate(PartySeatingTest.data):
            with self.subTest(instance=i):
                known = instance['known']
                expected = instance['expected']

                t = PartySeatingTest.party(known)
                self.assertTrue(
                  has_type(self.party_ret_value, t),
                  f"expected type: {self.party_ret_value} "
                  f"but {type(t)} (value: {t}) was returned.")
                success, A, B = t

                if not expected:
                    self.assertFalse(success)
                    self.assertSetEqual(A, set())
                    self.assertSetEqual(B, set())
                else:
                    self.assertKnown(known, A, B)


if __name__ == '__main__':
    # Set logging config to show debug messages:
    logging.basicConfig(level=logging.DEBUG)
    # run unit tests (failfast=True stops testing after the first failed test):
    unittest.main(failfast=True)
