import pytest_check as check

from mypackage.mymodule import my_perfect_function


def test_fizz():
    """should return Fizz when divisible by 3 and not 5."""
    check.equal(my_perfect_function(3), "Fizz")


def test_buzz():
    """should return Buzz when divisible by 5 and not 3."""
    check.equal(my_perfect_function(5), "Buzz")


def test_fizzbuzz():
    """should return FizzBuzz when divisible by both 5 and 3."""
    check.equal(my_perfect_function(15), "FizzBuzz")


def test_num():
    """should return num when divisible by neither of 5 and 3."""
    check.equal(my_perfect_function(13), "13")
