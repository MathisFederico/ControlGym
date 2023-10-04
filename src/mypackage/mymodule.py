"""Example of a module."""


def my_perfect_function(num: int) -> str:
    """Example of a function.

    This function has a complete docstring, uses type hints, and is unit tested.
    Of course, this is a simple case, but keeping things simple is the way to code.

    This function is an implementation of the classic fizzbuzz example.

    Args:
        text (str): Text to compute the half-lenght of.

    Returns:
        int: Half-lenght of the input text.
    """
    out = ""
    if num % 3 == 0:
        out += "Fizz"
    if num % 5 == 0:
        out += "Buzz"
    if out == "":
        out = str(num)
    return out
