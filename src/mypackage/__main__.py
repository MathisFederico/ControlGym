"""Entry module of the package.

You can run this script using:
python -m mypackage

"""

import mypackage


def main() -> None:
    """Entry point of the package"""
    print(f"Hello world from {mypackage.__name__} ({mypackage.__doc__})")


if __name__ == "__main__":
    main()
