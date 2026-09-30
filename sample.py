"""Sample module for AST parser testing."""
import os
import sys

def calculate_sum(a, b):
    """Calculate the sum of two numbers."""
    result = a + b
    if result > 100:
        print(f"Large sum: {result}")
    else:
        print(f"Small sum: {result}")
    return result


def main():
    """Entry point of the sample module."""
    x = 10
    y = 20
    total = calculate_sum(x, y)
    print(f"Total: {total}")
    config = {"debug": False}
    return total


class DataProcessor:
    """Process data with various transformation methods."""

    def __init__(self, data):
        self.data = data

    def process(self):
        """Process the data by doubling each element."""
        return [d * 2 for d in self.data]

    def filter_even(self):
        """Filter and return only even numbers."""
        return [d for d in self.data if d % 2 == 0]

    def summary(self):
        """Return summary statistics of the data."""
        total = sum(self.data)
        count = len(self.data)
        avg = total / count if count else 0
        result = {"total": total, "count": count, "avg": avg}
        return result
