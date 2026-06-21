# python_demo.py
from functools import lru_cache

class DataProcessor:
    """A simple processor class in Python."""
    def __init__(self, name: str):
        self.name = name

    def process_data(self, items: list) -> list:
        """Processes the list of items by reversing them."""
        return [str(item)[::-1] for item in items]

@lru_cache(maxsize=128)
def calculate_metrics(values: tuple[float, ...]) -> dict[str, float]:
    """Decorated function to compute summary statistics."""
    if not values:
        return {"mean": 0.0, "sum": 0.0}
    return {
        "mean": sum(values) / len(values),
        "sum": sum(values)
    }
