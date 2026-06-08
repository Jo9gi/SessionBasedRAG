import threading
from collections import OrderedDict

class LRUCache:
    """Thread‑safe LRU cache for storing per‑session FAISS stores (or any object)."""
    def __init__(self, capacity: int = 3):
        self.capacity = capacity
        self.store = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key not in self.store:
                return None
            # Move to end to mark as recently used
            self.store.move_to_end(key)
            return self.store[key]

    def put(self, key, value):
        with self.lock:
            self.store[key] = value
            self.store.move_to_end(key)
            if len(self.store) > self.capacity:
                # pop least‑recently used item
                self.store.popitem(last=False)
