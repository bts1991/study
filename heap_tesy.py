import time
import math

class MinHeap:
    def __init__(self):
        self.heap = []

    def insert(self, val):
        start = time.time()
        self.heap.append(val)
        self._bubble_up(len(self.heap) - 1)
        end = time.time()
        print(f"[INSERT] {val} → {((end - start) * 1000):.3f} ms")
        self.print_heap()

    def pop(self):
        if not self.heap:
            return None
        start = time.time()
        self._swap(0, len(self.heap) - 1)
        min_val = self.heap.pop()
        self._heapify(0)
        end = time.time()
        print(f"[POP] {min_val} → {((end - start) * 1000):.3f} ms")
        self.print_heap()
        return min_val

    def _bubble_up(self, idx):
        parent = (idx - 1) // 2
        if idx > 0 and self.heap[idx] < self.heap[parent]:
            self._swap(idx, parent)
            self._bubble_up(parent)

    def _heapify(self, idx):
        smallest = idx
        left = 2 * idx + 1
        right = 2 * idx + 2
        n = len(self.heap)

        if left < n and self.heap[left] < self.heap[smallest]:
            smallest = left
        if right < n and self.heap[right] < self.heap[smallest]:
            smallest = right

        if smallest != idx:
            self._swap(idx, smallest)
            self._heapify(smallest)

    def _swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def print_heap(self):
        print("Current Heap (array):", self.heap)
        self.print_tree()
        print("-" * 40)

    def print_tree(self):
        n = len(self.heap)
        if n == 0:
            print("[empty heap]")
            return

        level = 0
        i = 0
        while i < n:
            count = 2**level
            level_vals = self.heap[i:i+count]
            space = " " * (2 ** (max(0, math.ceil(math.log2(n + 1))) - level))
            print(space.join(str(v) for v in level_vals))
            i += count
            level += 1


# 테스트 실행
heap = MinHeap()
for num in [5, 3, 8, 1, 7, 2]:
    heap.insert(num)

heap.pop()
heap.pop()
