✅ 전체 목적
이 코드는 Min Heap(최소 힙) 자료구조를 직접 구현하고:
- 각 삽입(insert) 및 삭제(pop) 시
- **걸린 시간(ms)**을 출력하며
- 힙의 내부 구조를 배열 형태와 트리 형태로 시각화합니다.

✅ 클래스 구성 상세 설명
`__init__`: 생성자

    def __init__(self):
        self.heap = []
- 빈 리스트로 힙 초기화 (self.heap은 배열 형태로 힙을 표현함)


`insert(val)`: 힙에 값 삽입

    start = time.time()
    self.heap.append(val)
    self._bubble_up(len(self.heap) - 1)
    end = time.time()

- 값을 힙 끝에 추가 (append)
- 부모 노드와 비교하며 위로 올라가는 _bubble_up() 호출 → 힙 속성 유지
- 시간 측정 (단위: 밀리초)


        print(f"[INSERT] {val} → {((end - start) * 1000):.3f} ms")
        self.print_heap()

- 삽입 연산 시간과 현재 힙 구조를 출력

`_bubble_up(idx)`: 삽입 후 위로 올리기


    parent = (idx - 1) // 2
    if idx > 0 and self.heap[idx] < self.heap[parent]:
        self._swap(idx, parent)
        self._bubble_up(parent)

- Min Heap은 “항상 부모 노드 ≤ 자식 노드”가 되어야 합니다.
새로 삽입된 값이 이 조건을 위반하는 경우, 부모와 자리를 바꾸며 위로 올라가야 합니다 → 이 과정을 bubble-up(버블 업) 또는 heapify-up이라고 부릅니다.
- `parent = (idx - 1) // 2`
  - 트리에서 현재 노드가 idx 위치에 있을 때, 그 부모의 인덱스를 계산하는 공식
  - 부모 인덱스: parent = (child_index - 1) // 2
  - 왼쪽 자식: 2 * parent + 1
  - 오른쪽 자식: 2 * parent + 2
```python
        Index: 0   1   2   3   4   5   6
        Heap:  [5, 7, 8, 9, 10, 11, 12]
        Tree:  5
             /    \
            7      8
           / \    / \
          9  10  11  12
```
- `if idx > 0 and self.heap[idx] < self.heap[parent]`
  - idx > 0: 루트(인덱스 0)보다 위로는 못 올라갑니다. 루트면 종료
  - self.heap[idx] < self.heap[parent]: 자식이 부모보다 작으면, Min Heap 조건 위반 → 자리 바꾸기 필요.
- `self._swap(idx, parent)`
  - 현재 노드와 부모 노드를 교환
  - 즉, 더 작은 값이 위로 올라가도록 위치를 바꿈
- `self._bubble_up(parent)`
  - 한 단계 swap 하고 끝나는 게 아니라, 그 부모도 또 위에 부모보다 작을 수 있음
  - 따라서, 다시 위쪽으로 올라가며 반복 → 결국 제자리 찾을 때까지 위로 bubble
- 이 과정이 O(log n)만큼 수행됨

`pop()`: 최소값 꺼내기

    self._swap(0, len(self.heap) - 1)
    min_val = self.heap.pop()
    self._heapify(0)

- `self._swap(0, len(self.heap) - 1)`
  - 루트 노드(인덱스 0)와 마지막 노드를 서로 교환
  - 리스트의 pop() 연산은 마지막 요소만 빠르게 제거 가능 (O(1))
  - 하지만 우리가 꺼내고 싶은 건 **가장 위에 있는 루트 노드(최소값)**
  - 그래서 루트와 마지막 노드를 바꾸고 → 마지막 노드를 제거
  - heap = [1, 3, 8, 5] → swap → [5, 3, 8, 1]
- `min_val = self.heap.pop()`
  - pop()은 마지막 노드를 제거
  - 이때 실제로 제거되는 값은 min_val 변수에 저장
  - heap = [5, 3, 8, 1] → pop() → heap = [5, 3, 8]
  - min_val = 1
    - 우리가 원한 루트 노드(최소값)
- `self._heapify(0)`
  - 루트 노드가 이제 엉뚱한 값으로 바뀌었으니, "자식들과 비교하며" 아래로 내려보내면서 힙 구조를 복구
  - 이 과정을 down-heap 또는 heapify-down이라고 함
  - heap = [5, 3, 8] → heapify() → heap = [3, 5, 8]



`_heapify(idx)`: 삭제 후 아래로 내려보내기

    smallest = idx
    left = 2 * idx + 1 // 왼쪽 자식의 index
    right = 2 * idx + 2 // 오른쪽 자식의 index
    n = len(self.heap)

    if left < n and self.heap[left] < self.heap[smallest]:
        smallest = left
    if right < n and self.heap[right] < self.heap[smallest]:
        smallest = right

    if smallest != idx:
        self._swap(idx, smallest)
        self._heapify(smallest)

- `smallest = idx`
  - 현재 노드를 가장 작은 값의 위치라고 가정하고 시작
- 왼쪽/오른쪽 자식과 비교
  - 왼쪽 자식이 있고, 현재 노드보다 더 작으면 smallest를 왼쪽으로 변경
  - 오른쪽 자식도 더 작다면, 그쪽으로 다시 smallest 갱신
- `if smallest != idx:`
  - 결국 smallest가 원래의 idx와 같으면 → 부모가 자식보다 작거나 같음 → OK!
  - 하지만 다르면 → 부모가 더 큼 → Min Heap 조건 위반
→ swap 후 다시 heapify 필요

```python
        heap = [5, 3, 8]
                ↑
                idx=0
        left = 1 → heap[1] = 3  
        right = 2 → heap[2] = 8

        heap[1] < heap[0] → smallest = 1
```

- 자식 중 더 작은 쪽과 비교해서 자식보다 크면 swap
- 재귀적으로 반복하여 힙 구조 유지 (역시 O(log n))

`_swap(i, j)`: 값 바꾸기

    self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
- 두 인덱스의 값을 교환