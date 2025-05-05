# 1. Priority Queue(우선순위 큐)
    - 일반적인 큐: FIFO 선입선출
    - Priority Queue: 데이터에 우선순위 부여 → 우선순위 높은 것이 먼저 나감
        ● 삽입순서 무관
        ● 우선순위는 주로 정수로 표현
        ● 예시1: 응급실에서는 심각한 환자부터 치료
        ● 예시2: 운영체제 스케쥴러는 우선순위 높은 자료부터 처리


# 2. Heap
    - Priority Queue를 빠르게 구현하게 위해 만든 트리기반 자료 구조
    - Complete Binary Tree 형태를 취함
        ● 왼쪽부터 빈틈없이 채워진 이진 트리 = 모든 상위 레벨은 꽉 참 + 마지막 층만 왼쪽부터 참
        ![alt](https://cdn.programiz.com/sites/tutorial2program/files/complete-binary-tree_0.png)
        ![alt](C:\Users\user\Downloads\complete-binary-tree.png)