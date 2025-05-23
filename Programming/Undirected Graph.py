class UndirectedGraphList: ## 인접리스트
    def __init__(self):
        self.graph = {}

    def add_vertex(self, v):
        if v not in self.graph:
            self.graph[v] = []

    def add_edge(self, v1, v2):
        self.add_vertex(v1)
        self.add_vertex(v2)
        self.graph[v1].append(v2)
        self.graph[v2].append(v1)  # 무방향: 양방향 연결

    def display(self):
        for v in self.graph:
            print(f"{v} -> {self.graph[v]}")
            

class DirectedGraphList:
    def __init__(self):
        self.graph = {}

    def add_vertex(self, v):
        if v not in self.graph:
            self.graph[v] = []

    def add_edge(self, from_v, to_v):
        self.add_vertex(from_v)
        self.add_vertex(to_v)
        self.graph[from_v].append(to_v)  # 방향: from_v → to_v

    def display(self):
        for v in self.graph:
            print(f"{v} → {self.graph[v]}")

class UndirectedGraphMatrix: ## 인접행렬
    def __init__(self, vertices):
        self.vertices = vertices
        self.size = len(vertices)
        self.matrix = [[0] * self.size for _ in range(self.size)]

    def add_edge(self, v1, v2):
        i = self.vertices.index(v1)
        j = self.vertices.index(v2)
        self.matrix[i][j] = 1
        self.matrix[j][i] = 1  # 무방향: 양쪽 설정

    def display(self):
        print("   " + "  ".join(self.vertices))
        for i, row in enumerate(self.matrix):
            print(f"{self.vertices[i]}  {row}")

class DirectedGraphMatrix:
    def __init__(self, vertices):
        self.vertices = vertices
        self.size = len(vertices)
        self.matrix = [[0] * self.size for _ in range(self.size)]

    def add_edge(self, from_v, to_v):
        i = self.vertices.index(from_v)
        j = self.vertices.index(to_v)
        self.matrix[i][j] = 1  # 방향만 저장

    def display(self):
        print("   " + "  ".join(self.vertices))
        for i, row in enumerate(self.matrix):
            print(f"{self.vertices[i]}  {row}")


# 예시 실행
print("\n무방향 그래프 (인접 리스트 방식):")
g_list = UndirectedGraphList()
g_list.add_edge('A', 'B')
g_list.add_edge('A', 'C')
g_list.add_edge('B', 'D')
g_list.add_edge('C', 'D')
g_list.add_edge('D', 'E')
g_list.display()

# 예시 실행
print("유방향 그래프 (인접 리스트 방식):")
g_list = DirectedGraphList()
g_list.add_edge('A', 'B')
g_list.add_edge('A', 'C')
g_list.add_edge('B', 'D')
g_list.add_edge('C', 'D')
g_list.add_edge('D', 'E')
g_list.display()

# 예시 실행
print("\n무방향 그래프 (인접 행렬 방식):")
vertices = ['A', 'B', 'C', 'D', 'E']
g_matrix = UndirectedGraphMatrix(vertices)
g_matrix.add_edge('A', 'B')
g_matrix.add_edge('A', 'C')
g_matrix.add_edge('B', 'D')
g_matrix.add_edge('C', 'D')
g_matrix.add_edge('D', 'E')
g_matrix.display()


# 예시 실행
print("\n유방향 그래프 (인접 행렬 방식):")
vertices = ['A', 'B', 'C', 'D', 'E']
g_matrix = DirectedGraphMatrix(vertices)
g_matrix.add_edge('A', 'B')
g_matrix.add_edge('A', 'C')
g_matrix.add_edge('B', 'D')
g_matrix.add_edge('C', 'D')
g_matrix.add_edge('D', 'E')
g_matrix.display()





