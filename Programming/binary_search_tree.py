class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, key):
        self.root = self._insert(self.root, key)
        
    def _insert(self, node, key):
        if node is None:
            return Node(key)
        if key < node.key:
            node.left = self._insert(node.left, key)
        elif key > node.key:
            node.right = self._insert(node.right, key)    
        
        return node

    def search(self, key):
        return self._search(self.root, key)

    def _search(self, node, key):
        if node is None or node.key == key:
            return node
        if key < node.key:
            return self._search(node.left, key)
        else:
            return self._search(node.right, key)

    def delete(self, key):
        self.root = self._delete(self.root, key)

    def _delete(self, node, key):
        if node is None:
            return node
        print("노드: ", node)
        print("노드의 키: ", node.key)
        print("키: ", key)
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            successor = self._min_value_node(node.right)
            print("후계자: ", successor.key)
            node.key = successor.key
            node.right = self._delete(node.right, successor.key)
        return node

    def _min_value_node(self, node):
        current = node
        print("----------------------최소값 찾기 시작----------------------")
        print("현재?: ", current.key)
        while current.left:
            current = current.left
        print("최소 노드?: ", current.key)
        print("----------------------최소값 찾기 종료----------------------")
        return current

    def inorder(self):
        def _inorder(node):
            if node:
                _inorder(node.left)
                print(node.key, end=" ")
                _inorder(node.right)
        _inorder(self.root)
        print()

    # 트리 시각적 출력
    def print_tree(self):
        def _print(node, prefix="", is_left=True):
            if node is not None:
                _print(node.right, prefix + ("│   " if is_left else "    "), False)
                print(prefix + ("└── " if is_left else "┌── ") + str(node.key))
                _print(node.left, prefix + ("    " if is_left else "│   "), True)
        _print(self.root)


tree = BST()
for key in [55, 29, 18, 40, 20, 60, 10, 30, 50, 70, 5, 35, 54, 80]:
# for key in [50, 40, 30, 20, 10, 9, 8, 7]:
    tree.insert(key)

print("생성 후 트리 구조:")
tree.print_tree()

print("----------------------삭제 시작----------------------")
tree.delete(40)
print("----------------------삭제 종료----------------------")

print("삭제 후 트리 구조:")
tree.print_tree()