import threading
from multiprocessing import Process
import time

# 무거운 계산 작업 (CPU-bound)
def heavy_task(n):
    print(f"[작업-{n}] 시작")
    count = 0
    for i in range(10**7):  # 계산량 많음
        count += i
    print(f"[작업-{n}] 완료")


def run_multithreading():
    print("\n--- 멀티스레딩 실행 ---")
    start = time.time()
    threads = []
    for i in range(5):
        t = threading.Thread(target=heavy_task, args=(i,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    end = time.time()
    print(f"멀티스레딩 총 실행 시간: {end - start:.2f}초")


def run_multiprocessing():
    print("\n--- 멀티프로세싱 실행 ---")
    start = time.time()
    processes = []
    for i in range(5):
        p = Process(target=heavy_task, args=(i,))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    end = time.time()
    print(f"멀티프로세싱 총 실행 시간: {end - start:.2f}초")


if __name__ == "__main__":
    run_multithreading()
    run_multiprocessing()
