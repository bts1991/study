import threading
from multiprocessing import Process
import time

def light_task(n):
    print(f"[작업-{n}] 시작")
    time.sleep(1)
    print(f"[작업-{n}] 완료")

def run_multithreading():
    print("\n--- 멀티스레딩 실행 ---")
    # 멀티스레딩
    start = time.time()
    threads = []
    for i in range(5):
        t = threading.Thread(target=light_task, args=(i,))
        # args=(i,)는 파이썬에서 함수에 전달할 인자들을 튜플 형태로 넘기는 문법
        # 이 경우 light_task(i)가 호출됨
        # 왜 (i,)처럼 쉼표(,) 필요? 튜플을 만들기 위한 최소 문법, (i)는 튜플이 아님
        # 
        threads.append(t) # 스레드나 프로세스 객체를 리스트에 저장하는 행위인데,
                          # 이걸 하는 실질적인 이유는 .join() 등의 제어를 나중에 하기 위해
        t.start()

    for t in threads:
        t.join() # 해당 스레드(또는 프로세스)가 종료될 때까지 기다림
    end = time.time()
    print(f"멀티스레딩 총 실행 시간: {end - start:.2f}초")
    
def run_multiprocessing():
    print("\n--- 멀티프로세싱 실행 ---")
    # 멀티프로세싱
    start = time.time()
    processes = []
    for i in range(5):
        p = Process(target=light_task, args=(i,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    end = time.time()
    print(f"멀티프로세싱 총 실행 시간: {end - start:.2f}초")



if __name__ == "__main__":
    run_multithreading()
    run_multiprocessing()
