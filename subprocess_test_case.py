import numpy
from typing import List, Dict, Any, TypedDict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Queue
import asyncio
import time
import numpy as np
import paddle
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, TypedDict
import atexit
import threading
import os

class RoutingManager(object):
    def __init__(self) -> None:

        # Initialize routing store
        self._routing_store = RoutingStoreLocal()

        # Initialize routing store wrapper
        self._routing_store_process = StoreWrapper(
            routing_store=self._routing_store
        )

class StoreTask(TypedDict):
    task_type: str
    key: str
    data: np.ndarray

class StoreProcess(Process):
    def __init__(self, task_queue: Queue, routing_store: object) -> None:
        self._task_quequ = task_queue
        self._routing_store = routing_store

    def run(self):
        print(f"[R3] Start Running Store Wrapper in sub process {os.getpid()}")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while True:
                try:
                    
                    task = self._task_quequ.get()
                    
                    if task is None: # Sentinel
                        self._task_quequ.task_done()
                        break
                    
                    if task['task_type'] == 'put':
                        executor.submit(self.process_put_task, task)
                    elif task['task_type'] == 'clear_store':
                        executor.submit(self.process_clear_store_task, task)
                        self._task_quequ.task_done()
                    elif task['task_type'] == 'clear_prefix_batch':
                        executor.submit(self.process_clear_prefix_batch_task, task)
                    else:
                        raise ValueError(f'Unknown task type: {task["task_type"]}')

                except Exception as e:
                    self._task_quequ.task_done()
                    raise ValueError(f'{e}')
        
        print(f"[Consumer Process {Process.current_process().pid}] Shutdown.")

    def process_put_task(self, store_task: StoreTask) -> None:
        """ """
        self._routing_store.put(store_task.key, store_task.data)

    def process_clear_store_task(self, store_task: StoreTask) -> None:
        """ """
        self._routing_store.clear()

    def process_clear_prefix_batch_task(self, store_task: StoreTask) -> None:
        """ """
        self._routing_store.delete_prefix_batch(store_task.key)

class StoreWrapper(object):
    def __init__(self) -> None:
        # Initialize task queue
        layer_num = 61
        max_request = 200
        self.queue_max_size = layer_num * max_request
        self._task_queue = Queue(maxsize=self.queue_max_size)
        self._monitor_thread: threading.Thread = None
        self._stop_monitor = threading.Event()

        # Initialize consumer process
        self._routing_store_process = StoreProcess(
            task_queue=self._task_queue,
            routing_store=self._routing_store
        )
        self._is_running = False

        # Register atexit handler
        atexit.register(self.shutdown)
    
    def shutdown(self):
        """ """
        if not self._is_running:
            return
        print.info("Shutting down...")
        self._is_running = False

        # Put a sentinel value to signal the consumer to stop
        try:
            self._task_queue.put_nowait(None)
        except:
            pass
        if self._consumer_process and self._consumer_process.is_alive():
            # Wait for all tasks to be processed
            self._consumer_process.join(timeout=5.0)
            if self._consumer_process.is_alive():
                self._consumer_process.terminate()
        self._is_running = False

    def start_store_warpper(self):
        """ """
        if self._wrapper_is_running:
            return
        self._is_running = True

        # Start monitor thread
        self._stop_monitor.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_queue_load, daemon=True)
        self._monitor_thread.start()

        # Start Routing Store Wrapper in sub process
        self._routing_store_process.run()

    def _monitor_queue_load(self):
        """ """
        while not self._stop_monitor.is_set():
            time.sleep(2.0)
            qsize = self._task_queue.qsize()
            
            # Alarm when the task exceeds 80% of the queue capacity
            if qsize > self.queue_max_size * 0.8:
                print(
                    f"[Monitor] Queue load is HIGH: {qsize}/{self.queue_max_size}. "
                    f"Dropped tasks so far: {self._dropped_tasks}. "
                    "Consider increasing max_workers or queue_max_size."
                )
            else:
                print(f"[Monitor] Queue load: {qsize}/{self.queue_max_size}. Healthy.")

    def submit_put_task(self, routing_indices: paddle.Tensor, rollout_id: str, layer_idx: int) -> None:
        """ Submit a put task to the task queue"""
        if not self._is_running:
            raise RuntimeError("Store not started.")

        rdma_rollout_key = f"{rollout_id}_{layer_idx}"
        routing_indices_cpu = routing_indices.cpu()
        routing_indices_np = np.array(routing_indices_cpu.numpy(), copy=True)
        
        task: StoreTask = {
            "type": "put",
            "key": rdma_rollout_key,
            "data": routing_indices_np
        }

        try:
            self._task_queue.put_nowait(task)
        except Exception: 
            raise RuntimeError(
                f"Queue is FULL. Dropping put task for key: {rdma_rollout_key}. "
            )
    
    def submit_clear_task(self) -> None:
        """ Submit clear store task """
        if not self._is_running:
            raise RuntimeError("Store not started.")
        
        task: StoreTask = {
            "type": "clear_store",
            "key": None,
            "data": None
        }

        try:
            self._task_queue.put_nowait(task)
            # Wait for the task to be processed
            self._task_queue.join()
        except Exception:
            raise RuntimeError(
                f"Queue is FULL. Dropping put task for key: clear_store. "
            )

    def submit_clear_prefix_batch_task(self, rollout_id) -> None:
        """ Submit clear prefix batch task"""
        if not self._is_running:
            raise RuntimeError("Store not started.")
        prefix_batch = self.get_needed_clear_ids(rollout_id)

        if prefix_batch is None:
            return

        task :StoreTask = {
            "type": "clear_prefix_batch",
            "key": prefix_batch,
            "data": None
        }
        try:
            self._task_queue.put_nowait(task)
        except Exception:
            raise RuntimeError(
                f"Queue is FULL. Dropping put task for key: clear_store. ")

