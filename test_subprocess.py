import asyncio
import time
import numpy as np
import paddle
import logging
from multiprocessing import Process, Queue
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, TypedDict
import atexit
import threading

# ... (省略之前的 Mock 和基础类定义，保持不变) ...

class PutTask(TypedDict):
    type: str
    key: str
    data: np.ndarray

class RoutingStoreRDMA(RoutingStoreBase):
    """
    Producer-Consumer RDMA Store with NON-BLOCKING producer.
    Goal: Main process never waits for IO.
    """

    def __init__(self, fd_config: FDConfig, max_workers: int = 4, queue_max_size: int = 10000) -> None:
        super().__init__(fd_config=fd_config)
        try:
            from p2pstore import P2PClient, P2PConfig
        except ModuleNotFoundError:
            raise ModuleNotFoundError("RoutingStoreRDMA and p2pstore only supported in RLHF environment.")

        self.max_workers = max_workers
        self.queue_max_size = queue_max_size
        
        # 使用更大的队列减少丢弃概率
        self._task_queue: Queue = Queue(maxsize=self.queue_max_size)
        
        self._consumer_process: Process = None
        self._monitor_thread: threading.Thread = None
        self._stop_monitor = threading.Event()

        self.p2p_config = P2PConfig(metadata_server=fd_config.routing_replay_config.rdma_store_server)
        self.p2p_client = None # 将在子进程中初始化

        self._is_running = False
        self._dropped_tasks = 0
        
        atexit.register(self.shutdown)

    # --- 消费者侧逻辑 (子进程) ---
    
    def _consumer_worker(self, task: PutTask):
        """工作线程执行实际的 put"""
        key = task['key']
        data = task['data']
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.p2p_client.put(key, data))
        except Exception as e:
            logger.error(f"Worker failed for key {key}: {e}")
        finally:
            loop.close()

    def _consumer_process_main(self, task_queue: Queue, p2p_config: P2PConfig):
        """消费者进程主循环"""
        print(f"[Consumer Process {Process.current_process().pid}] Started with {self.max_workers} workers.")
        self.p2p_client = P2PClient(p2p_config)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while True:
                try:
                    # 阻塞等待任务，这是消费者进程该做的事
                    task = task_queue.get()
                    if task is None: # Sentinel
                        break
                    
                    # 提交给线程池异步执行
                    executor.submit(self._consumer_worker, task)
                    
                except Exception as e:
                    logger.error(f"Consumer loop error: {e}")
                    break
        
        print(f"[Consumer Process {Process.current_process().pid}] Shutdown.")

    # --- 生产者侧逻辑 (主进程) ---

    def _monitor_queue_load(self):
        """后台监控线程：仅用于观察，绝不阻塞主逻辑"""
        while not self._stop_monitor.is_set():
            time.sleep(2.0)
            qsize = self._task_queue.qsize()
            # 如果队列长度超过 80%，说明消费者跟不上了，需要告警
            if qsize > self.queue_max_size * 0.8:
                logger.warning(
                    f"[Monitor] Queue load is HIGH: {qsize}/{self.queue_max_size}. "
                    f"Dropped tasks so far: {self._dropped_tasks}. "
                    "Consider increasing max_workers or queue_max_size."
                )
            else:
                logger.info(f"[Monitor] Queue load: {qsize}/{self.queue_max_size}. Healthy.")

    def start(self):
        """启动消费者进程和监控线程"""
        if self._is_running:
            return

        self._is_running = True
        self._consumer_process = Process(
            target=self._consumer_process_main,
            args=(self._task_queue, self.p2p_config),
            daemon=True
        )
        self._consumer_process.start()

        # 启动监控线程（守护线程）
        self._stop_monitor.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_queue_load, daemon=True)
        self._monitor_thread.start()
        
        logger.info(f"RoutingStoreRDMA started. Consumer PID: {self._consumer_process.pid}")

    async def put(self, routing_indices: paddle.Tensor, rollout_id: str, layer_idx: int) -> None:
        """
        【非阻塞】生产者接口：极速入队，立即返回。
        如果队列满了，直接丢弃并计数（也可以选择抛异常或其他策略）。
        """
        if not self._is_running:
            raise RuntimeError("Store not started.")

        rdma_rollout_key = f"{rollout_id}_{layer_idx}"
        
        # 数据准备（这部分在主进程做，因为需要访问 Tensor）
        routing_indices_cpu = routing_indices.cpu()
        routing_indices_np = np.array(routing_indices_cpu.numpy(), copy=True)
        
        task: PutTask = {
            "type": "put",
            "key": rdma_rollout_key,
            "data": routing_indices_np
        }

        try:
            # 核心：put_nowait 绝对不阻塞
            self._task_queue.put_nowait(task)
        except Exception: 
            # 队列满了
            self._dropped_tasks += 1
            logger.warning(
                f"Queue is FULL. Dropping put task for key: {rdma_rollout_key}. "
                f"Total dropped: {self._dropped_tasks}"
            )
            # 这里不抛异常，不阻塞，仅仅记录日志

    async def fused_put(self, routing_indices: paddle.Tensor, rollout_id: str) -> None:
        """【非阻塞】生产者接口：极速入队"""
        if not self._is_running:
            raise RuntimeError("Store not started.")

        rdma_rollout_key = f"{rollout_id}"
        routing_indices_cpu = routing_indices.cpu()
        routing_indices_np = routing_indices_cpu.numpy()

        task: PutTask = {
            "type": "fused_put",
            "key": rdma_rollout_key,
            "data": routing_indices_np
        }

        try:
            self._task_queue.put_nowait(task)
        except Exception:
            self._dropped_tasks += 1
            logger.warning(
                f"Queue is FULL. Dropping fused_put task for key: {rdma_rollout_key}. "
                f"Total dropped: {self._dropped_tasks}"
            )

    # --- 同步/管理接口 ---

    def wait_completion(self, timeout: float = 30.0):
        """
        【可选同步】等待所有队列中的任务被处理完。
        仅在程序退出前调用，平时不要调用。
        """
        if not self._is_running:
            return

        logger.info("Waiting for consumer to finish remaining tasks...")
        start = time.time()
        
        # 1. 发送停止信号给消费者进程
        self._task_queue.put(None)
        
        # 2. 等待消费者进程结束
        self._consumer_process.join(timeout=timeout)
        
        if self._consumer_process.is_alive():
            logger.error("Consumer did not finish in time. Terminating.")
            self._consumer_process.terminate()
        
        # 3. 停止监控
        self._stop_monitor.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

        logger.info(f"Wait completed in {time.time() - start:.2f}s. Total dropped tasks: {self._dropped_tasks}")

    def shutdown(self):
        """优雅关闭"""
        if not self._is_running:
            return
        
        logger.info("Shutting down...")
        self._is_running = False
        
        # 确保队列里有东西让消费者醒来（如果之前空了）
        try:
            self._task_queue.put_nowait(None)
        except:
            pass

        if self._consumer_process and self._consumer_process.is_alive():
            self._consumer_process.join(timeout=5.0)
            if self._consumer_process.is_alive():
                self._consumer_process.terminate()
        
        self._is_running = False
        logger.info("Shutdown complete.")

    # ... (get, clear 等同步方法保持不变，它们直接创建临时 client) ...
    def get(self, rollout_id: str, layer_idx: int = None) -> paddle.Tensor:
        rdma_rollout_key = f"{rollout_id}_{layer_idx}" if layer_idx is not None else rollout_id
        # 临时创建 client 用于同步读
        tmp_client = P2PClient(self.p2p_config)
        tmp_routing = asyncio.run(tmp_client.get(rdma_rollout_key))
        return paddle.to_tensor(tmp_routing)

    def clear(self, rollout_id: str, layer_idx: int = None) -> None:
        rdma_rollout_key = f"{rollout_id}_{layer_idx}" if layer_idx is not None else rollout_id
        tmp_client = P2PClient(self.p2p_config)
        asyncio.run(tmp_client.delete(rdma_rollout_key))

    async def clear_prefix_batch(self, roullout_id_prefixes: List[str]):
        tmp_client = P2PClient(self.p2p_config)
        await tmp_client.delete_prefix_batch(roullout_id_prefixes)

    def clear_store(self):
        tmp_client = P2PClient(self.p2p_config)
        asyncio.run(tmp_client.clear())