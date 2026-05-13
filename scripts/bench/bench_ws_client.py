"""WS client benchmark — measures localhost round-trip latency floor."""
import asyncio
import statistics
import time
import websockets


async def main():
    N = 10000
    async with websockets.connect('ws://127.0.0.1:9100', ping_interval=None) as ws:
        await ws.send('warmup')
        await ws.recv()
        t_all_start = time.perf_counter()
        latencies = []
        for i in range(N):
            t0 = time.perf_counter()
            await ws.send(f'msg-{i:08d}')
            await ws.recv()
            latencies.append((time.perf_counter() - t0) * 1000)
        t_all = time.perf_counter() - t_all_start
    latencies.sort()
    print(f'WS echo round-trip ({N} msgs):')
    print(f'  mean   = {statistics.mean(latencies):.4f}ms')
    print(f'  median = {latencies[N // 2]:.4f}ms')
    print(f'  p95    = {latencies[int(N * 0.95)]:.4f}ms')
    print(f'  p99    = {latencies[int(N * 0.99)]:.4f}ms')
    print(f'  max    = {latencies[-1]:.4f}ms')
    print(f'  total  = {t_all * 1000:.1f}ms => {N / t_all:.0f} req/sec')


if __name__ == '__main__':
    asyncio.run(main())
