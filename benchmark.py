import asyncio
import time
import statistics
import psutil
from collections import defaultdict
import httpx

BASE_URL = "http://127.0.0.1:8000"

TEST_TEXTS = [
    "Привет",
    "Это короткий текст для тестирования.",
    "Этот текст средней длины содержит несколько предложений для проверки работы модели.",
    "Данный текст имеет значительную длину и содержит множество различных слов и предложений, чтобы проверить как модель обрабатывает более длинные входные данные. Это важно для понимания производительности при работе с реальными пользовательскими запросами.",
    "Краткое описание: сервис предоставляет возможность получения векторных представлений текста на русском языке с использованием модели rubert-mini-frida. Модель поддерживает различные префиксы для оптимизации под конкретные задачи: поиск по запросам, поиск по документам, парафразирование, классификация и другие.",
    "Machine Learning (ML) and Natural Language Processing (NLP) are rapidly evolving fields that are transforming how we interact with technology. Deep learning models, particularly transformer-based architectures like BERT, have revolutionized text understanding and generation. The rubert-mini-frida model is an example of a compact yet effective model designed for Russian language processing, offering a good balance between performance and computational efficiency.",
    "В современном мире искусственный интеллект и машинное обучение играют ключевую роль в развитии технологий. Компании по всему миру инвестируют миллиарды долларов в разработку новых моделей и алгоритмов. Особое внимание уделяется обработке естественного языка (NLP), где модели типа BERT и GPT показывают впечатляющие результаты. Российские разработчики также вносят свой вклад в эту область, создавая специализированные модели для работы с русским языком.",
    "a" * 500,
    "word " * 150,
]

PREFIXES = [
    "search_query",
    "search_document",
    "paraphrase",
    "categorize",
    "categorize_sentiment",
    "categorize_topic",
    "categorize_entailment",
]


async def wait_for_health():
    print("Ожидание готовности сервиса...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        # ждём не более двух минут
        for _ in range(120):
            try:
                resp = await client.get(f"{BASE_URL}/health")
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "healthy":
                        print(
                            f"Сервис готов! embedding_dim={data.get('embedding_dim')}"
                        )
                        return True
            except:
                pass
            await asyncio.sleep(1)
    return False


async def send_request(client, text_idx, prefix_idx):
    text = TEST_TEXTS[text_idx % len(TEST_TEXTS)]
    prefix = PREFIXES[prefix_idx % len(PREFIXES)]

    start = time.perf_counter()
    try:
        resp = await client.post(
            f"{BASE_URL}/embed", json={"text": text, "prefix": prefix}, timeout=30.0
        )
        latency = (time.perf_counter() - start) * 1000
        return {
            "latency": latency,
            "status": resp.status_code,
            "success": resp.status_code == 200,
            "text_len": len(text),
        }
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return {
            "latency": latency,
            "status": 0,
            "success": False,
            "error": str(e),
            "text_len": len(text),
        }


async def benchmark_concurrent(duration_seconds=30, concurrent_requests=10):
    print(f"\n{'=' * 60}")
    print(f"БЕНЧМАРК: {duration_seconds}s, {concurrent_requests} параллельных запросов")
    print(f"{'=' * 60}")

    process = psutil.Process()
    results = []
    active_requests_history = []

    cpu_samples = []
    memory_samples = []

    start_time = time.time()
    last_report = start_time

    async with httpx.AsyncClient(timeout=30.0) as client:
        while time.time() - start_time < duration_seconds:
            batch_start = time.time()
            batch_tasks = []

            for i in range(concurrent_requests):
                text_idx = len(results) + i
                prefix_idx = len(results) + i
                batch_tasks.append(send_request(client, text_idx, prefix_idx))

            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)

            active_requests_history.append(concurrent_requests)

            cpu_samples.append(process.cpu_percent(interval=None))
            memory_samples.append(process.memory_info().rss / 1024 / 1024)

            current_time = time.time()
            if current_time - last_report >= 5:
                elapsed = current_time - start_time
                total_rps = len(results) / elapsed
                success_rate = (
                    sum(1 for r in results if r["success"]) / len(results) * 100
                )
                avg_latency = (
                    statistics.mean(r["latency"] for r in results) if results else 0
                )
                print(
                    f"  [{elapsed:.0f}s] RPS: {total_rps:.1f}, Success: {success_rate:.1f}%, Avg latency: {avg_latency:.0f}ms"
                )
                last_report = current_time

            batch_duration = time.time() - batch_start
            if batch_duration < 0.05:
                await asyncio.sleep(0.05 - batch_duration)

    return results, cpu_samples, memory_samples, active_requests_history


def analyze_results(results, cpu_samples, memory_samples, duration):
    print(f"\n{'=' * 60}")
    print("РЕЗУЛЬТАТЫ БЕНЧМАРКА")
    print(f"{'=' * 60}")

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    latencies = sorted([r["latency"] for r in successful])

    def percentile(data, p):
        if not data:
            return 0
        idx = int(len(data) * p / 100)
        idx = min(idx, len(data) - 1)
        return data[idx]

    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)
    p99 = percentile(latencies, 99)

    total_requests = len(results)
    total_duration = duration
    rps = total_requests / total_duration

    avg_cpu = statistics.mean(cpu_samples) if cpu_samples else 0
    max_cpu = max(cpu_samples) if cpu_samples else 0
    avg_memory = statistics.mean(memory_samples) if memory_samples else 0
    max_memory = max(memory_samples) if memory_samples else 0

    model_inference_time = p50

    print(f"\nLatency (мс):")
    print(f"   P50:  {p50:>8.1f} ms  {'✅' if p50 < 200 else '❌'}")
    print(f"   P95:  {p95:>8.1f} ms  {'✅' if p95 < 500 else '❌'}")
    print(f"   P99:  {p99:>8.1f} ms  {'✅' if p99 < 2000 else '❌'}")
    print(f"   Avg:  {statistics.mean(latencies) if latencies else 0:>8.1f} ms")
    print(f"   Min:  {min(latencies) if latencies else 0:>8.1f} ms")
    print(f"   Max:  {max(latencies) if latencies else 0:>8.1f} ms")

    print(f"\nThroughput:")
    print(f"   RPS:  {rps:>8.1f} req/s")
    print(f"   Total requests: {total_requests}")

    print(f"\nModel Inference Time (P50):")
    print(
        f"   ~{model_inference_time:>6.1f} ms  {'✅' if model_inference_time < 50 else '❌'}"
    )

    print(f"\nCPU Usage:")
    print(f"   Average: {avg_cpu:>6.1f}%  {'✅' if avg_cpu < 80 else '❌'}")
    print(f"   Max:     {max_cpu:>6.1f}%")

    print(f"\nMemory Usage:")
    print(f"   Average: {avg_memory:>6.1f} MB")
    print(f"   Max:     {max_memory:>6.1f} MB")

    print(f"\n{'─' * 60}")
    print("Status:")
    status_p50 = "PASS" if p50 < 200 else "FAIL"
    status_p95 = "PASS" if p95 < 500 else "FAIL"
    status_p99 = "PASS" if p99 < 2000 else "FAIL"
    status_cpu = "PASS" if avg_cpu < 80 else "FAIL"
    status_rps = "GOOD" if rps >= 20 else "LOW"

    print(f"   P50 latency:  {status_p50} (< 200ms)")
    print(f"   P95 latency:  {status_p95} (< 500ms)")
    print(f"   P99 latency:  {status_p99} (< 2000ms)")
    print(f"   CPU usage:    {status_cpu} (< 80%)")
    print(f"   RPS:          {status_rps} (target: >= 20)")

    if failed:
        print(f"\nFailed requests: {len(failed)}/{total_requests}")
        for f in failed[:3]:
            print(f"   - {f.get('error', 'Unknown error')}")

    return {
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "avg_latency": statistics.mean(latencies) if latencies else 0,
        "min_latency": min(latencies) if latencies else 0,
        "max_latency": max(latencies) if latencies else 0,
        "rps": rps,
        "total_requests": total_requests,
        "avg_cpu": avg_cpu,
        "max_cpu": max_cpu,
        "avg_memory": avg_memory,
        "max_memory": max_memory,
        "model_inference_time": model_inference_time,
        "success_rate": len(successful) / total_requests * 100 if total_requests else 0,
    }


async def run_benchmarks():
    if not await wait_for_health():
        print("Сервис не готов. Запустите: fastapi dev")
        return

    print("\n" + "=" * 60)
    print("ТЕСТ 1: Низкая нагрузка (5 параллельных, 15 секунд)")
    print("=" * 60)
    results1, cpu1, mem1, _ = await benchmark_concurrent(
        duration_seconds=15, concurrent_requests=5
    )
    stats1 = analyze_results(results1, cpu1, mem1, 15)

    await asyncio.sleep(2)

    print("\n" + "=" * 60)
    print("ТЕСТ 2: Средняя нагрузка (10 параллельных, 30 секунд)")
    print("=" * 60)
    results2, cpu2, mem2, _ = await benchmark_concurrent(
        duration_seconds=30, concurrent_requests=10
    )
    stats2 = analyze_results(results2, cpu2, mem2, 30)

    await asyncio.sleep(2)

    print("\n" + "=" * 60)
    print("ТЕСТ 3: Высокая нагрузка (20 параллельных, 30 секунд)")
    print("=" * 60)
    results3, cpu3, mem3, _ = await benchmark_concurrent(
        duration_seconds=30, concurrent_requests=20
    )
    stats3 = analyze_results(results3, cpu3, mem3, 30)

    print("\n" + "=" * 60)
    print("СВОДНАЯ ТАБЛИЦА")
    print("=" * 60)
    print(f"{'Test':<25} {'P50':<10} {'P95':<10} {'P99':<10} {'RPS':<10} {'CPU%':<10}")
    print("─" * 75)
    print(
        f"{'Low load (5 parallel)':<25} {stats1['p50']:<10.1f} {stats1['p95']:<10.1f} {stats1['p99']:<10.1f} {stats1['rps']:<10.1f} {stats1['avg_cpu']:<10.1f}"
    )
    print(
        f"{'Medium load (10 parallel)':<25} {stats2['p50']:<10.1f} {stats2['p95']:<10.1f} {stats2['p99']:<10.1f} {stats2['rps']:<10.1f} {stats2['avg_cpu']:<10.1f}"
    )
    print(
        f"{'High load (20 parallel)':<25} {stats3['p50']:<10.1f} {stats3['p95']:<10.1f} {stats3['p99']:<10.1f} {stats3['rps']:<10.1f} {stats3['avg_cpu']:<10.1f}"
    )

    print("\n" + "=" * 60)
    print("ЦЕЛЕВЫЕ ПОРОГИ (из README.md)")
    print("=" * 60)
    print(f"{'Metric':<30} {'Result':<15} {'Target':<15} {'Status':<10}")
    print("─" * 70)
    p50_ok = "✅" if stats2["p50"] < 200 else "❌"
    p95_ok = "✅" if stats2["p95"] < 500 else "❌"
    p99_ok = "✅" if stats2["p99"] < 2000 else "❌"
    rps_ok = "✅" if stats2["rps"] >= 20 else "❌"
    inf_ok = "✅" if stats2["model_inference_time"] < 50 else "❌"
    cpu_ok = "✅" if stats2["avg_cpu"] < 80 else "❌"
    print(
        f"{'P50 Latency':<30} {stats2['p50']:.1f} ms{'':>10} {'< 200ms':<15} {p50_ok}"
    )
    print(
        f"{'P95 Latency':<30} {stats2['p95']:.1f} ms{'':>10} {'< 500ms':<15} {p95_ok}"
    )
    print(
        f"{'P99 Latency':<30} {stats2['p99']:.1f} ms{'':>10} {'< 2000ms':<15} {p99_ok}"
    )
    print(f"{'RPS':<30} {stats2['rps']:.1f} req/s{'':>6} {'>= 20':<15} {rps_ok}")
    print(
        f"{'Model Inference Time':<30} {stats2['model_inference_time']:.1f} ms{'':>10} {'< 50ms':<15} {inf_ok}"
    )
    print(f"{'CPU Usage':<30} {stats2['avg_cpu']:.1f}%{'':>10} {'< 80%':<15} {cpu_ok}")

    return stats2


if __name__ == "__main__":
    asyncio.run(run_benchmarks())
