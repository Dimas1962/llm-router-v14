#!/usr/bin/env python3
"""
Benchmark for Unified LLM Router v2.0
Tests routing performance across different strategies
"""

import os
os.environ['TQDM_DISABLE'] = '1'  # Disable progress bars

from src.unified.unified_router import UnifiedRouter, UnifiedRequest, RoutingStrategy
import asyncio
import time
import argparse
import random


async def benchmark(num_queries=5, concurrent=1, verbose=True):
    if verbose:
        print('=' * 70)
        print('Unified LLM Router v2.0 - Benchmark')
        print('=' * 70)
        print(f'\nConfiguration:')
        print(f'  Queries: {num_queries}')
        print(f'  Concurrent: {concurrent}')
        print('\nInitializing router...')

    router = UnifiedRouter(
        enable_batching=True,
        enable_quality_check=True,
        enable_snapshots=True,
        enable_monitoring=True
    )

    if verbose:
        print('✓ Router initialized with all 21 components\n')

    # Query templates
    query_templates = [
        ('Explain quantum computing', RoutingStrategy.QUALITY_FOCUSED),
        ('Write Python function to sort list', RoutingStrategy.BALANCED),
        ('Translate "hello" to Russian', RoutingStrategy.COST_AWARE),
        ('Debug this code: print(x)', RoutingStrategy.BALANCED),
        ('Analyze microservices architecture', RoutingStrategy.CASCADE),
        ('What is the capital of France?', RoutingStrategy.COST_AWARE),
        ('Optimize SQL query performance', RoutingStrategy.QUALITY_FOCUSED),
        ('Convert JSON to XML', RoutingStrategy.COST_AWARE),
        ('Explain neural networks', RoutingStrategy.QUALITY_FOCUSED),
        ('Fix memory leak in C++', RoutingStrategy.BALANCED),
    ]

    # Generate queries
    queries = []
    for i in range(num_queries):
        template_idx = i % len(query_templates)
        query, strategy = query_templates[template_idx]
        queries.append((f"{query} (query {i+1})", strategy))

    if verbose and num_queries <= 10:
        print(f'Running benchmark with {num_queries} queries...\n')
        print(f'{"Query":<35} {"Strategy":<20} {"Model":<20} {"Conf":<8} {"Quality":<8}')
        print('-' * 100)
    else:
        print(f'Running stress test: {num_queries} queries, {concurrent} concurrent...')

    start = time.time()
    results = []
    errors = 0

    # Process queries with concurrency control
    if concurrent == 1:
        # Sequential processing
        for i, (query, strategy) in enumerate(queries):
            try:
                request = UnifiedRequest(
                    query=query,
                    strategy=strategy,
                    user_id=f"benchmark_user_{i % 10}"
                )

                result = await router.route(request)
                results.append(result)

                if verbose and num_queries <= 10:
                    query_short = query[:32] + '...' if len(query) > 32 else query
                    print(f'{query_short:<35} {strategy.value:<20} {result.model:<20} {result.confidence:.3f}    {result.quality_score:.3f}')
                elif i > 0 and i % 10 == 0:
                    print(f'  Processed {i}/{num_queries} queries...')

            except Exception as e:
                errors += 1
                if verbose:
                    print(f'  Error on query {i}: {e}')
    else:
        # Concurrent processing with semaphore
        semaphore = asyncio.Semaphore(concurrent)

        async def process_query(i, query, strategy):
            async with semaphore:
                try:
                    request = UnifiedRequest(
                        query=query,
                        strategy=strategy,
                        user_id=f"benchmark_user_{i % 10}"
                    )
                    result = await router.route(request)
                    return result
                except Exception as e:
                    if verbose:
                        print(f'  Error on query {i}: {e}')
                    return None

        tasks = [process_query(i, q, s) for i, (q, s) in enumerate(queries)]
        results = await asyncio.gather(*tasks)
        results = [r for r in results if r is not None]
        errors = num_queries - len(results)

    elapsed = time.time() - start

    # Summary
    print('\n' + '=' * 70)
    print('Benchmark Results')
    print('=' * 70)
    print(f'Total queries:      {num_queries}')
    print(f'Successful:         {len(results)}')
    print(f'Errors:             {errors}')
    print(f'Total time:         {elapsed:.2f}s')
    print(f'Average latency:    {elapsed/num_queries:.3f}s per query')
    print(f'Throughput:         {num_queries/elapsed:.2f} queries/sec')

    if concurrent > 1:
        print(f'Concurrency:        {concurrent}')
        print(f'Effective QPS:      {num_queries/elapsed:.2f}')

    print('=' * 70)

    # Component statistics
    if verbose:
        print('\nRouter Statistics:')
        stats = router.get_stats()

        print(f'\n  Unified Router:')
        print(f'    Total requests:     {stats["unified"]["total_requests"]}')
        print(f'    Successful:         {stats["unified"]["successful_requests"]}')
        print(f'    Failed:             {stats["unified"]["failed_requests"]}')
        print(f'    Quality retries:    {stats["unified"]["quality_retries"]}')
        print(f'    Avg latency:        {stats["unified"]["avg_latency"]:.3f}s')

        print(f'\n  Context Manager (v2):')
        print(f'    Cache hits:         {stats["context_v2"]["cache_hits"]}')
        print(f'    Cache misses:       {stats["context_v2"]["cache_misses"]}')
        print(f'    Cache hit rate:     {stats["context_v2"]["cache_hits"]/(stats["context_v2"]["cache_hits"]+stats["context_v2"]["cache_misses"])*100 if stats["context_v2"]["cache_hits"]+stats["context_v2"]["cache_misses"] > 0 else 0:.1f}%')
        print(f'    Cache size:         {stats["context_v2"]["cache_size"]}')

        print(f'\n  Quality Check:')
        print(f'    Total checks:       {stats["quality"]["total_checks"]}')
        print(f'    Passed:             {stats["quality"]["passed"]}')
        print(f'    Failed:             {stats["quality"]["failed"]}')

        if "batching" in stats:
            print(f'\n  Batching Layer:')
            print(f'    Total batches:      {stats["batching"]["total_batches"]}')
            print(f'    Total requests:     {stats["batching"]["total_requests"]}')
            if stats["batching"]["total_batches"] > 0:
                print(f'    Avg batch size:     {stats["batching"]["avg_batch_size"]:.2f}')

    print('\n' + '=' * 70)
    print('Benchmark complete!')
    print('=' * 70)

    return {
        'total': num_queries,
        'successful': len(results),
        'errors': errors,
        'elapsed': elapsed,
        'qps': num_queries/elapsed,
        'avg_latency': elapsed/num_queries
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark Unified Router v2.0')
    parser.add_argument('--queries', type=int, default=5, help='Number of queries to run')
    parser.add_argument('--concurrent', type=int, default=1, help='Number of concurrent requests')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')

    args = parser.parse_args()

    asyncio.run(benchmark(
        num_queries=args.queries,
        concurrent=args.concurrent,
        verbose=not args.quiet
    ))
