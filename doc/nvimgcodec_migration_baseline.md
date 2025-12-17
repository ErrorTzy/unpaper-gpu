# nvImageCodec Migration Baseline

Baseline performance measurements before migrating from direct nvJPEG to unified nvImageCodec API.

**Date**: 2025-12-17
**Hardware**: CUDA GPU (see system info below)
**Software**: unpaper with nvJPEG decode/encode

## Test Configuration

- **Image count**: 50 JPEG images
- **Image source**: imgsrc001.png converted to JPEG (quality 95)
- **CUDA streams**: 8
- **Jobs**: 8
- **Iterations**: 3 (after 1 warmup for pipeline test)

## Baseline Results

### Full Processing Pipeline (bench_jpeg_pipeline.py)

Tests the complete workflow including all image processing filters.

| Path | Total Time | Per Image | Throughput |
|------|------------|-----------|------------|
| Standard (nvJPEG decode → GPU → D2H → PBM) | 13307ms | 266.1ms | 3.76 img/s |
| GPU Pipeline (nvJPEG decode → GPU → nvJPEG encode → JPEG) | 12641ms | 252.8ms | 3.96 img/s |

**Speedup**: 1.05x (GPU pipeline vs standard path)
**Time saved**: 666ms total (13.3ms per image)

Output sizes:
- Standard (PBM): 414.7 MB total (8493.5 KB/img)
- GPU (JPEG): 63.4 MB total (1298.4 KB/img)

### Decode-Only Benchmark (bench_batch.py --no-processing)

Tests raw decode/encode throughput without image processing filters.

| Configuration | Total Time | Per Image | Throughput |
|---------------|------------|-----------|------------|
| CUDA batch (jobs=8, no processing) | 2020ms | 40.4ms | 24.8 img/s |

## Key Observations

1. **Processing dominates runtime**: Full processing takes ~253ms/img vs ~40ms/img decode-only, showing that image processing filters (blackfilter, deskew, etc.) account for ~84% of total time.

2. **nvJPEG encode provides modest speedup**: The GPU encode path is only ~5% faster than D2H + CPU encode for full processing workflows. The benefit is more significant for decode-heavy workloads.

3. **Raw throughput is good**: At ~25 img/s for decode-only, the nvJPEG pipeline has headroom. The bottleneck is in the processing filters.

## Migration Success Criteria

After migrating to nvImageCodec, performance should be:
- **Within 5% of baseline** for all metrics
- No regression in decode-only throughput (~25 img/s)
- No regression in full pipeline throughput (~4 img/s)

## Commands Used

```bash
# Full processing pipeline
python tools/bench_jpeg_pipeline.py --images 50 --warmup 1 --iterations 3

# Decode-only (no processing)
python tools/bench_batch.py --images 50 --jpeg --devices cuda --threads 8 --no-processing --iterations 3
```
