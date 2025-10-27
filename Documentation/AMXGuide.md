# AMX Programming Guide - Essentials

## Critical Setup

**1. Request OS permission (or your program will segfault):**
```cpp
syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, 18); // XFEATURE_XTILEDATA = 18
```

**2. Configure tiles:**
```cpp
TileConfig cfg{};
cfg.palette_id = 1;  // Always 1
cfg.rows[i] = num_rows;
cfg.colsb[i] = num_bytes;  // BYTES not elements!
_tile_loadconfig(&cfg);
```

**3. Release when done:**
```cpp
_tile_release();  // Must call before exit
```

## Core Constraints

- **8 tile registers** (tmm0-tmm7), max 16 rows × 64 bytes each
- **All pointers must be 64-byte aligned**
- **VNNI layout required** for src2 (B matrix): transpose + pack in groups of 4 (INT8) or 2 (BF16)
- **Stride is in bytes**, not elements

## Instructions

### Setup & Data Movement
```cpp
void _tile_loadd(__tile dst, const void *base, int stride);
void _tile_stored(__tile src, void *base, int stride);
void _tile_zero(__tile tile);

// Non-temporal load (doesn't add to cache)
void _tile_stream_loadd(__tile dst, const void *base, int stride);
```

### INT8 Matrix Multiply (signed/unsigned combinations)
```cpp
_tile_dpbssd(dst, src1, src2);  // signed × signed → INT32
_tile_dpbsud(dst, src1, src2);  // signed × unsigned → INT32
_tile_dpbusd(dst, src1, src2);  // unsigned × signed → INT32
_tile_dpbuud(dst, src1, src2);  // unsigned × unsigned → INT32
```
**Operation:** dst += src1 × src2
- src1: M×K INT8 (K % 4 == 0)
- src2: (K/4)×(N×4) INT8 in VNNI format
- dst: M×N INT32

### BF16 Matrix Multiply
```cpp
_tile_dpbf16ps(dst, src1, src2);  // BF16 × BF16 → FP32
```
**Operation:** dst += src1 × src2
- src1: M×K BF16 (K % 2 == 0)
- src2: (K/2)×(N×2) BF16 in VNNI format
- dst: M×N FP32

## Quick Reference

| Type | Max Tile Size | K Constraint | Output Type |
|------|---------------|--------------|-------------|
| INT8 | 16×64 bytes | K % 4 == 0 | INT32 (16 elements/row) |
| BF16 | 16×64 bytes | K % 2 == 0 | FP32 (16 elements/row) |

## Critical Dimensions

| Metric | INT8 | BF16 |
|--------|------|------|
| Tile size | 16×64 (1024B) | 16×32 (1024B) |
| Ops/tile | 4096 (4 VNNI) | 512 |
| Default L2 block | 384×384×384 | 384×384×384 |
| L2 working set | ~442 KB | ~884 KB |
| Arithmetic intensity | 2048 FLOP/byte | 2048 FLOP/byte |
| Memory stride | 64 bytes | 64 bytes |
| Register blocking | 2×2 (4 tiles) | 2×2 (4 tiles) |
