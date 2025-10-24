# CuTe Layout Algebra: Complete Specification

## Definitions

**IntTuple**: `I ::= ℤ | (I₁, ..., Iₙ)`
- `size(I) = ∏ᵢ Iᵢ`, `rank(I) = 1 if I ∈ ℤ else n`, `depth(I) = 0 if I ∈ ℤ else 1 + max(depth(Iᵢ))`

**Layout**: `L = (S, D)` where `congruent(S, D)`
- Semantics: `L: ℕ → ℤ` via `L(c) = ⟨idx2crd(c, S), D⟩`
- `cosize(L) = L(size(L) - 1) + 1` (codomain size)

**Coordinate Mapping** (colexicographical, right-to-left):
```
idx2crd(i, s) = i % s                                    if s ∈ ℤ
idx2crd(i, (s₁,...,sₙ)) = (idx2crd(i % size(s₁), s₁),
                            idx2crd(⌊i/size(s₁)⌋, (s₂,...,sₙ)))
```

**Index Mapping** (inner product):
```
crd2idx(c, s, d) = c · d                                 if s, d ∈ ℤ
crd2idx(c, S, D) = Σᵢ crd2idx(idx2crd(cᵢ, Sᵢ), Sᵢ, Dᵢ)   otherwise
```

---

## Operations

### Coalesce: `c(L) → L`

**Rules** (binary on flattened, left-to-right):
```
s₀:d₀ ++ s₁:d₁ → s₀:d₀           if s₁ = 1
               → s₁:d₁           if s₀ = 1
               → s₀·s₁:d₀        if d₁ = s₀·d₀
               → (s₀,s₁):(d₀,d₁) otherwise
```

Postcondition: `size(c(L)) = size(L) ∧ ∀i, c(L)(i) = L(i)`

---

### Composition: `A ∘ B → L`

**Semantics**: `(A ∘ B)(c) = A(B(c))`

**Left-Distributivity**: `A ∘ (B₁,...,Bₙ) = (A ∘ B₁,...,A ∘ Bₙ)` when B injective

**Algorithm for** `A ∘ s:d`:
1. **Divide**: Transform A to "every dᵗʰ element"
   - For each mode `sᵢ:dᵢ` in A, compute strided version
   - Residue tracks remaining stride to divide out
2. **Mod**: Keep first s elements
   - Truncate shape to accumulated size ≥ s

Postcondition: `compatible(B, A ∘ B) ∧ ∀i < size(B), (A ∘ B)(i) = A(B(i))`

---

### Tiler Concept

**Definition** (recursive):
```
T ::= L                    (Layout)
    | (T₁, ..., Tₙ)       (tuple of Tilers)
    | S                    (Shape, interpreted as S:1̄)
```

**Interpretation**: `τ(T)`
```
τ(L) = L                           if L is a Layout
τ(S) = S:1̄                         if S is a Shape
τ((T₁,...,Tₙ)) = (τ(T₁),...,τ(Tₙ)) (recursive)
```

**Application**: For operation `op(A, T)` where `rank(A) = r`:
- If `rank(τ(T)) = r`: apply by-mode `(op(A⟨0⟩,τ(T)⟨0⟩), ..., op(A⟨r-1⟩,τ(T)⟨r-1⟩))`
- If `rank(τ(T)) = 1`: apply to flattened A
- Otherwise: error (rank mismatch)

---

### Complement: `A* = complement(A, M) → L`

**Post-conditions** for `R = complement(A, M)`:
1. **Coverage**: `cosize((A, R)) ≥ size(M)`
2. **Minimality**: `cosize(R) ≥ ⌈size(M) / cosize(A)⌉`
3. **Ordering**: `∀i ∈ [1, size(R)), R(i-1) < R(i)` (unique)
4. **Disjointness**: `∀i,j: R(i) ≠ A(j)`

**Algorithm** (for `A = S:D`, `M` with `size(M) = m`):
```python
min_stride = min(flatten(D))
period = size(A) * min_stride              # NOT cosize(A)!

residues_covered = {A(i) % min_stride | i < size(A)}
residues_needed = [0, min_stride) \ residues_covered

# Inner offsets (residue classes)
inner_offsets = [0] ∪ residues_needed if residues_needed else [0]

# Outer periods (tiling repetitions)
num_periods = ⌈m / period⌉

return (len(inner_offsets), num_periods):(1, period)
```

**IntTuple Cotarget**: When `M` is IntTuple (not scalar), result preserves static structure for divisibility.

**Semantics**: Complement finds **tiling offsets**. The concatenated layout `(A, A*)` iterates colexicographically to cover `[0, M)` exactly once.

**Example**: `complement(4:2, 24) = (2,3):(1,8)`
- `min_stride = 2, period = 8`
- `residues_covered = {0}, residues_needed = [1]`
- `inner_offsets = [0, 1], num_periods = 3`
- Result: `(2,3):(1,8)` = 2 offsets × 3 periods

Postcondition: `∀c ∈ [0, M), ∃!i: c ∈ image((A, A*)(i))`

---

### Division: `A ⊘ B → L`

**Definition**: `A ⊘ B := A ∘ (B, B*)` where `B* = complement(B, size(A))`

**Semantics**: Partition A into (tile, rest) where tile = A ∘ B

**Type Wrapping Critical**:
```python
# WRONG: Flattens structure
combined = Layout(B.shape + Bstar.shape, B.stride + Bstar.stride)

# RIGHT: Preserves rank-2 semantics
combined = Layout((B.shape, Bstar.shape), (B.stride, Bstar.stride))
```

Postcondition: `layout⟨0⟩(A ⊘ B) ≡ A ∘ B`

---

### Product: `A ⊗ B → L`

**Definition**: `A ⊗ B := (A, A* ∘ B)` where `A* = complement(A, size(A) · cosize(B))`

**Semantics**: Replicate A according to pattern B

**Type Wrapping Critical**:
```python
# Always create rank-2, preserve A as unit in mode-0
return Layout((A.shape, composed.shape), (A.stride, composed.stride))
# NOT: flatten or merge A's modes
```

Postcondition: `compatible(A, layout⟨0⟩(A ⊗ B)) ∧ compatible(B, layout⟨1⟩(A ⊗ B))`

---

## Variant Operations

### Division Variants

**Input**: Layout `A = (M, N, L, ...): (d_M, d_N, d_L, ...)`, Tiler `B = <T_M, T_N>`

**Base**: `logical_divide(A, B)` applies by-mode:
```
logical_divide(A, B) = ((T_M, R_M), (T_N, R_N), L, ...)
where T_M = M ∘ B⟨0⟩, R_M = complement(B⟨0⟩, size(M))
```

**Projections**: `π_tile((T,R)) = T`, `π_rest((T,R)) = R`

**Variants**:
```
zipped_divide(A, B) = ((T_M, T_N), (R_M, R_N, L, ...))
tiled_divide(A, B)  = ((T_M, T_N), R_M, R_N, L, ...)
flat_divide(A, B)   = (T_M, T_N, R_M, R_N, L, ...)
```

**Invariant**: `layout⟨0⟩(zipped_divide(A, B)) ≡ A ∘ B`

---

### Product Variants

**Input**: Layouts `A = (M, N, ...): (d_M, d_N, ...)`, `B = (P, Q, ...): (d_P, d_Q, ...)`

**Requirement**: `rank(A) = rank(B)` (rank-sensitive)

**Base**: `logical_product(A, B)` applies by-mode:
```
logical_product(A, B) = ((M, T_M), (N, T_N), ...)
where T_M = complement(A⟨0⟩, size(A⟨0⟩)·cosize(B⟨0⟩)) ∘ B⟨0⟩
```

**Variants**:
```
blocked_product(A, B) = ((M, T_M), (N, T_N), ...)          # tile-first
raked_product(A, B)   = ((T_M, M), (T_N, N), ...)          # rest-first (interleaved)
zipped_product(A, B)  = ((M, N, ...), (T_M, T_N, ...))
tiled_product(A, B)   = ((M, N, ...), T_M, T_N, ...)
flat_product(A, B)    = (M, N, T_M, T_N, ...)
```

**Semantics**:
- **blocked_product**: Tile A appears as "blocks" arranged by B
- **raked_product**: Elements of A interleaved ("cyclic distribution") by B

---

## Summary Table

| Operation | Input Ranks | Result Structure | Semantics |
|-----------|-------------|------------------|-----------|
| `logical_divide(A,B)` | any | `((T,R), ...)` per mode | Partition by-mode |
| `zipped_divide(A,B)` | any | `(Tiles, Rests)` | Gather tiles, gather rests |
| `tiled_divide(A,B)` | any | `(Tiles, R₀, R₁, ...)` | Tile unit, flatten rests |
| `flat_divide(A,B)` | any | `(T₀, T₁, R₀, R₁, ...)` | Fully flatten |
| `logical_product(A,B)` | `r = r` | `((M,T), ...)` per mode | Replicate by-mode |
| `blocked_product(A,B)` | `r = r` | rank-r, blocks | Tile-first combination |
| `raked_product(A,B)` | `r = r` | rank-r, raked | Rest-first (interleaved) |
| `zipped_product(A,B)` | `r = r` | `(Originals, Tiles)` | Group by origin |
| `tiled_product(A,B)` | `r = r` | `(Originals, T₀, T₁, ...)` | Flatten tiles |
| `flat_product(A,B)` | `r = r` | `(M, N, T₀, T₁, ...)` | Fully flatten |

---

## Key Theorems

**T1**: `(A ∘ B) ∘ C ≡ A ∘ (B ∘ C)` (associativity)

**T2**: `layout⟨0⟩(A ⊘ B) ≡ A ∘ B` (division-composition duality)

**T3**: `layout⟨0⟩(A ⊗ B)` reproduces A (product preserves tile)

**T4**: `∀variant ∈ {zipped, tiled, flat}, layout⟨0⟩(variant_divide(A,B)) ≡ A ∘ B`

**T5**: `rank(blocked_product(A,B)) = rank(raked_product(A,B)) = rank(A) = rank(B)`

**T6**: `(A, complement(A, M))` partitions `[0, M)` via colexicographical iteration

---

## Implementation Critical Points

### Size vs Cosize vs Period
```
size(L)   = |domain|     # iteration bounds
cosize(L) = L(size-1)+1  # codomain span
period    = size·min_stride  # tiling repetition (for complement)
```

### Colexicographical Concatenation
`(A, B)` iterates: exhaust all of A first, then advance in B
- **Not** Cartesian product (A₀B₀, A₀B₁, A₁B₀, ...)
- **Is** Right-to-left (A₀B₀, A₁B₀, A₂B₀, ..., AₙB₀, A₀B₁, ...)

### Static Integer Propagation
All operations preserve compile-time known values. With C++ template metaprogramming:
```cpp
auto layout = make_layout(Int<4>{}, Int<2>{});
auto result = composition(layout, Int<3>{}); // Computed at compile time
// Zero runtime cost
```

### Tiler Application
Operations accept `T ::= Layout | (T₁, ..., Tₙ) | Shape`
- Apply operation by-mode via left-distributivity
- Shape interpreted as stride-1 layouts

---

## Notation

- `S:D` = `Layout(S, D)`
- `(A, B)` = concatenation (creates multi-mode)
- `<A, B>` = tiler tuple (distinguishes from concatenation)
- `⟨x, y⟩` = inner product
- `⌈x⌉` = ceiling
- `≡` = functional equivalence
- `A⟨i⟩` = mode i of A
- `1̄` = all-ones stride tuple
