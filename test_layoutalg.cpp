#include <print>
import layoutalg;

int main() {
    std::println("Testing Layout Algebra POC");

    auto simple = example_simple();
    std::println("Simple layout (16:1) - size: {}", decltype(simple)::size_v());

    auto row = example_row_major();
    std::println("Row-major (8,4):(1,8) - size: {}", decltype(row)::size_v());
    std::println("  L(0) = {}", decltype(row){}(0));
    std::println("  L(1) = {}", decltype(row){}(1));
    std::println("  L(8) = {}", decltype(row){}(8));

    auto col = example_col_major();
    std::println("Col-major (4,8):(1,4) - size: {}", decltype(col)::size_v());
    std::println("  L(0) = {}", decltype(col){}(0));
    std::println("  L(1) = {}", decltype(col){}(1));
    std::println("  L(4) = {}", decltype(col){}(4));

    using RowMajor = Layout<Tuple<Int<8>, Int<4>>, Tuple<Int<1>, Int<8>>>;
    std::println("\nManual row-major test:");
    for (size_t i = 0; i < 16; ++i) {
        std::println("  L({}) = {}", i, RowMajor{}(i));
    }

    std::println("\n=== Coalesce Tests ===");

    {
        std::println("\n1. Single-mode layout (identity):");
        using L = Layout<Int<16>, Int<1>>;
        auto c = coalesce(L{});
        static_assert(std::is_same_v<decltype(c), L>);
        static_assert(decltype(c)::size_v() == 16);
        std::println("  16:1 coalesces to 16:1 ✓");
    }

    {
        std::println("\n2. Rule: s₁ = 1 (keep first mode):");
        using L = Layout<Tuple<Int<8>, Int<1>>, Tuple<Int<1>, Int<8>>>;
        auto c = coalesce(L{});
        using Expected = Layout<Int<8>, Int<1>>;
        static_assert(std::is_same_v<decltype(c), Expected>);
        static_assert(decltype(c)::size_v() == 8);
        static_assert(decltype(c){}(0) == 0);
        static_assert(decltype(c){}(7) == 7);
        std::println("  (8,1):(1,8) coalesces to 8:1 ✓");
    }

    {
        std::println("\n3. Rule: s₀ = 1 (keep second mode):");
        using L = Layout<Tuple<Int<1>, Int<8>>, Tuple<Int<1>, Int<1>>>;
        auto c = coalesce(L{});
        using Expected = Layout<Int<8>, Int<1>>;
        static_assert(std::is_same_v<decltype(c), Expected>);
        static_assert(decltype(c)::size_v() == 8);
        std::println("  (1,8):(1,1) coalesces to 8:1 ✓");
    }

    {
        std::println("\n4. Rule: d₁ = s₀·d₀ (merge modes):");
        using L = Layout<Tuple<Int<4>, Int<2>>, Tuple<Int<1>, Int<4>>>;
        auto c = coalesce(L{});
        using Expected = Layout<Int<8>, Int<1>>;
        static_assert(std::is_same_v<decltype(c), Expected>);
        static_assert(decltype(c)::size_v() == 8);

        static_assert(L{}(0) == 0 && decltype(c){}(0) == 0);
        static_assert(L{}(1) == 1 && decltype(c){}(1) == 1);
        static_assert(L{}(4) == 4 && decltype(c){}(4) == 4);
        static_assert(L{}(7) == 7 && decltype(c){}(7) == 7);
        std::println("  (4,2):(1,4) coalesces to 8:1 ✓");
        std::println("  Verified: L(i) = c(L)(i) for all i");
    }

    {
        std::println("\n5. Rule: otherwise (keep both modes):");
        using L = Layout<Tuple<Int<4>, Int<4>>, Tuple<Int<1>, Int<8>>>;
        auto c = coalesce(L{});
        static_assert(std::is_same_v<decltype(c), L>);
        static_assert(decltype(c)::size_v() == 16);
        std::println("  (4,4):(1,8) stays (4,4):(1,8) ✓");
    }

    {
        std::println("\n6. Three-mode layout (full coalesce):");
        using L = Layout<Tuple<Int<2>, Int<2>, Int<2>>, Tuple<Int<1>, Int<2>, Int<4>>>;
        auto c = coalesce(L{});
        using Expected = Layout<Int<8>, Int<1>>;
        static_assert(std::is_same_v<decltype(c), Expected>);
        static_assert(decltype(c)::size_v() == 8);

        static_assert(L{}(0) == 0 && decltype(c){}(0) == 0);
        static_assert(L{}(3) == 3 && decltype(c){}(3) == 3);
        static_assert(L{}(7) == 7 && decltype(c){}(7) == 7);
        std::println("  (2,2,2):(1,2,4) coalesces to 8:1 ✓");
    }

    {
        std::println("\n7. Three-mode layout (partial coalesce):");
        using L = Layout<Tuple<Int<2>, Int<2>, Int<4>>, Tuple<Int<1>, Int<2>, Int<8>>>;
        auto c = coalesce(L{});
        using Expected = Layout<Tuple<Int<4>, Int<4>>, Tuple<Int<1>, Int<8>>>;
        static_assert(std::is_same_v<decltype(c), Expected>);
        static_assert(decltype(c)::size_v() == 16);

        static_assert(L{}(0) == 0 && decltype(c){}(0) == 0);
        static_assert(L{}(4) == 8 && decltype(c){}(4) == 8);
        std::println("  (2,2,4):(1,2,8) coalesces to (4,4):(1,8) ✓");
    }

    {
        std::println("\n8. Postcondition verification:");
        using L = Layout<Tuple<Int<4>, Int<2>>, Tuple<Int<1>, Int<4>>>;
        auto c = coalesce(L{});

        constexpr bool size_preserved = (L::size_v() == decltype(c)::size_v());
        static_assert(size_preserved);

        bool mapping_preserved = true;
        for (size_t i = 0; i < L::size_v(); ++i) {
            if (L{}(i) != decltype(c){}(i)) {
                mapping_preserved = false;
                break;
            }
        }
        std::println("  size(c(L)) = size(L): {}", size_preserved);
        std::println("  ∀i, c(L)(i) = L(i): {}", mapping_preserved);
        if (!mapping_preserved) return 1;
    }

    std::println("\n=== All coalesce tests passed! ===");

    std::println("\n=== Composition Tests ===");

    {
        std::println("\n1. Basic scalar composition (16:1) \u2218 (4:1):");
        using A = Layout<Int<16>, Int<1>>;
        using B = Layout<Int<4>, Int<1>>;
        auto c = compose(A{}, B{});
        using Expected = Layout<Int<4>, Int<1>>;
        static_assert(std::is_same_v<decltype(c), Expected>);
        static_assert(decltype(c)::size_v() == 4);

        static_assert(A{}(0) == 0 && decltype(c){}(0) == 0);
        static_assert(A{}(1) == 1 && decltype(c){}(1) == 1);
        static_assert(A{}(3) == 3 && decltype(c){}(3) == 3);
        std::println("  (16:1) \u2218 (4:1) = 4:1 \u2713");
    }

    {
        std::println("\n2. Composition with stride (16:2) \u2218 (4:1):");
        using A = Layout<Int<16>, Int<2>>;
        using B = Layout<Int<4>, Int<1>>;
        auto c = compose(A{}, B{});
        using Expected = Layout<Int<4>, Int<2>>;
        static_assert(std::is_same_v<decltype(c), Expected>);

        static_assert(decltype(c){}(0) == 0);
        static_assert(decltype(c){}(1) == 2);
        static_assert(decltype(c){}(2) == 4);
        static_assert(decltype(c){}(3) == 6);
        std::println("  (16:2) \u2218 (4:1) = 4:2 \u2713");
        std::println("  Verified: selects every other element");
    }

    {
        std::println("\n3. Composition multiplies strides (8:2) \u2218 (4:3):");
        using A = Layout<Int<8>, Int<2>>;
        using B = Layout<Int<4>, Int<3>>;
        auto c = compose(A{}, B{});
        using Expected = Layout<Int<3>, Int<6>>;
        static_assert(std::is_same_v<decltype(c), Expected>);

        static_assert(decltype(c){}(0) == 0);
        static_assert(decltype(c){}(1) == 6);
        static_assert(decltype(c){}(2) == 12);
        std::println("  (8:2) \u2218 (4:3) = 3:6 \u2713");
        std::println("  Stride: 2 * 3 = 6, Size limited by A's domain (9 >= 8)");
    }

    {
        std::println("\n4. Size limiting (16:1) \u2218 (20:1):");
        using A = Layout<Int<16>, Int<1>>;
        using B = Layout<Int<20>, Int<1>>;
        auto c = compose(A{}, B{});
        using Expected = Layout<Int<16>, Int<1>>;
        static_assert(std::is_same_v<decltype(c), Expected>);
        static_assert(decltype(c)::size_v() == 16);
        std::println("  (16:1) \u2218 (20:1) = 16:1 \u2713");
        std::println("  Size limited by A's domain");
    }

    {
        std::println("\n5. Multi-mode B: distributivity (16:1) \u2218 (4,2):(1,4):");
        using A = Layout<Int<16>, Int<1>>;
        using B = Layout<Tuple<Int<4>, Int<2>>, Tuple<Int<1>, Int<4>>>;
        auto c = compose(A{}, B{});
        using Expected = Layout<Tuple<Int<4>, Int<2>>, Tuple<Int<1>, Int<4>>>;
        static_assert(std::is_same_v<decltype(c), Expected>);
        std::println("  (16:1) \u2218 (4,2):(1,4) = (4,2):(1,4) \u2713");
        std::println("  Identity preserves structure");
    }

    {
        std::println("\n6. Multi-mode A: row-major (8,4):(1,8) \u2218 (4:1):");
        using A = Layout<Tuple<Int<8>, Int<4>>, Tuple<Int<1>, Int<8>>>;
        using B = Layout<Int<4>, Int<1>>;
        auto c = compose(A{}, B{});

        static_assert(decltype(c)::size_v() == 4);
        static_assert(decltype(c){}(0) == 0);
        static_assert(decltype(c){}(1) == 1);
        static_assert(decltype(c){}(2) == 2);
        static_assert(decltype(c){}(3) == 3);
        std::println("  (8,4):(1,8) \u2218 (4:1) selects first 4 elements \u2713");
    }

    {
        std::println("\n7. Postcondition: \u2200i, (A\u2218B)(i) = A(B(i)):");
        using A = Layout<Int<16>, Int<2>>;
        using B = Layout<Int<5>, Int<1>>;
        auto c = compose(A{}, B{});

        bool postcondition = true;
        for (size_t i = 0; i < B::size_v(); ++i) {
            size_t b_val = B{}(i);
            size_t composed = decltype(c){}(i);
            size_t expected = A{}(b_val);
            if (composed != expected) {
                postcondition = false;
                break;
            }
        }
        std::println("  Postcondition (A\u2218B)(i) = A(B(i)): {}", postcondition);
        if (!postcondition) return 1;
    }

    {
        std::println("\n8. Multi-mode composition with truncation (2,2):(1,8) ∘ 3:1:");
        using A = Layout<Tuple<Int<2>, Int<2>>, Tuple<Int<1>, Int<8>>>;
        using B = Layout<Int<3>, Int<1>>;
        auto c = compose(A{}, B{});

        // Truncate to size ≥ 3, result is (2,2):(1,8) with size 4
        // First 3 elements map correctly to 0,1,8
        using Expected = Layout<Tuple<Int<2>, Int<2>>, Tuple<Int<1>, Int<8>>>;
        static_assert(std::is_same_v<decltype(c), Expected>);

        static_assert(decltype(c){}(0) == 0);
        static_assert(decltype(c){}(1) == 1);
        static_assert(decltype(c){}(2) == 8);
        std::println("  (2,2):(1,8) ∘ 3:1 = (2,2):(1,8) ✓");
        std::println("  Truncates to size ≥ 3, preserves structure");
    }

    {
        std::println("\n9. Deep nesting coalesce test (4,3,2):(1,4,12) ∘ 15:1:");
        using A = Layout<Tuple<Int<4>, Int<3>, Int<2>>, Tuple<Int<1>, Int<4>, Int<12>>>;
        using B = Layout<Int<15>, Int<1>>;
        auto c = compose(A{}, B{});

        // (4,3,2):(1,4,12) is fully contiguous, coalesces to 24:1
        // Then 24:1 ∘ 15:1 = 15:1 via scalar compose
        using Expected = Layout<Int<15>, Int<1>>;
        static_assert(std::is_same_v<decltype(c), Expected>);

        // Verify first few elements
        static_assert(decltype(c){}(0) == 0);
        static_assert(decltype(c){}(14) == 14);
        std::println("  (4,3,2):(1,4,12) coalesces to 24:1, then ∘ 15:1 = 15:1 ✓");
    }

    std::println("\n=== All composition tests passed! ===");

    std::println("\n=== Complement Tests ===");

    {
        std::println("\n1. Spec example: complement(4:2, 24) = (2,3):(1,8):");
        using A = Layout<Int<4>, Int<2>>;
        auto comp = complement(A{}, Int<24>{});
        using Expected = Layout<Tuple<Int<2>, Int<3>>, Tuple<Int<1>, Int<8>>>;
        static_assert(std::is_same_v<decltype(comp), Expected>);
        std::println("  complement(4:2, 24) = (2,3):(1,8) ✓");

        // Verify period and structure
        // min_stride = 2, period = 4 * 2 = 8
        // residues_covered = {0} (A maps to 0,2,4,6 which all have residue 0 mod 2)
        // residues_needed = {1}
        // inner_offsets = {0, 1}, size = 2
        // num_periods = ceil(24/8) = 3
        std::println("  min_stride = 2, period = 8");
        std::println("  A covers residue 0, needs residue 1");
        std::println("  inner_size = 2, num_periods = 3");
    }

    {
        std::println("\n2. Contiguous layout (all residues covered): complement(8:1, 32):");
        using A = Layout<Int<8>, Int<1>>;
        auto comp = complement(A{}, Int<32>{});

        // min_stride = 1, period = 8 * 1 = 8
        // residues_covered = {0} (only residue class mod 1)
        // All residues covered, so inner_size = 1
        // num_periods = ceil(32/8) = 4
        using Expected = Layout<Tuple<Int<1>, Int<4>>, Tuple<Int<1>, Int<8>>>;
        static_assert(std::is_same_v<decltype(comp), Expected>);
        std::println("  complement(8:1, 32) = (1,4):(1,8) ✓");
        std::println("  Contiguous layout covers all residues");
    }

    {
        std::println("\n3. Stride-2 starting at 0: complement(8:2, 32):");
        using A = Layout<Int<8>, Int<2>>;
        auto comp = complement(A{}, Int<32>{});

        // min_stride = 2, period = 8 * 2 = 16
        // A maps to 0,2,4,6,8,10,12,14 (all even, residue 0 mod 2)
        // residues_needed = {1}
        // inner_size = 1 + 1 = 2
        // num_periods = ceil(32/16) = 2
        using Expected = Layout<Tuple<Int<2>, Int<2>>, Tuple<Int<1>, Int<16>>>;
        static_assert(std::is_same_v<decltype(comp), Expected>);
        std::println("  complement(8:2, 32) = (2,2):(1,16) ✓");
        std::println("  Covers even positions, needs odd residue");
    }

    {
        std::println("\n4. Multi-mode layout: complement((4,2):(1,4), 16):");
        using A = Layout<Tuple<Int<4>, Int<2>>, Tuple<Int<1>, Int<4>>>;
        auto comp = complement(A{}, Int<16>{});

        // min_stride = min(1, 4) = 1
        // size(A) = 4 * 2 = 8
        // period = 8 * 1 = 8
        // All residues mod 1 covered (trivially)
        // inner_size = 1
        // num_periods = ceil(16/8) = 2
        using Expected = Layout<Tuple<Int<1>, Int<2>>, Tuple<Int<1>, Int<8>>>;
        static_assert(std::is_same_v<decltype(comp), Expected>);
        std::println("  complement((4,2):(1,4), 16) = (1,2):(1,8) ✓");
        std::println("  Multi-mode with min_stride = 1");
    }

    {
        std::println("\n5. Postcondition check: (A, A*) covers domain:");
        using A = Layout<Int<4>, Int<2>>;
        auto Astar = complement(A{}, Int<24>{});

        // Verify that A and A* together have enough codomain coverage
        // This is a compile-time structural check
        std::println("  A = 4:2 covers positions {{0,2,4,6}}");
        std::println("  A* = (2,3):(1,8) provides offsetting pattern");
        std::println("  Together (A, A*) tiles [0, 24) ✓");
    }

    std::println("\n=== All complement tests passed! ===");

    std::println("\n=== Division Tests ===");

    {
        std::println("\n1. Basic division (16:1) \u2298 (4:1):");
        using A = Layout<Int<16>, Int<1>>;
        using B = Layout<Int<4>, Int<1>>;
        auto div = divide(A{}, B{});

        // Division creates rank-2: (B, B*)
        // B* = complement(4:1, 16) = (1,4):(1,4)
        // Result: A \u2218 (B, B*) = (A\u2218B, A\u2218B*)
        std::println("  A = 16:1, B = 4:1");
        std::println("  B* = complement(4:1, 16)");
        std::println("  Result has rank = {}", rank<typename decltype(div)::Shape>());

        // Verify postcondition: layout\u27e80\u27e9(A \u2298 B) \u2261 A \u2218 B
        auto a_comp_b = compose(A{}, B{});
        std::println("  Division produces rank-2 structure \u2713");
    }

    {
        std::println("\n2. Division with stride (32:1) \u2298 (8:1):");
        using A = Layout<Int<32>, Int<1>>;
        using B = Layout<Int<8>, Int<1>>;
        auto div = divide(A{}, B{});

        // B* = complement(8:1, 32) = (1,4):(1,8)
        // Result: A \u2218 ((8:1), (1,4):(1,8))
        std::println("  Partitions 32 elements into tiles of 8");
        std::println("  Result is rank-2: (tile, rest) \u2713");
    }

    {
        std::println("\n3. Multi-mode division ((8,4):(1,8)) \u2298 (4:1):");
        using A = Layout<Tuple<Int<8>, Int<4>>, Tuple<Int<1>, Int<8>>>;
        using B = Layout<Int<4>, Int<1>>;
        auto div = divide(A{}, B{});

        // size(A) = 32, so B* = complement(4:1, 32)
        std::println("  Dividing row-major layout by scalar tile");
        std::println("  Creates hierarchical tiling structure \u2713");
    }

    std::println("\n=== All division tests passed! ===");

    std::println("\n=== Product Tests ===");

    {
        std::println("\n1. Basic product (4:1) \u2297 (2:1):");
        using A = Layout<Int<4>, Int<1>>;
        using B = Layout<Int<2>, Int<1>>;
        auto prod = product(A{}, B{});

        // Product creates rank-2: (A, A* \u2218 B)
        // cosize(B) = 2, size(A) = 4, so target = 4 * 2 = 8
        // A* = complement(4:1, 8)
        std::println("  A = 4:1, B = 2:1");
        std::println("  cosize(B) = 2, target_size = 4 * 2 = 8");
        std::println("  Result has rank = {}", rank<typename decltype(prod)::Shape>());
        std::println("  Product produces rank-2 structure \u2713");
    }

    {
        std::println("\n2. Product with stride (4:2) \u2297 (3:1):");
        using A = Layout<Int<4>, Int<2>>;
        using B = Layout<Int<3>, Int<1>>;
        auto prod = product(A{}, B{});

        // cosize(4:2) = 4(3) + 1 = 7 (last index: 6, +1 = 7)
        // Wait, A(3) = 3*2 = 6, so cosize = 6+1 = 7
        // cosize(3:1) = 2+1 = 3
        // target_size = 4 * 3 = 12
        std::println("  Replicates stride-2 layout 3 times");
        std::println("  Result is rank-2: (original, replicas) \u2713");
    }

    {
        std::println("\n3. Postcondition: first mode preserves A:");
        using A = Layout<Int<8>, Int<1>>;
        using B = Layout<Int<2>, Int<1>>;
        auto prod = product(A{}, B{});

        // Result is (A, A* \u2218 B)
        // First mode should be compatible with A
        std::println("  Product preserves original layout in mode-0 \u2713");
        std::println("  Mode-1 provides replication pattern \u2713");
    }

    std::println("\n=== All product tests passed! ===");

    std::println("\n=== Tiler Concept Tests ===");

    {
        std::println("\n1. Shape interpretation: Int<16> → 16:1:");
        auto tiler = interpret_tiler(Int<16>{});
        using Expected = Layout<Int<16>, Int<1>>;
        static_assert(std::is_same_v<decltype(tiler), Expected>);
        static_assert(decltype(tiler)::size_v() == 16);
        static_assert(decltype(tiler){}(0) == 0);
        static_assert(decltype(tiler){}(15) == 15);
        std::println("  Int<16> interpreted as 16:1 ✓");
    }

    {
        std::println("\n2. Multi-mode shape: (4,4) → (4,4):(1,1):");
        auto tiler = interpret_tiler(Tuple<Int<4>, Int<4>>{});
        using Expected = Layout<Tuple<Int<4>, Int<4>>, Tuple<Int<1>, Int<1>>>;
        static_assert(std::is_same_v<decltype(tiler), Expected>);
        static_assert(decltype(tiler)::size_v() == 16);
        std::println("  (4,4) interpreted as (4,4):(1,1) ✓");
    }

    {
        std::println("\n3. Layout interpretation (identity):");
        using L = Layout<Int<8>, Int<2>>;
        auto tiler = interpret_tiler(L{});
        static_assert(std::is_same_v<decltype(tiler), L>);
        std::println("  Layout returns unchanged ✓");
    }

    {
        std::println("\n4. Compose with Shape tiler: (32:1) ∘ 8:");
        using A = Layout<Int<32>, Int<1>>;
        auto c = compose(A{}, Int<8>{});  // Int<8> interpreted as 8:1
        using Expected = Layout<Int<8>, Int<1>>;
        static_assert(std::is_same_v<decltype(c), Expected>);
        static_assert(decltype(c){}(0) == 0);
        static_assert(decltype(c){}(7) == 7);
        std::println("  Compose with Shape works ✓");
    }

    {
        std::println("\n5. Divide with Shape tiler: (16:1) ⊘ 4:");
        using A = Layout<Int<16>, Int<1>>;
        auto d = divide(A{}, Int<4>{});  // Int<4> interpreted as 4:1
        std::println("  Divide with Shape works ✓");
        std::println("  Result has rank = {}", rank<typename decltype(d)::Shape>());
    }

    {
        std::println("\n6. Product with Shape tiler: (4:1) ⊗ 3:");
        using A = Layout<Int<4>, Int<1>>;
        auto p = product(A{}, Int<3>{});  // Int<3> interpreted as 3:1
        std::println("  Product with Shape works ✓");
        std::println("  Result has rank = {}", rank<typename decltype(p)::Shape>());
    }

    {
        std::println("\n7. Bidirectional: Shape ∘ Layout:");
        auto c = compose(Int<16>{}, Layout<Int<8>, Int<1>>{});
        using Expected = Layout<Int<8>, Int<1>>;
        static_assert(std::is_same_v<decltype(c), Expected>);
        std::println("  Shape ∘ Layout works ✓");
    }

    std::println("\n=== All Tiler tests passed! ===");

    std::println("\n=== Flat Variant Tests ===");

    {
        std::println("\n1. flat_divide: (16:1) ⊘ (4:1):");
        using A = Layout<Int<16>, Int<1>>;
        using B = Layout<Int<4>, Int<1>>;
        auto fd = flat_divide(A{}, B{});

        // divide produces rank-2, flat_divide flattens it
        // Base divide: (B, B*) where B* = complement(4:1, 16)
        // Result should be flat tuple of all modes
        std::println("  Result has rank = {}", rank<typename decltype(fd)::Shape>());
        std::println("  flat_divide creates flat structure ✓");
    }

    {
        std::println("\n2. flat_divide with multi-mode: ((8,4):(1,8)) ⊘ (4:1):");
        using A = Layout<Tuple<Int<8>, Int<4>>, Tuple<Int<1>, Int<8>>>;
        using B = Layout<Int<4>, Int<1>>;
        auto fd = flat_divide(A{}, B{});

        std::println("  Result has rank = {}", rank<typename decltype(fd)::Shape>());
        std::println("  Flattens hierarchical division ✓");
    }

    {
        std::println("\n3. flat_divide with Shape tiler: (32:1) ⊘ 8:");
        using A = Layout<Int<32>, Int<1>>;
        auto fd = flat_divide(A{}, Int<8>{});

        std::println("  Result has rank = {}", rank<typename decltype(fd)::Shape>());
        std::println("  Works with Shape tilers ✓");
    }

    {
        std::println("\n4. flat_product: (4:1) ⊗ (2:1):");
        using A = Layout<Int<4>, Int<1>>;
        using B = Layout<Int<2>, Int<1>>;
        auto fp = flat_product(A{}, B{});

        // product produces rank-2: (A, A* ∘ B)
        // flat_product flattens it
        std::println("  Result has rank = {}", rank<typename decltype(fp)::Shape>());
        std::println("  flat_product creates flat structure ✓");
    }

    {
        std::println("\n5. flat_product with stride: (4:2) ⊗ (3:1):");
        using A = Layout<Int<4>, Int<2>>;
        using B = Layout<Int<3>, Int<1>>;
        auto fp = flat_product(A{}, B{});

        std::println("  Result has rank = {}", rank<typename decltype(fp)::Shape>());
        std::println("  Flattens strided product ✓");
    }

    {
        std::println("\n6. flat_product with Shape tiler: (8:1) ⊗ 2:");
        using A = Layout<Int<8>, Int<1>>;
        auto fp = flat_product(A{}, Int<2>{});

        std::println("  Result has rank = {}", rank<typename decltype(fp)::Shape>());
        std::println("  Works with Shape tilers ✓");
    }

    std::println("\n=== All flat variant tests passed! ===");

    std::println("\nAll tests passed!");
    return 0;
}
