module;
#include <cstddef>
#include <type_traits>
#include <utility>
#include <algorithm>
export module layoutalg;

export template<auto V>
struct Int {
    static constexpr auto value = V;
    constexpr operator decltype(V)() const { return V; }
};

template<typename T>
concept IntType = requires {
    { T::value } -> std::convertible_to<size_t>;
};

export template<typename... Ts>
struct Tuple {};

template<typename T>
struct is_tuple : std::false_type {};
template<typename... Ts>
struct is_tuple<Tuple<Ts...>> : std::true_type {};

template<typename T>
concept TupleType = is_tuple<T>::value;

template<typename T>
concept IntTupleType = IntType<T> || TupleType<T>;

template<size_t Idx, typename T>
struct get_impl;

template<typename T, typename... Ts>
struct get_impl<0, Tuple<T, Ts...>> {
    using type = T;
};

template<size_t Idx, typename T, typename... Ts>
struct get_impl<Idx, Tuple<T, Ts...>> {
    using type = typename get_impl<Idx - 1, Tuple<Ts...>>::type;
};

export template<size_t Idx, TupleType T>
using get = typename get_impl<Idx, T>::type;

export template<IntTupleType T>
consteval size_t size() {
    if constexpr (IntType<T>) {
        return T::value;
    } else {
        return []<typename... Ts>(Tuple<Ts...>) {
            return (size<Ts>() * ... * 1);
        }(T{});
    }
}

export template<IntTupleType T>
consteval size_t rank() {
    if constexpr (IntType<T>) {
        return 1;
    } else {
        return []<typename... Ts>(Tuple<Ts...>) {
            return sizeof...(Ts);
        }(T{});
    }
}

export template<IntTupleType T>
consteval size_t depth() {
    if constexpr (IntType<T>) {
        return 0;
    } else {
        return []<typename... Ts>(Tuple<Ts...>) {
            return 1 + std::max({depth<Ts>()...});
        }(T{});
    }
}

template<IntTupleType S, IntTupleType D>
constexpr size_t layout_impl(size_t i) {
    if constexpr (IntType<S> && IntType<D>) {
        return (i % S::value) * D::value;
    } else {
        return []<typename... Ss, typename... Ds>(Tuple<Ss...>, Tuple<Ds...>, size_t idx) -> size_t {
            static_assert(sizeof...(Ss) == sizeof...(Ds));

            size_t result = 0;
            size_t remaining = idx;

            [&]<size_t... Is>(std::index_sequence<Is...>) {
                ((
                    [&] {
                        constexpr size_t si_size = size<get<Is, Tuple<Ss...>>>();
                        size_t local_idx = remaining % si_size;
                        remaining /= si_size;
                        result += layout_impl<get<Is, Tuple<Ss...>>, get<Is, Tuple<Ds...>>>(local_idx);
                    }()
                ), ...);
            }(std::make_index_sequence<sizeof...(Ss)>{});

            return result;
        }(S{}, D{}, i);
    }
}

export template<IntTupleType S, IntTupleType D>
struct Layout {
    using Shape = S;
    using Stride = D;

    static constexpr auto shape = S{};
    static constexpr auto stride = D{};

    static consteval size_t size_v() { return size<S>(); }

    static constexpr size_t operator()(size_t i) {
        return layout_impl<S, D>(i);
    }

    static constexpr size_t cosize() {
        return Layout{}(size_v() - 1) + 1;
    }
};

// Layout concept: detect if T is a Layout type
template<typename T>
struct is_layout : std::false_type {};
template<typename S, typename D>
struct is_layout<Layout<S, D>> : std::true_type {};

template<typename T>
concept LayoutType = is_layout<T>::value;


export constexpr auto example_simple() {
    using L = Layout<Int<16>, Int<1>>;
    static_assert(L::size_v() == 16);
    static_assert(L{}(0) == 0);
    static_assert(L{}(5) == 5);
    static_assert(L{}(15) == 15);
    return L{};
}

export constexpr auto example_row_major() {
    using L = Layout<Tuple<Int<8>, Int<4>>, Tuple<Int<1>, Int<8>>>;
    static_assert(L::size_v() == 32);
    static_assert(L{}(0) == 0);
    static_assert(L{}(1) == 1);
    static_assert(L{}(8) == 8);
    return L{};
}

export constexpr auto example_col_major() {
    using L = Layout<Tuple<Int<4>, Int<8>>, Tuple<Int<1>, Int<4>>>;
    static_assert(L::size_v() == 32);
    static_assert(L{}(0) == 0);
    static_assert(L{}(1) == 1);
    static_assert(L{}(4) == 4);
    return L{};
}

// Coalesce: s₀:d₀ ++ s₁:d₁ → simplified layout
// Merges adjacent modes when possible, preserving L(i) for all i
// Postcondition: size(c(L)) = size(L) ∧ ∀i, c(L)(i) = L(i)

export template<auto S, auto D>
consteval auto coalesce(Layout<Int<S>, Int<D>>) {
    // Single mode: already coalesced
    return Layout<Int<S>, Int<D>>{};
}

export template<auto S0, auto S1, auto D0, auto D1>
consteval auto coalesce(Layout<Tuple<Int<S0>, Int<S1>>, Tuple<Int<D0>, Int<D1>>>) {
    // Binary coalesce rules (left-to-right):

    if constexpr (S1 == 1) {
        // Rule: s₀:d₀ ++ 1:d₁ → s₀:d₀
        // Second mode is singleton, contributes nothing
        return Layout<Int<S0>, Int<D0>>{};
    } else if constexpr (S0 == 1) {
        // Rule: 1:d₀ ++ s₁:d₁ → s₁:d₁
        // First mode is singleton, contributes nothing
        return Layout<Int<S1>, Int<D1>>{};
    } else if constexpr (D1 == S0 * D0) {
        // Rule: s₀:d₀ ++ s₁:d₁ → (s₀·s₁):d₀  when d₁ = s₀·d₀
        // Modes are consecutive in memory, merge them
        // Example: (4:1) ++ (2:4) = 8:1  (contiguous 1D array)
        return Layout<Int<S0 * S1>, Int<D0>>{};
    } else {
        // Rule: s₀:d₀ ++ s₁:d₁ → (s₀,s₁):(d₀,d₁)  otherwise
        // Modes cannot merge (e.g., strided, non-contiguous)
        return Layout<Tuple<Int<S0>, Int<S1>>, Tuple<Int<D0>, Int<D1>>>{};
    }
}

// Three-mode coalesce helpers: handle left-to-right reduction
// After coalescing (s₀:d₀) ++ (s₁:d₁), merge result with (s₂:d₂)

template<auto S01, auto D01, auto S2, auto D2>
consteval auto coalesce_helper_3(Layout<Int<S01>, Int<D01>>, Int<S2>, Int<D2>) {
    // Case 1: First pair fully merged to s₀₁:d₀₁
    // Continue: coalesce(s₀₁:d₀₁, s₂:d₂)
    return coalesce(Layout<Tuple<Int<S01>, Int<S2>>, Tuple<Int<D01>, Int<D2>>>{});
}

template<auto S0, auto S1, auto D0, auto D1, auto S2, auto D2>
consteval auto coalesce_helper_3(Layout<Tuple<Int<S0>, Int<S1>>, Tuple<Int<D0>, Int<D1>>>, Int<S2>, Int<D2>) {
    // Case 2: First pair stayed as (s₀:d₀, s₁:d₁)
    // Try coalescing the rightmost pair: (s₁:d₁) ++ (s₂:d₂)
    auto second = coalesce(Layout<Tuple<Int<S1>, Int<S2>>, Tuple<Int<D1>, Int<D2>>>{});
    using Second = decltype(second);

    if constexpr (IntType<typename Second::Shape>) {
        // Second pair merged: Result is (s₀:d₀, s₁₂:d₁₂) where s₁₂ = s₁ merged with s₂
        constexpr auto s12 = Second::Shape::value;
        constexpr auto d12 = Second::Stride::value;
        return Layout<Tuple<Int<S0>, Int<s12>>, Tuple<Int<D0>, Int<d12>>>{};
    } else {
        // No further coalescing possible: Result is (s₀:d₀, s₁:d₁, s₂:d₂)
        constexpr auto s1_final = get<0, typename Second::Shape>::value;
        constexpr auto s2_final = get<1, typename Second::Shape>::value;
        constexpr auto d1_final = get<0, typename Second::Stride>::value;
        constexpr auto d2_final = get<1, typename Second::Stride>::value;
        return Layout<Tuple<Int<S0>, Int<s1_final>, Int<s2_final>>,
                     Tuple<Int<D0>, Int<d1_final>, Int<d2_final>>>{};
    }
}

export template<auto S0, auto S1, auto S2, auto D0, auto D1, auto D2>
consteval auto coalesce(Layout<Tuple<Int<S0>, Int<S1>, Int<S2>>, Tuple<Int<D0>, Int<D1>, Int<D2>>>) {
    // Three-mode coalesce: apply binary coalesce left-to-right
    // (s₀:d₀, s₁:d₁, s₂:d₂) → coalesce(coalesce(s₀:d₀, s₁:d₁), s₂:d₂)
    auto first = coalesce(Layout<Tuple<Int<S0>, Int<S1>>, Tuple<Int<D0>, Int<D1>>>{});
    return coalesce_helper_3(first, Int<S2>{}, Int<D2>{});
}

// Helper: Find minimum value in a (possibly nested) IntTuple
template<IntTupleType D>
consteval auto flatten_min() {
    if constexpr (IntType<D>) {
        return D::value;
    } else {
        return []<typename... Ds>(Tuple<Ds...>) {
            return std::min({flatten_min<Ds>()...});
        }(D{});
    }
}

// Helper: Scale all strides by a factor (for compose divide step)
template<IntTupleType D, auto factor>
consteval auto scale_strides() {
    if constexpr (IntType<D>) {
        return Int<D::value * factor>{};
    } else {
        return []<typename... Ds>(Tuple<Ds...>) {
            return Tuple<decltype(scale_strides<Ds, factor>())...>{};
        }(D{});
    }
}

// ========================================
// Tiler Concept (Section: Tiler Concept)
// ========================================
// A Tiler T can be:
//   - Layout L (used as-is)
//   - Shape S (interpreted as S:1̄)
//   - Tuple of Tilers (for by-mode application, handled at operation level)

// Helper: Create all-ones stride matching shape structure
// Converts Shape S to stride 1̄ with same structure
template<IntTupleType S>
consteval auto make_ones_stride() {
    if constexpr (IntType<S>) {
        return Int<1>{};
    } else {
        return []<typename... Ss>(Tuple<Ss...>) {
            return Tuple<decltype(make_ones_stride<Ss>())...>{};
        }(S{});
    }
}

// Tiler interpretation: τ(T) → Layout
// If T is already a Layout, return it
// If T is a Shape (IntTuple), interpret as S:1̄
export template<typename T>
consteval auto interpret_tiler(T t) {
    if constexpr (LayoutType<T>) {
        // Already a Layout, return as-is
        return t;
    } else if constexpr (IntTupleType<T>) {
        // Shape → Layout with all-ones stride
        using Ones = decltype(make_ones_stride<T>());
        return Layout<T, Ones>{};
    }
}

// Helper: Prepend a type to a tuple (for flattening in truncate_shape)
template<typename T, typename U>
struct prepend_helper;

template<typename T, typename U>
requires IntType<U>
struct prepend_helper<T, U> {
    using type = Tuple<T, U>;
};

template<typename T, typename... Us>
struct prepend_helper<T, Tuple<Us...>> {
    using type = Tuple<T, Us...>;
};

template<typename T, typename U>
using prepend = typename prepend_helper<T, U>::type;

// Helper: Truncate shape to first n elements (for compose mod step)
template<auto S, auto n>
consteval auto truncate_shape(Int<S>, Int<n>) {
    constexpr decltype(n) result = n < S ? n : S;
    return Int<result>{};
}

template<typename S1, typename... Ss, auto n>
consteval auto truncate_shape(Tuple<S1, Ss...>, Int<n>) {
    constexpr auto s1 = size<S1>();
    if constexpr (n <= s1) {
        // All needed elements fit in first mode
        return Int<n>{};
    } else if constexpr (sizeof...(Ss) == 0) {
        // No more modes, return what we have
        return S1{};
    } else {
        // Need elements from subsequent modes
        constexpr decltype(n) next_target = (n + s1 - 1) / s1;  // ceil(n / s1)
        auto rest = truncate_shape(Tuple<Ss...>{}, Int<next_target>{});
        using Result = prepend<S1, decltype(rest)>;
        return Result{};
    }
}

// ========================================
// Composition (Section: Composition)
// ========================================
// (A ∘ B)(c) = A(B(c))
// Selects elements from A according to pattern B
// Example: (16:2) ∘ (4:1) = 4:2  (every other element, first 4)
// Postcondition: compatible(B, A ∘ B) ∧ ∀i < size(B), (A ∘ B)(i) = A(B(i))

export template<auto SA, auto DA, auto SB, auto DB>
consteval auto compose(Layout<Int<SA>, Int<DA>>, Layout<Int<SB>, Int<DB>>) {
    // Single-mode ∘ Single-mode: A(B(i)) = A(i * DB) = (i * DB) * DA = i * (DA * DB)
    // Result: stride multiplies (DA * DB)
    // Result size: limited by both SB and how many of B's outputs fit in A's domain
    //
    // B generates positions: 0*DB, 1*DB, 2*DB, ..., (SB-1)*DB
    // These must be valid indices for A (i.e., < SA)
    // Max valid i: i * DB < SA  →  i < SA/DB
    // Result size: min(SB, ceil(SA/DB))

    constexpr auto max_from_a = DB == 0 ? SB : (SA + DB - 1) / DB;  // ceil(SA/DB)
    constexpr auto result_size = max_from_a < SB ? max_from_a : SB;
    return Layout<Int<result_size>, Int<DA * DB>>{};
}

export template<typename SA, typename DA, auto SB, auto DB>
consteval auto compose(Layout<SA, DA> a, Layout<Int<SB>, Int<DB>> b) {
    if constexpr (IntType<SA>) {
        // Single-mode A, use scalar compose
        return compose(a, b);
    } else {
        // Multi-mode A ∘ Single-mode B
        // Strategy: Try to coalesce first for simplification
        auto a_coalesced = coalesce(a);
        using CoalescedShape = typename decltype(a_coalesced)::Shape;

        if constexpr (IntType<CoalescedShape>) {
            // Coalesced to scalar, use scalar compose
            return compose(a_coalesced, b);
        } else {
            // Still multi-mode after coalesce
            // Use algebraic approach: divide (scale strides) + mod (truncate shape)
            using CoalescedStride = typename decltype(a_coalesced)::Stride;

            // Step 1: Divide - scale all strides by DB
            using ScaledStrides = decltype(scale_strides<CoalescedStride, DB>());

            // Step 2: Mod - truncate shape to first SB elements
            auto new_shape = truncate_shape(CoalescedShape{}, Int<SB>{});

            return Layout<decltype(new_shape), ScaledStrides>{};
        }
    }
}

template<typename SA, typename DA, typename... SBs, typename... DBs, size_t... Is>
consteval auto compose_helper(Layout<SA, DA> a, Tuple<SBs...>, Tuple<DBs...>, std::index_sequence<Is...>) {
    return Layout<
        Tuple<typename decltype(compose(a, Layout<get<Is, Tuple<SBs...>>, get<Is, Tuple<DBs...>>>{}))::Shape...>,
        Tuple<typename decltype(compose(a, Layout<get<Is, Tuple<SBs...>>, get<Is, Tuple<DBs...>>>{}))::Stride...>
    >{};
}

export template<typename SA, typename DA, typename SB, typename DB>
consteval auto compose(Layout<SA, DA> a, Layout<SB, DB> b) {
    if constexpr (IntType<SB>) {
        // B is single-mode, handled above
        return compose(a, b);
    } else {
        // A ∘ Multi-mode B: apply left-distributivity
        // A ∘ (B₁,...,Bₙ) = (A ∘ B₁,...,A ∘ Bₙ)
        return []<typename... SBs, typename... DBs>(Layout<SA, DA> la, Tuple<SBs...> sb, Tuple<DBs...> db) {
            return compose_helper(la, sb, db, std::make_index_sequence<sizeof...(SBs)>{});
        }(a, SB{}, DB{});
    }
}

// Tiler-aware compose: accepts Shapes as tilers
export template<typename TA, typename TB>
consteval auto compose(TA a, TB b)
    requires (LayoutType<TA> || IntTupleType<TA>) && (LayoutType<TB> || IntTupleType<TB>)
          && (!(LayoutType<TA> && LayoutType<TB>))  // Avoid ambiguity with Layout overloads
{
    auto layout_a = interpret_tiler(a);
    auto layout_b = interpret_tiler(b);
    return compose(layout_a, layout_b);
}

// Complement: A* = complement(A, M)
// Finds tiling offsets such that (A, A*) covers [0, M) exactly once
// Postcondition: cosize((A, A*)) ≥ size(M)
// Example: complement(4:2, 24) = (2,3):(1,8)

// Helper: Check if a specific residue is covered by any A(i) % min_stride
template<IntTupleType S, IntTupleType D, auto min_stride, auto residue>
consteval bool is_residue_covered() {
    return []<size_t... Is>(std::index_sequence<Is...>) {
        return ((layout_impl<S, D>(Is) % min_stride == residue) || ...);
    }(std::make_index_sequence<size<S>()>{});
}

// Helper: Count how many residues in [0, min_stride) are NOT covered by A
template<IntTupleType S, IntTupleType D, auto min_stride>
consteval auto count_uncovered_residues() {
    return []<size_t... Rs>(std::index_sequence<Rs...>) {
        return static_cast<decltype(min_stride)>(((is_residue_covered<S, D, min_stride, Rs>() ? 0 : 1) + ...));
    }(std::make_index_sequence<min_stride>{});
}

// Complement: A* = complement(A, M)
// Unified implementation - flatten_min handles both scalar and multi-mode strides
export template<IntTupleType SA, IntTupleType DA, auto M>
consteval auto complement(Layout<SA, DA>, Int<M>) {
    // Algorithm from spec:
    // min_stride = min(flatten(D))
    // period = size(A) * min_stride
    // residues_covered = {A(i) % min_stride | i < size(A)}
    // residues_needed = [0, min_stride) \ residues_covered
    // inner_offsets = [0] ∪ residues_needed if residues_needed else [0]
    // num_periods = ⌈m / period⌉
    // return (len(inner_offsets), num_periods):(1, period)

    constexpr decltype(M) min_stride = flatten_min<DA>();
    constexpr decltype(M) layout_size = size<SA>();
    constexpr decltype(M) period = layout_size * min_stride;

    // Count uncovered residues
    constexpr decltype(M) uncovered = count_uncovered_residues<SA, DA, min_stride>();

    // Inner offsets size: 1 if all covered, else 1 + uncovered
    constexpr decltype(M) inner_size = uncovered == 0 ? decltype(M)(1) : decltype(M)(1) + uncovered;

    // Outer periods: ceil(M / period)
    constexpr decltype(M) num_periods = (M + period - 1) / period;

    return Layout<Tuple<Int<inner_size>, Int<num_periods>>, Tuple<Int<1>, Int<period>>>{};
}

// Tuple concatenation: creates rank-2 layout (A, B)
// Critical for Division and Product - preserves structure, does NOT flatten
export template<typename SA, typename DA, typename SB, typename DB>
consteval auto concat_layouts(Layout<SA, DA>, Layout<SB, DB>) {
    return Layout<Tuple<SA, SB>, Tuple<DA, DB>>{};
}

// Helper: Flatten two IntTuples into a single flat tuple
// Used for flat_divide and flat_product variants
template<typename S, typename T>
consteval auto flatten_two(S, T) {
    if constexpr (IntType<S> && IntType<T>) {
        // Both scalars: (Int, Int) → Tuple<Int, Int>
        return Tuple<S, T>{};
    } else if constexpr (IntType<S> && TupleType<T>) {
        // Scalar + Tuple: (Int, (T...)) → Tuple<Int, T...>
        return []<typename... Ts>(Tuple<Ts...>) {
            return Tuple<S, Ts...>{};
        }(T{});
    } else if constexpr (TupleType<S> && IntType<T>) {
        // Tuple + Scalar: ((S...), Int) → Tuple<S..., Int>
        return []<typename... Ss>(Tuple<Ss...>) {
            return Tuple<Ss..., T>{};
        }(S{});
    } else {
        // Both tuples: ((S...), (T...)) → Tuple<S..., T...>
        return []<typename... Ss, typename... Ts>(Tuple<Ss...>, Tuple<Ts...>) {
            return Tuple<Ss..., Ts...>{};
        }(S{}, T{});
    }
}

// ========================================
// Division (Section: Division)
// ========================================
// A ⊘ B := A ∘ (B, B*)
// Partitions A into (tile, rest) where tile = A ∘ B
// Postcondition: layout⟨0⟩(A ⊘ B) ≡ A ∘ B
export template<typename SA, typename DA, typename SB, typename DB>
consteval auto divide(Layout<SA, DA> a, Layout<SB, DB> b) {
    // Compute B* = complement(B, size(A))
    constexpr int size_a = size<SA>();
    auto b_star = complement(b, Int<size_a>{});

    // Create (B, B*) - this is the critical rank-2 structure
    auto b_concat = concat_layouts(b, b_star);

    // Compose: A ∘ (B, B*)
    return compose(a, b_concat);
}

// Tiler-aware divide: accepts Shapes as tilers
export template<typename TA, typename TB>
consteval auto divide(TA a, TB b)
    requires (LayoutType<TA> || IntTupleType<TA>) && (LayoutType<TB> || IntTupleType<TB>)
          && (!(LayoutType<TA> && LayoutType<TB>))
{
    auto layout_a = interpret_tiler(a);
    auto layout_b = interpret_tiler(b);
    return divide(layout_a, layout_b);
}

// ========================================
// Division Variants (Section: Division Variants)
// ========================================

// flat_divide: A ⊘ B → (T₀, T₁, R₀, R₁, ...) - fully flatten
// Takes the rank-2 result from divide and flattens all modes
export template<typename SA, typename DA, typename SB, typename DB>
consteval auto flat_divide(Layout<SA, DA> a, Layout<SB, DB> b) {
    auto div = divide(a, b);
    using DivShape = typename decltype(div)::Shape;
    using DivStride = typename decltype(div)::Stride;

    // Extract the two modes from rank-2 result
    using Shape0 = get<0, DivShape>;
    using Shape1 = get<1, DivShape>;
    using Stride0 = get<0, DivStride>;
    using Stride1 = get<1, DivStride>;

    // Flatten them into a single rank-n layout
    auto flat_shape = flatten_two(Shape0{}, Shape1{});
    auto flat_stride = flatten_two(Stride0{}, Stride1{});

    return Layout<decltype(flat_shape), decltype(flat_stride)>{};
}

// Tiler-aware flat_divide
export template<typename TA, typename TB>
consteval auto flat_divide(TA a, TB b)
    requires (LayoutType<TA> || IntTupleType<TA>) && (LayoutType<TB> || IntTupleType<TB>)
          && (!(LayoutType<TA> && LayoutType<TB>))
{
    auto layout_a = interpret_tiler(a);
    auto layout_b = interpret_tiler(b);
    return flat_divide(layout_a, layout_b);
}

// ========================================
// Product (Section: Product)
// ========================================
// A ⊗ B := (A, A* ∘ B)
// Replicates A according to pattern B
// Postcondition: compatible(A, layout⟨0⟩(A ⊗ B)) ∧ compatible(B, layout⟨1⟩(A ⊗ B))
export template<typename SA, typename DA, typename SB, typename DB>
consteval auto product(Layout<SA, DA> a, Layout<SB, DB> b) {
    // Compute A* = complement(A, size(A) * cosize(B))
    constexpr int size_a = size<SA>();
    constexpr int cosize_b = Layout<SB, DB>::cosize();
    constexpr int target_size = size_a * cosize_b;
    auto a_star = complement(a, Int<target_size>{});

    // Compose A* with B
    auto composed = compose(a_star, b);

    // Concatenate: (A, A* ∘ B)
    return concat_layouts(a, composed);
}

// Tiler-aware product: accepts Shapes as tilers
export template<typename TA, typename TB>
consteval auto product(TA a, TB b)
    requires (LayoutType<TA> || IntTupleType<TA>) && (LayoutType<TB> || IntTupleType<TB>)
          && (!(LayoutType<TA> && LayoutType<TB>))
{
    auto layout_a = interpret_tiler(a);
    auto layout_b = interpret_tiler(b);
    return product(layout_a, layout_b);
}

// ========================================
// Product Variants (Section: Product Variants)
// ========================================

// flat_product: A ⊗ B → (M, N, T₀, T₁, ...) - fully flatten
// Takes the rank-2 result from product and flattens all modes
export template<typename SA, typename DA, typename SB, typename DB>
consteval auto flat_product(Layout<SA, DA> a, Layout<SB, DB> b) {
    auto prod = product(a, b);
    using ProdShape = typename decltype(prod)::Shape;
    using ProdStride = typename decltype(prod)::Stride;

    // Extract the two modes from rank-2 result
    using Shape0 = get<0, ProdShape>;
    using Shape1 = get<1, ProdShape>;
    using Stride0 = get<0, ProdStride>;
    using Stride1 = get<1, ProdStride>;

    // Flatten them into a single rank-n layout
    auto flat_shape = flatten_two(Shape0{}, Shape1{});
    auto flat_stride = flatten_two(Stride0{}, Stride1{});

    return Layout<decltype(flat_shape), decltype(flat_stride)>{};
}

// Tiler-aware flat_product
export template<typename TA, typename TB>
consteval auto flat_product(TA a, TB b)
    requires (LayoutType<TA> || IntTupleType<TA>) && (LayoutType<TB> || IntTupleType<TB>)
          && (!(LayoutType<TA> && LayoutType<TB>))
{
    auto layout_a = interpret_tiler(a);
    auto layout_b = interpret_tiler(b);
    return flat_product(layout_a, layout_b);
}
