# C++23 Style Guide - Expert Reference

## General

Prefer pre-reserving memory and filling buffers when writes where possible.

## Output
```cpp
std::println("Hello, {}!", name);              // default stdout
std::println(stderr, "Error: {}", msg);        // to stderr/FILE*
```

## Concepts
```cpp
template<typename T>
concept RawStorage = std::same_as<T, uint16_t> || std::same_as<T, uint8_t> || std::same_as<T, float>;

template<typename T>
concept NotBFloat16 = !std::same_as<T, bfloat16>;  // prevent accidents
```

# Type Punning
```cpp
uint32_t bits = std::bit_cast<uint32_t>(f);        // safe reinterpret, optimizes to zero
float restored = std::bit_cast<float>(bits);       // works in constexpr
```

# Multi-dimensional Arrays
```cpp
std::mdspan<float, std::extents<int, 16, 16>> matrix(data);
auto elem = matrix[i, j];  // natural syntax for matrices
```

# if consteval
```cpp
int compute() {
    if consteval {
        return compile_time_path();
    } else {
        return runtime_path();
    }
}
```

## Structured Code
```cpp
auto [iter, inserted] = map.insert({key, value});
if (auto [it, ok] = map.try_emplace(k, v); ok) { use(it->second); }
for (auto&& elem : container) { }  // forwarding reference
```

## Compile-Time
```cpp
constexpr int factorial(int n) { /* ... */ }       // compile or runtime
consteval int must_be_compile_time(int x) { /* ... */ }  // compile only
```

## Concurrency
```cpp
std::jthread worker([](std::stop_token st) {       // auto-joins, auto-stops
    while (!st.stop_requested()) { work(); }
});
```

### Atomics
```cpp
counter.fetch_add(1, std::memory_order_relaxed);   // counters, no sync
counter.load(std::memory_order_acquire);            // read barrier
counter.store(val, std::memory_order_release);      // write barrier
counter.store(val, std::memory_order_seq_cst);      // default, safest
```

## Strings
```cpp
void process(std::string_view str);                 // no allocation
std::string msg = std::format("Val: {}, Hex: {:#x}", 42, 255);
```

## RAII
```cpp
std::scoped_lock lock(mutex1, mutex2);              // deadlock-free
std::unique_lock lock(mutex);                       // manual control
```

## Parameters
```cpp
void f(const T& x);           // large, read-only
void f(T x);                  // sink (will move/copy anyway)
void f(std::span<const T>);   // array, read-only
void f(std::span<T>);         // array, mutable
T f();                        // return by value (elision)
std::optional<T> f();         // nullable
std::unique_ptr<T> f();       // transfer ownership
```

## Initialization
```cpp
std::vector<int> v{1, 2, 3};                       // brace-init for aggregates
Point p{.x = 10, .y = 20};                         // designated (C++20)
struct Config { int timeout{1000}; };              // default members
```

## Coroutines

### Basic Patterns
```cpp
generator<int> range(int start, int end) {
    for (int i = start; i < end; ++i) co_yield i;
}

task<std::string> read_file(std::string path) {
    auto handle = co_await get_file_handle(path);
    co_return co_await handle.read_all();
}

async_generator<T> produce() {
    while (true) co_yield co_await fetch();
}
for co_await (auto& item : produce()) { use(item); }
```

### Lifetime Footgun

Reference parameters outlive the caller:
```cpp
generator<char> explode(const std::string& s) {
    for (char ch : s) co_yield ch;
}

explode("hello"s);  // DANGER: temporary destroyed, s becomes dangling

std::string str = "hello";
for (char c : explode(str)) { }  // OK: str outlives iteration
```

Prevent footguns:
```cpp
generator<char> explode(std::string s);              // copy into frame
generator<char> explode(const std::string&&) = delete;  // block temporaries
```
