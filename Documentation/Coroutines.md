# C++ Coroutines: Technical Implementation Reference

Low-level implementation guide for building custom coroutine infrastructure from compiler primitives. C++20/23/26.

## Compiler Transformation

Functions with `co_await`/`co_yield`/`co_return` become three functions:

- **Ramp**: Allocates frame, constructs promise, calls `get_return_object()`, evaluates `initial_suspend()`, returns coroutine object
- **Resume**: State machine with switch on suspend-point index
- **Destroy**: Runs destructors, frees frame

**Frame layout**: Promise, parameter copies (never references), locals spanning suspensions, suspend index, temporaries.

**Critical**: Parameters copied to frame. References don't extend lifetimes → always pass by value.

## promise_type Interface

**Required methods**:

```cpp
ReturnType get_return_object()  // First call, returns coroutine object
auto initial_suspend() noexcept  // suspend_always (lazy, enables HALO) or suspend_never (eager)
auto final_suspend() noexcept    // Must be noexcept. Return suspend_always (standard)
void return_value(T) | void return_void()  // Exactly one required
void unhandled_exception()       // Typical: exception_ = std::current_exception()
```

**Optional**:

```cpp
auto yield_value(T)              // For generators, return suspend_always
auto await_transform(T)          // Intercepts all co_await
void* operator new(size_t sz)    // sz is total frame size
static ReturnType get_return_object_on_allocation_failure()  // Enables nothrow allocation
```

## Awaiter Protocol

**Awaitable → Awaiter**: `promise.await_transform(expr)` → `expr.operator co_await()` → else `expr` itself.

**Awaiter interface**:

```cpp
bool await_ready() const noexcept
```
Fast-path: `true` skips suspension.

```cpp
void | bool | coroutine_handle<> await_suspend(coroutine_handle<> h)
```
Called after state saved:
- **void**: Always suspend
- **bool**: `true`=suspend, `false`=resume immediately
- **coroutine_handle<>**: Symmetric transfer (preferred)—tail call to returned handle. Return `std::noop_coroutine()` to return to caller.

```cpp
T await_resume()
```
Produces co_await result. Can throw.

**Threading constraint**: After `await_suspend()` publishes handle, another thread can resume/destroy. Never access `this`/`promise` after scheduling—only locals are safe.

## Symmetric Transfer (Critical for Performance)

**Problem**: Pre-C++20 calling `handle.resume()` in `await_suspend()` caused stack growth → overflow in chains.

**Solution**: Return `coroutine_handle<>` from `await_suspend()`. Compiler guarantees tail call—zero stack growth for unlimited chaining.

```cpp
coroutine_handle<> await_suspend(coroutine_handle<> continuation) {
    promise_.continuation_ = continuation;
    return next_coro_;  // Tail call, no stack frame
}

// In final_suspend awaiter:
coroutine_handle<> await_suspend(coroutine_handle<promise_type> h) {
    return h.promise().continuation_;  // Chain to next
}
```

**Always use symmetric transfer for coroutine chains.** Eliminates stack overflow, enables O(1) chain depth.

## HALO: Heap Allocation eLision Optimization

**Most critical performance optimization**. When conditions met, compiler allocates frame on stack → zero allocation overhead.

**Requirements**:
1. Start suspended: `initial_suspend()` returns `suspend_always`
2. Handle doesn't escape caller scope
3. Return object is move-only
4. Ramp function + `get_return_object()` inline at call site
5. Frame size known at call site

**Measured impact**: Generators 0-2ns overhead (matches hand-written iterators). Without HALO: 10-50ns per call (heap allocation cost).

**Design for HALO**:
- Always start suspended
- Make coroutine types move-only
- Keep wrapper functions small and inline
- Don't escape handles
- Lightweight return objects

**Verification**: Check assembly for absence of `operator new` calls, or use `-Rpass=coro-elide` (Clang).

**Compiler support**: Clang (reliable), MSVC 19.49+ (Feb 2025), GCC (in progress).

## coroutine_handle<> Operations

```cpp
void resume() | operator()()  // Resume at last suspend point
void destroy()                 // Destroy frame. All copies invalidate.
bool done()                    // True if at final suspend. UB if not suspended.
Promise& promise()             // Access promise (typed handles only)
static from_promise(Promise&)  // Reconstruct handle from promise
```

**Not RAII**—like raw pointer. Wrap in RAII type. All copies reference same coroutine.

## Memory Management

**Default**: `promise_type::operator new` or `::operator new`. Frame size determined post-optimization.

**Custom allocation**:
```cpp
void* operator new(size_t sz, std::allocator_arg_t, Allocator& alloc, Args&...) {
    // sz is total frame size, not sizeof(promise_type)
    // Store allocator copy in frame for operator delete
}

void operator delete(void* ptr, size_t sz) {
    // Retrieve allocator from frame, deallocate
}
```

**Nothrow**: Provide `get_return_object_on_allocation_failure()` for nothrow allocation.

## Performance Optimization

**Compilation**:
- Heavy template use → significant compile time cost (20-30% increases reported)
- Use forward declarations, minimize includes
- Consider explicit template instantiation

**Runtime (with HALO)**:
- Generators: 0-2ns overhead
- Tasks: ~2ns composition overhead
- Symmetric transfer: zero stack cost

**Runtime (without HALO)**:
- Heap allocation: 10-50ns per coroutine call
- Depends entirely on allocator

**Debug builds**: 10-100x slower (no HALO, no inlining). Never benchmark debug. Use release with debug symbols.

**Compiler differences**:
- **MSVC**: Aggressive elision, good debugging, historical ICEs
- **Clang**: Best HALO, strong optimization, LLVM coroutine pass
- **GCC**: Conservative optimization, very standards-compliant

## Production Best Practices

**Design rules**:
1. **Always start suspended** (`suspend_always` in `initial_suspend()`)—enables HALO, prevents races, simplifies debugging
2. **Always use symmetric transfer** when chaining—return `coroutine_handle<>` from `await_suspend()`
3. **Make critical ops noexcept**—`initial_suspend()`, `final_suspend()`, awaiter methods
4. **Wrap handles in RAII**—coroutine_handle is not RAII, manually destroy

**Common pitfalls**:
- Dangling parameter references
- Forgetting to destroy coroutines (use RAII wrappers)
- Resuming after final_suspend (UB)
- Accessing awaiter/promise after scheduling (UAF race)

## Minimal Working Implementations

**Generator**:
```cpp
template<typename T>
class generator {
    struct promise_type {
        T* value_ptr = nullptr;
        
        generator get_return_object() {
            return generator{coroutine_handle<promise_type>::from_promise(*this)};
        }
        
        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        std::suspend_always yield_value(T& value) noexcept {
            value_ptr = std::addressof(value);
            return {};
        }
        void return_void() noexcept {}
        void unhandled_exception() { std::terminate(); }
    };
    
    struct iterator {
        coroutine_handle<promise_type> h;
        iterator& operator++() { h.resume(); return *this; }
        T& operator*() const { return *h.promise().value_ptr; }
        bool operator==(std::default_sentinel_t) const { return h.done(); }
    };
    
    iterator begin() { if (h) h.resume(); return {h}; }
    std::default_sentinel_t end() { return {}; }
    
    ~generator() { if (h) h.destroy(); }
    generator(generator&& o) : h(std::exchange(o.h, {})) {}
    generator(const generator&) = delete;
    
private:
    explicit generator(coroutine_handle<promise_type> h) : h(h) {}
    coroutine_handle<promise_type> h;
};
```

**Async Task with Symmetric Transfer**:
```cpp
template<typename T>
class task {
    struct promise_type {
        std::variant<std::monostate, T, std::exception_ptr> result;
        std::coroutine_handle<> continuation;
        
        task get_return_object() {
            return task{coroutine_handle<promise_type>::from_promise(*this)};
        }
        
        std::suspend_always initial_suspend() noexcept { return {}; }
        
        auto final_suspend() noexcept {
            struct awaiter {
                bool await_ready() noexcept { return false; }
                std::coroutine_handle<> await_suspend(
                    coroutine_handle<promise_type> h) noexcept {
                    if (h.promise().continuation)
                        return h.promise().continuation;
                    return std::noop_coroutine();
                }
                void await_resume() noexcept {}
            };
            return awaiter{};
        }
        
        void return_value(T value) {
            result.template emplace<1>(std::move(value));
        }
        void unhandled_exception() {
            result.template emplace<2>(std::current_exception());
        }
    };
    
    struct awaiter {
        coroutine_handle<promise_type> h;
        
        bool await_ready() { return h.done(); }
        std::coroutine_handle<> await_suspend(
            std::coroutine_handle<> cont) noexcept {
            h.promise().continuation = cont;
            return h;  // Symmetric transfer
        }
        T await_resume() {
            auto& r = h.promise().result;
            if (auto* ex = std::get_if<2>(&r))
                std::rethrow_exception(*ex);
            return std::get<1>(std::move(r));
        }
    };
    
    awaiter operator co_await() && { return {h}; }
    
    ~task() { if (h) h.destroy(); }
    task(task&& o) : h(std::exchange(o.h, {})) {}
    task(const task&) = delete;
    
private:
    explicit task(coroutine_handle<promise_type> h) : h(h) {}
    coroutine_handle<promise_type> h;
};
```

Key features: Lazy start, RAII handles, symmetric transfer in final_suspend and awaiter, exception handling, move-only for HALO.
