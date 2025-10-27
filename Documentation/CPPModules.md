## File Extension
Use `.cppm` for module files

Module:
```cpp
export module math_utils;

export int add(int a, int b) {
    return a + b;
}
```

Import:
**File**: `main.cpp`
```cpp
import math_utils;
```

# Using headers with module fragment
- **Creating**: Use `export module module_name;` at the top
- **Exports**: Only items marked with `export` are visible to importers
- **Traditional includes**: Can still use `#include` in modules via global module fragment:
  ```cpp
  module;
  #include <iostream>
  export module my_module;
  ```
