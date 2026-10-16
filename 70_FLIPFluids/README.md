https://github.com/user-attachments/assets/fc8bfac9-c8fb-49f1-87d6-251aae015945

## Running on Linux (X11)

```bash
cmake --build build/linux-clang-release -j4 --target 70_flipfluids
(cd examples_tests/70_FLIPFluids/bin && DISPLAY=:1 ./70_flipfluids)
```

XCB windows don't deliver mouse or keyboard input yet, so when no input device is connected the camera orbits the tank on its own and the water is dropped again every 12 seconds. Stop it with Ctrl+C.
