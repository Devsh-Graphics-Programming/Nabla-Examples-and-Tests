@echo off
REM Nsight Graphics live-edit wrapper: recompiles the ORIGINAL beauty.hlsl from disk
REM so edits to the real source files (and their includes) are picked up on recompile.
REM Nsight invokes this as: nsight_recompile_beauty.cmd "<output.spv>" "<ignored temp.hlsl>" [-I"<ignored>"]
REM Only %1 (the output path) is used; the captured temp and any appended -I are ignored.
"F:\Nablas\Nabla\tools\nsc\bin\nsc_rwdi.exe" ^
  -Fo %1 ^
  -no-nbl-builtins -spirv -Wno-c++11-extensions  -Wno-c++1z-extensions -Wno-c++14-extensions -Wno-gnu-static-float-init  -Wno-local-type-template-args ^
  -isystem F:\Nablas\Nabla\include ^
  -isystem F:\Nablas\Nabla\3rdparty\dxc\dxc\external\SPIRV-Headers\include ^
  -isystem F:\Nablas\Nabla\3rdparty\boost\superproject\libs\preprocessor\include ^
  -isystem F:\Nablas\Nabla\build\dynamic\src\nbl\device\include ^
  -I F:\Nablas\Nabla\examples_tests\common\include ^
  -I F:\Nablas\Nabla\examples_tests\40_PathTracer\include ^
  -I F:\Nablas\Nabla\examples_tests\40_PathTracer\app_resources\pathtrace ^
  -T lib_6_9 -E main -DNBL_USE_SER=1 -enable-16bit-types -fvk-use-scalar-layout -Zpr -O3 -HV 202x ^
  -fspv-target-env=vulkan1.3 -Zi -fspv-debug=file -fspv-debug=source -fspv-debug=line -fspv-debug=tool ^
  F:\Nablas\Nabla\examples_tests\40_PathTracer\app_resources\pathtrace\beauty.hlsl
