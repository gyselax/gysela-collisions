help ([[
This modulefile defines the system paths and environment
variables needed to use the ROCm Toolkit. The ROCm modulefile
enables ROC Profiler, ROC Tracer, HIP, and ROCr.

===================================================================
To see AMD release information,  visit:
https://rocmdocs.amd.com/en/latest
===================================================================
]])
--------------------------------------------------------------------------------
-- Template variables
--------------------------------------------------------------------------------

local MOD_LEVEL = "6.3.3"
local ROCM_PATH = "/opt/software/rocm/6.3.3"

--------------------------------------------------------------------------------
-- Derived variables
--------------------------------------------------------------------------------

setenv("ROCM_PATH", ROCM_PATH)

prepend_path("PATH", ROCM_PATH .. "/bin")
prepend_path("PATH", ROCM_PATH .. "/lib/llvm/bin")

prepend_path("LD_LIBRARY_PATH", ROCM_PATH .. "/lib")
prepend_path("LD_LIBRARY_PATH", ROCM_PATH .. "/lib/rocprofiler")
prepend_path("LD_LIBRARY_PATH", ROCM_PATH .. "/lib/roctracer")

prepend_path("CMAKE_PREFIX_PATH", ROCM_PATH)
-- prepend_path("CMAKE_PREFIX_PATH", ROCM_PATH .. "/lib/cmake/hip")

prepend_path("MANPATH", ROCM_PATH .. "/share/man")
prepend_path("MANPATH", ROCM_PATH .. "/lib/llvm/share/man1")

-- Dubious:
-- setenv("HIP_PATH", ROCM_PATH)
-- local AMD_GCN           = ROCM_PATH .. "/amdgcn/bitcode"
-- setenv("DEVICE_LIB_PATH", AMD_GCN)
-- setenv("HIP_DEVICE_LIB_PATH", AMD_GCN)
