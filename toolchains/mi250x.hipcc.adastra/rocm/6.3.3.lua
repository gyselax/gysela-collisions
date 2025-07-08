--------------------------------------------------------------------------------
-- Template variables
--------------------------------------------------------------------------------

local ROCM_PATH = "/opt/software/rocm/6.3.3"

--------------------------------------------------------------------------------
-- Derived variables
--------------------------------------------------------------------------------

pushenv("ROCM_PATH", ROCM_PATH)

prepend_path("PATH", ROCM_PATH .. "/bin")
prepend_path("PATH", ROCM_PATH .. "/lib/llvm/bin")

prepend_path("LD_LIBRARY_PATH", ROCM_PATH .. "/lib")
-- -- NOTE: could conflict with other toolchains if mixing compilers.
-- prepend_path("LD_LIBRARY_PATH", ROCM_PATH .. "/lib/llvm/lib")
-- NOTE: some codes need that (vLLM)
-- prepend_path("PYTHONPATH", ROCM_PATH .. "/share/amd_smi")

prepend_path("CMAKE_PREFIX_PATH", ROCM_PATH)
-- NOTE: hipcc adds --hip-link. Tell clang not to read ${ROCM_PATH}/lib/llvm/bin/rocm.cfg, then the driver relies only
-- on the user flags and environment variables.
pushenv("HIPCC_COMPILE_FLAGS_APPEND", (os.getenv("HIPCC_COMPILE_FLAGS_APPEND") or "") .. " --no-default-config ")
