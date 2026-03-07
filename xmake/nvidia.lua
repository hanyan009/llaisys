
-- Define a helper to handle cuda linking issues
function link_cuda_runtime(target)
    -- Add cudart dependency
    target:add("links", "cudart")
    
    -- When building a shared library that links static libraries with CUDA code,
    -- we need to handle device code linking carefully.
    -- However, simply linking cudart is often enough if we are not using separate compilation (-rdc=true).
    -- If we turn off -rdc=true, we don't need to link cudadevrt and don't need device linking step.
end

target("llaisys-device-nvidia")
    set_kind("static")
    add_rules("mode.debug", "mode.release")
    add_rules("cuda")
    set_languages("cxx17")
    set_warnings("all", "error")
    
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        -- IMPORTANT: Disable RDC (Relocatable Device Code) to avoid complex device linking issues
        -- when linking this static lib into a shared lib later.
        -- If we needed RDC, we would have to use `set_policy("build.cuda.devlink", true)` on the final shared lib.
        add_cuflags("-Xcompiler -fPIC") 
        add_values("cuda.rdc", false)
    end

    add_links("cudart", "cublas", "cudnn")
    add_files("../src/device/nvidia/*.cu")
    add_files("../src/device/nvidia/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    add_rules("mode.debug", "mode.release")
    add_rules("cuda")
    set_languages("cxx17")
    set_warnings("all", "error")
    
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler -fPIC")
        add_values("cuda.rdc", false)
    end

    add_links("cudart", "cublas", "cudnn")
    add_files("../src/ops/*/nvidia/*.cu")
    add_files("../src/ops/*/nvidia/*.cpp")

    on_install(function (target) end)
target_end()
