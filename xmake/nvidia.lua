target("llaisys-device-nvidia")
    set_kind("static")
    add_rules("mode.debug", "mode.release")
    add_rules("cuda")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler -fPIC")
    end

    add_links("cudart")
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
    end

    add_links("cudart")
    add_files("../src/ops/*/nvidia/*.cu")
    add_files("../src/ops/*/nvidia/*.cpp")

    on_install(function (target) end)
target_end()
