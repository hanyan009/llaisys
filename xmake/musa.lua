rule("musa")
    set_extensions(".mu")
    on_build_file(function (target, sourcefile, opt)
        import("core.tool.compiler")
        local objectfile = target:objectfile(sourcefile)
        os.mkdir(path.directory(objectfile))

        local final_flags = {"-c"}
        
        -- Get compilation flags for CXX
        local compinst = compiler.load("cxx", {target = target})
        local compflags = compinst:compflags({sourcefile = sourcefile, target = target})
        for _, flag in ipairs(compflags) do
            table.insert(final_flags, flag)
        end
        
        if not table.contains(final_flags, "-fPIC") and not is_plat("windows") then
            table.insert(final_flags, "-fPIC")
        end

        table.insert(final_flags, "-o")
        table.insert(final_flags, objectfile)
        table.insert(final_flags, sourcefile)

        os.vrunv("mcc", final_flags)
        table.insert(target:objectfiles(), objectfile)
    end)
rule_end()

target("llaisys-device-musa")
    set_kind("static")
    add_rules("mode.debug", "mode.release")
    add_rules("musa")
    set_languages("cxx17")
    set_warnings("all", "error")
    
    add_includedirs("/usr/local/musa/include")
    
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_links("musart", "mublas", "mudnn")
    add_linkdirs("/usr/local/musa/lib")
    
    add_files("../src/device/musa/*.mu")
    add_files("../src/device/musa/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops-musa")
    set_kind("static")
    add_deps("llaisys-tensor")
    add_rules("mode.debug", "mode.release")
    add_rules("musa")
    set_languages("cxx17")
    set_warnings("all", "error")
    
    add_includedirs("/usr/local/musa/include")
    
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_links("musart", "mublas", "mudnn")
    add_linkdirs("/usr/local/musa/lib")
    
    add_files("../src/ops/*/musa/*.mu")
    add_files("../src/ops/*/musa/*.cpp")

    on_install(function (target) end)
target_end()
