rule("musa")
    set_extensions(".mu")
    on_build_file(function (target, sourcefile, opt)
        import("core.tool.compiler")
        local cxx = target:tool("cxx")
        
        local objectfile = target:objectfile(sourcefile)
        os.mkdir(path.directory(objectfile))

        local flags = {"-c", "-fPIC", "-O3", "-std=c++17"}
        for _, inc in ipairs(target:get("includedirs") or {}) do
            table.insert(flags, "-I" .. inc)
        end
        
        -- Try to extract all compiler flags for C++
        local compinst = compiler.load("cxx", {target = target})
        local compflags = compinst:compflags({sourcefile = sourcefile, target = target})
        
        -- We can just pass compflags to mcc
        local final_flags = {"-c"}
        for _, flag in ipairs(compflags) do
            table.insert(final_flags, flag)
        end
        table.insert(final_flags, "-o")
        table.insert(final_flags, objectfile)
        table.insert(final_flags, sourcefile)

        os.vrunv("mcc", final_flags)
        table.insert(target:objectfiles(), objectfile)
    end)
rule_end()

target("test_musa")
    set_kind("binary")
    add_rules("musa")
    add_includedirs("my_inc")
    add_files("test.mu")
