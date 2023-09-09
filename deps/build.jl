using Libdl

cd("..") do
    try
        run(`git subtree pull --prefix=deps/spasm -m "merging spasm" https://github.com/laurentbartholdi/spasm master`)
    catch
        @warn "I'm experiencing problems with `git subtree pull`; cross your fingers"
    end
end

cd("spasm") do
    run(`make`)
end
