using Libdl

cd("spasm") do
    run(`autoreconf -i`)
    run(`./configure --enable-openmp`)
    cd("src") do
        run(`make`)
    end
end
