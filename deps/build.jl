using Libdl

curdir = pwd()

cd("givaro") do
    run(`./autogen.sh`)
    run(`make install prefix=$curdir/usr`)
end

cd("fflas-ffpack") do
    run(`./autogen.sh GIVARO_CFLAGS=-I../usr/include`)
    run(`make install prefix=$curdir/usr`)
end

cd("spasm") do
    run(`cmake .`)
    run(`make`)
end
