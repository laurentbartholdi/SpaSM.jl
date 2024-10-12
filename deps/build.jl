using Libdl

installdir = pwd()*"/usr"

cd("givaro-4.2.0") do
    run(`./configure --prefix=$installdir`)
    run(`make install`)
end

cd("fflas-ffpack") do
    run(`./autogen.sh PKG_CONFIG_PATH=$installdir/lib/pkgconfig --prefix=$installdir`)
    run(`make install`)
end

cd("spasm") do
    run(addenv(`cmake .`, "PKG_CONFIG_PATH" => "$installdir/lib/pkgconfig"))
    run(`make`)
end
