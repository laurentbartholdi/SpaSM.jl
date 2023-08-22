using Libdl

"""
run(`tar xfz $(@__DIR__)/$nautyver.tar.gz -C $(@__DIR__)`)
cd(()->run(`./configure --enable-tls`), nautydir)
cd(()->run(`make nauty.a CCOBJ='${CC} -fPIC -c ${CFLAGS} -o $@'`), nautydir)
"""
