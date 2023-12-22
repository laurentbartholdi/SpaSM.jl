update_spasm() = run(`git subtree pull --prefix=deps/spasm -m "merging spasm" https://github.com/laurentbartholdi/spasm master`)

update_givaro() = run(`git subtree pull --prefix=deps/givaro -m "merging givaro" https://github.com/linbox-team/givaro master`)

update_fflas_ffpack() = run(`git subtree pull --prefix=deps/fflas-ffpack -m "merging fflas-ffpack" https://github.com/linbox-team/fflas-ffpack master`)
