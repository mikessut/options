from options.random_walk import SVJ


def plot_svj_walks():
    
    und_price = 1
    v = .04
    v_avg = .04
    T = 1
    rho = -.64
    eta = .39
    #eta = .2
    lam = 1.15
    alpha = -.1151
    delta = .0967
    lamJ = .1308

    svj = SVJ(und_price, lam, eta, rho, v_avg, v, lamJ, alpha, delta, .1, 1/365)
    svj.nsteps(10*365)
    ax = svj.plot()
    print(svj._num_jumps)

    for _ in range(10):
        svj = SVJ(und_price, lam, eta, rho, v_avg, v, lamJ, alpha, delta, .1, 1/365)
        svj.nsteps(10*365)
        svj.plot(ax)
        print(svj._num_jumps)

