def mags(band, teff='', logg='', bin=1):
    from scipy.io.idl import readsav
    from collections import Counter


    s = readsav(path + 'modelspeclowresdustywise.save')
    Fr, Wr = [i for i in s.modelspec['fsyn']], [i for i in s['wsyn']]
    Tr, Gr = [int(i) for i in s.modelspec['teff']], [round(i, 1) for i in s.modelspec['logg']]

    # The band to compute
    RSR_x, RSR_y, lambda_eff = get_filters(path)[band]

    # Option to specify an effective temperature value
    if teff:
        t = [i for i, x in enumerate(s.modelspec['teff']) if x == teff]
        if len(t) == 0:
            print
            "No such effective temperature! Please choose from 1400K to 4500K in 50K increments or leave blank to select all."
    else:
        t = range(len(s.modelspec['teff']))

    # Option to specify a surfave gravity value
    if logg:
        g = [i for i, x in enumerate(s.modelspec['logg']) if x == logg]
        if len(g) == 0:
            print
            "No such surface gravity! Please choose from 3.0 to 6.0 in 0.1 increments or leave blank to select all."
    else:
        g = range(len(s.modelspec['logg']))

    # Pulls out objects that fit criteria above
    obj = list((Counter(t) & Counter(g)).elements())
    F = [Fr[i][::bin] for i in obj]
    T = [Tr[i] for i in obj]
    G = [Gr[i] for i in obj]
    W = Wr[::bin]

    # Interpolate to find new filter y-values
    I = interp(W, RSR_x, RSR_y, left=0, right=0)

    # Convolve the interpolated flux with each filter (FxR = RxF)
    FxR = [convolution(i, I) for i in F]

    # Integral of RSR curve over all lambda
    R0 = trapz(I, x=W)

    # Integrate to find the spectral flux density per unit wavelength [ergs][s-1][cm-2] then divide by R0 to get [erg][s-1][cm-2][cm-1]
    F_lambda = [trapz(y, x=W) / R0 for y in FxR]

    # Calculate apparent magnitude of each spectrum in each filter band
    Mags = [round(-2.5 * log10(m / F_lambda_0(band)), 3) for m in F_lambda]

    result = sorted(zip(Mags, T, G, F, I, FxR), key=itemgetter(1, 2))
    result.insert(0, W)

    return result