def label_smooth(eps=0.1):
    return 1.0 - 0.5 * eps, 0.5 * eps