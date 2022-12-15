def random_decide(rng, probability):
    return rng.binomial(n=1, p=probability) == 1
