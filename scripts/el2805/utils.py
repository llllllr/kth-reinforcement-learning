def decide_random(rng, probability):
    return rng.binomial(n=1, p=probability) == 1
