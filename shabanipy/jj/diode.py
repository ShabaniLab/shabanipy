"""Model of Ic(B//) from arxiv:2303.01902."""


def icp_model(x, imax, b, c, bstar):
    """Positive critical current, Ic+."""
    return imax * (1 - b * (1 + c * np.sign(x - bstar)) * (x - bstar) ** 2)


def icm_model(x, imax, b, c, bstar):
    """Negative critical current, |Ic-| (magnitude)."""
    return icp_model(-x, imax, b, c, bstar)
