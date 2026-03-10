class EMAFilter:
    """
    Exponential Moving Average filter for temporal smoothing.

    The filter is defined as:

        s_t = alpha * x_t + (1 - alpha) * s_{t-1}

    where:
        x_t : current input value
        s_t : current smoothed value
        s_{t-1} : previous smoothed value
        alpha : smoothing factor (0 < alpha ≤ 1)

    A larger alpha makes the filter more responsive to new values,
    while a smaller alpha produces smoother but slower responses.
    """

    def __init__(self, alpha=0.3):
        """
        Initialize the EMA filter.

        Parameters
        ----------
        alpha : float, optional
            Smoothing factor controlling the influence of the
            current input value. Default is 0.3.
        """
        self.alpha = alpha
        self.value = None

    def update(self, x: float) -> float:
        """
        Update the filter with a new observation.

        Parameters
        ----------
        x : float
            New input value.

        Returns
        -------
        float
            Smoothed output value.
        """
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha*x + (1-self.alpha)*self.value
        return self.value