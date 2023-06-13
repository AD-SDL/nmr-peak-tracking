"""Utilities associated with TrackPy"""
from collections import deque

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from scipy.stats import linregress
from trackpy.predict import NullPredict
from trackpy.linking.utils import Point


class PeakMovementPredict(NullPredict):
    """Predict the velocity using a polynomial function that has a slope of 0 at t=0

    We choose this model form knowing that the time spacing between frames is logarithmic.
    The spacing between the first frames is, therefore, very small such that the apparent
    peak velocity at low times is zero.

    We fit the fully polynomial with up to ``max_history`` number of points.
    We use a linear model when there are fewer than ``min_history`` number of points in the history.

    Args:
        order: Order of the fitting polynomial
        max_history: Maximum number of previous frames to include in the fitting
        min_history: Minimum number of previous frames before fitting the full polynomial
    """

    def __init__(self, order: int = 3, max_history: int | None = None, min_history: int | None = None):
        super().__init__()
        if order <= 1:
            raise ValueError('Order must be at least 2')
        self.order = order
        self.max_history = max_history
        self.min_history = min_history
        self.histories: list[deque[tuple[int, float]]] = []

    def fit_predictor(self, frame: np.ndarray, offset: np.ndarray) -> Polynomial:
        """Fit a polynomial predictor with a linear term of zero

        Args:
            frame: Frame ID for each observation
            offset: NMR offset at each observation
        """
        # Truncate as needed
        if self.max_history is not None:
            ids = np.argsort(frame)[:self.max_history]
            frame = np.array(frame)[ids]
            offset = np.array(offset)[ids]

        # If too few points, make a linear model
        if self.min_history is not None and len(frame) < self.min_history:
            result = linregress(frame, offset)
            return Polynomial([result.intercept, result.slope])

        # Make the fitting matrix
        w = np.zeros((len(frame), self.order))
        w[:, 0] = 1
        w[:, 1] = frame * frame
        for i in range(2, self.order):
            w[:, i] = w[:, i - 1] * frame

        # Compute the LSQ fit
        coef = np.linalg.lstsq(w, offset, rcond=None)[0]

        # Insert a coefficient for x^1
        coef = np.insert(coef, 1, 0)
        return Polynomial(coef)

    def observe(self, frame: pd.DataFrame):
        # Update the position histories for each projectile in the frame
        frame = frame.sort_values('particle').set_index('particle')
        for particle_id, data in frame.iterrows():
            if particle_id >= len(self.histories):
                self.histories.append(deque(maxlen=self.max_history))
            self.histories[particle_id].append((data['frame'], data['x']))

    def predict(self, t1, particles: list[Point]):
        output = []
        for point in particles:
            my_history = self.histories[point.track.id]
            if len(my_history) < self.order + 2:
                output.append(point.pos)
            else:
                frame, offset = zip(*self.histories[point.track.id])
                poly = self.fit_predictor(np.array(frame), np.array(offset))
                output.append([point.pos[0], poly(t1)])
        return np.array(output)

