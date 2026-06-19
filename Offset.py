import numpy as np

def offset_curve(x, y, coeffs, d, side='left'):
        """
        Parameters
        ----------
        coeffs : array-like *** decreasing order of power ***
            coefficients of y-x graph
        y, x : array-like
            Coordinates of the original curve (ordered points).
        d : float
            Offset distance. side='left' offsets to the left of travel direction.
        side : str
            'left' or 'right' relative to curve direction.

        Returns
        -------
        xo, yo : ndarray
            Coordinates of the offset curve, same length as the input.
        """

        coeffs_d = []
        for i in range(len(coeffs)-1):
            coeffs_d.append(coeffs[i]*(len(coeffs)-1-i))
        derivative_coeffs = np.array(coeffs_d)
        
        dxdy = np.polyval(derivative_coeffs, y)  # dx/dy
        norm = np.sqrt(1.0 + dxdy**2)

        # left unit normal: rotate tangent (dxdy, 1) by +90° → (-1, dxdy)
        nx = -1.0 / norm
        ny = dxdy / norm

        sign = 1 if side == 'left' else -1
        x0 = x + sign * d * nx
        y0 = y + sign * d * ny
        
        return x0, y0