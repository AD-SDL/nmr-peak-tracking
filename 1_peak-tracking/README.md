# TrackPy

Try to use Trackpy's prediction features.
It uses a []

## Attempts

We tried a few built-in predictors: 

- `NearestVelocityPredict` assumes that the velocity of particles varies spatially. 
   Not quite like ours, given that two nearby particles can be headed in opposite directions.


## Next Steps:

A few ideas:

- Write a predictor where we propagate each particle forward based a polynomial fit to each particle's position.
