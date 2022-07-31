# Saturn-planetary-system-NBody-simlation
An N-body Astrodynamical simulation of the Saturn planetary system with a rouge planet encounter. 
Coded in Python and Pytorch, orbits are propagated using a Symplectic Leapfrog numerical integration scheme.
This project was originally started to investigate the effect of orbital reasonances between Saturn's major satellites and a collisioness model of the ring system, although it's much more fun to throw another Saturn sized rouge planet in and watch the chaos.
The initial position and velocity vectors are computed from Kepler's equations (the solution to the one body problem).
This simulation uses Newton's universal law of Gravitation, post Newtonian corrections should be negligable on this scale.
The major satellites of Saturn are modelled as massive particles and the ring system is modelled as 10000 massless particles, collision detection will delete particles that pass within R_saturn of Saturn or the other planet. 
If performance is poor I recommend reducing n_tracers which detemines how many particles are in the rings.
Feel free to download and play around with the code.
