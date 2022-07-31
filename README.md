# Saturn-planetary-system-NBody-simlation
An N-body Astrodynamical simulation of the Saturn planetary system with a rouge planet encounter. 
Coded in Python and Pytorch. Orbits are propagated using a Symplectic Leapfrog numerical integration scheme.
This project was origionally started to investigate the effect of orbital reasonances between Saturn's major satellites and a collisioness model of the ring system, although it's much more fun to throw a another Saturn sized rouge planet in and watch the chaos.
The major satellites of Saturn are modelled as massive particles and the ring system is modelled as 10000 massless tracers, collision detection will delete tracers or moons that pass within R_saturn of saturn or the other planet. 
If performance is poor I recommend reducing n_tracers which detemines how many particles are in the rings.
Feel free to download and play around with the code.
