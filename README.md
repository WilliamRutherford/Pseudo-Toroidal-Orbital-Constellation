# Pseudo-Toroidal Orbital Constellation
A least-squares fitting of elliptical orbits to form a torus.
 
A torus can be deconstructed into [Villarceau circles](https://en.wikipedia.org/wiki/Villarceau_circles). However, since their centers deviate from the origin, they are not possible orbits.
Instead, we form an elliptical orbit with parameters for it's minor radius, major radius, and inclination, to get a close approximation that *is* feasible.
 
The toruses larger, major radius is taken to always be 1, with the minor radius of the exterior / cross-section being changed. 

Below are some images of the resulting orbital constellations. On the left is the orbits of multiple bodies. On the right is the cross-section of a true torus (red) and our approximate torus (blue).

Outer Radius of 0.1:
![small](/fig_plots/Toroidal_OR_0-1.png "Outer Radius 0.1")

Outer Radius of 0.3:
![medium](/fig_plots/Toroidal_OR_0-3.png "Outer Radius 0.3")

Outer Radius of 0.5:
![halfway](/fig_plots/Toroidal_OR_0-5.png "Outer Radius 0.5")

Outer Radius of 0.7:
![large](/fig_plots/Toroidal_OR_0-7.png "Outer Radius 0.7")
