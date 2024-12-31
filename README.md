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


Here are the graph and lines of fit for outer_radii ratio versus major / minor radii, and inclination:

![multi-radius_plot](/fig_plots/per_outer_radii.png "Multi-Radius Plot")

Torus Outer Radii => Ellipse Major Radii: 
0.9965819026351573 + 0.050167827790387184 x**1 - 0.06757418306645466 x**2 + 0.24114481818282416 x**3
R value: 0.9909209156667388

Torus Outer Radii => Orbit Inclination:
0.029019297543903366 + 0.6975263101918348 x**1 + 0.7173987741769189 x**2
R value: 0.9838486953612806