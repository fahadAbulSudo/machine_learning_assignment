Description:
	Different product manufacturing companies or e-commerce firms who have to deliver the products at different locations have to solve the Vehicle Routing Problem (VRP) for optimizing the delivery. It is similar to the Traveling Salesman Problem.

These VRP problems have different type of constraints such as:
1.	Vehicle Routing Problem (VRP)
a.	This is simple VRP without any constraints which contains single depot and multiple delivery locations
2.	Vehicle Routing Problem Time Window (VRPTW)
a.	This type has time constraints where each location has a specific time window. Vehicles must visit the locations only between the specified time
3.	Capacitated Vehicle Routing Problem (CVRP)
a.	In this type, each vehicle has a certain load carrying capacity.
4.	Resource Constraints
a.	Here we have constraints with respect to the resources such as, 10 vehicles but only 3 drivers, or we have 4 vehicles but at a time only 2 vehicles can be loaded/unloaded at the depot.
5.	Pickups and Deliveries
a.	This type of problem has multiple pickup points rather than just a single depot.
6.	Penalty and Dropping visits
a.	Here while optimizing the route, we can drop certain locations and bare the penalty for dropping them.

Reference: https://developers.google.com/optimization/routing/vrp#example

Capacited_VRP is an example solved by taking the data (latitude & longitude) of 1000 locations
