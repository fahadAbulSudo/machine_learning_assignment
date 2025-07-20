UCI problem statement link : https://archive.ics.uci.edu/ml/datasets/Daily+and+Sports+Activities



In this dataset we have many features:
The data looks like this:
19 activities (a) 
8 users (p) 
60 segments (s) 
5 units on torso (T), right arm (RA), left arm (LA), right leg (RL), left leg (LL) 
9 sensors on each unit (x,y,z accelerometers, x,y,z gyroscopes, x,y,z magnetometers) 

So to reduce it I extracted the features and clustered per activity depending on their features.
