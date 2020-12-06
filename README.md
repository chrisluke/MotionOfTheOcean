# Motion Of The Ocean - Deep Learning Project 

Use reinforcement learning to train a simulated humanoid to imitate a variety of motion skills from motioncapture data.


## Dependencies
``pip3 install gast==0.3.3``

``pip3 install pybullet``

``pip install tensorflow==2.3.1``

``OpenGL >= 3.2``

``pip install mpi4py`` (NOTE: You must install ``mpich`` before installing mpi4py. On Macs, you can achive this by running: 
```
brew install mpich
```

## arg_files
A number of argument files are already provided in `args/` for the different skills. 
`train_[something]_args.txt` files are setup for `train_model.py` to train a policy and 
`run_[something]_args.txt` files are setup for `run_visualizer.py` to run the corresponding 
policy located in `Saved_Models/`. Make sure that the reference motion `--motion_file` 
corresponds to the motion that your policy was trained for, otherwise the policy will not run properly.

## Training Models
To train a policy, use `train_model.py` by specifying an argument file and the number of worker processes.
For example,
```
python3 train_model.py --arg_file train_humanoid3d_walk_args.txt --num_workers 4
```
will train a policy to walk using 4 workers. It typically takes about 60 millions samples 
to train one policy, which can take a day when training with 16 workers. 

*NOTE* This our code does not follow the paper exactly. It is based on this 
[Medium article.](https://towardsdatascience.com/proximal-policy-optimization-ppo-with-tensorflow-2-x-89c9430ecc26)

I believe one of the issues is where it calls 
```
prob = agentoo7.actor(np.array([state]))
probs.append(prob[0])
```
This is wrong because the actor is not actually returning probablities. It is returning an action, which is a Tensor of size 36. This size makes sense because the humanoid has 36 Action Parameters. 


## Running Visualizer on Our Models:
You can visualize how the model performs by running the `run_visualizer.py` file.
For example:

``` 
python3 run_visualizer.py --arg_file run_humanoid3d_walk_args.txt
```
will run the visualizer for the model in `Saved_Models/` that was trained against the walking motion capture data.


## Visualizer Interface 
- ctrl + click and drag will pan the camera
- left click and drag will apply a force on the character at a particular location
- scrollwheel will zoom in/out
- pressing space will pause/resume the simulation

## Motion capture Data
Motion capture data can be found in `data/motions/`.  The motion files follow the JSON format. The `"Loop"` field 
specifies whether or not the motion is cyclic.`"wrap"` specifies a cyclic motion that will wrap back to the start 
at the end, while `"none"` specifies an acyclic motion that will stop once it reaches the end of the motion. Each 
vector in the `"Frames"` list specifies a keyframe in the motion. Each frame has the following format:
```
[
	duration of frame in seconds (1D),
	root position (3D),
	root rotation (4D),
	chest rotation (4D),
	neck rotation (4D),
	right hip rotation (4D),
	right knee rotation (1D),
	right ankle rotation (4D),
	right shoulder rotation (4D),
	right elbow rotation (1D),
	left hip rotation (4D),
	left knee rotation (1D),
	left ankle rotation (4D),
	left shoulder rotation (4D),
	left elbow rotation (1D)
]
```

Positions are specified in meters, 3D rotations for spherical joints are specified as quaternions `(w, x, y ,z)`,
and 1D rotations for revolute joints (e.g. knees and elbows) are represented with a scalar rotation in radians. The root
positions and rotations are in world coordinates, but all other joint rotations are in the joint's local coordinates.
To use your own motion clip, convert it to a similar style JSON file.
