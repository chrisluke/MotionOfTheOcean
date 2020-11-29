# Motion Of The Ocean - Deep Learning Project 

Use reinforcement learning to train a simulated humanoid to imitate a variety of motion skills from motioncapture data.


## Dependencies
``pip3 install gast==0.2.2``

``pip3 install pybullet``

``pip install tensorflow==1.14``

``pip install pandas``

``OpenGL >= 3.2``

``pip install mpi4py`` (NOTE: You must install ``mpich`` before installing mpi4py. On Macs, you can achive this by running: 
```
brew install mpich
```

## Training Models:
To train a policy, use `mpi_run.py` by specifying an argument file and the number of worker processes.
For example,
```
python3 mpi_run.py --arg_file train_humanoid3d_walk_args.txt --num_workers 16
```
will train a policy to perform a spinkick using 4 workers. As training progresses, it will regularly
print out statistics and log them to `output/` along with a `.ckpt` of the latest policy.
It typically takes about 60 millions samples to train one policy, which can take a day
when training with 16 workers. 16 workers is likely the max number of workers that the
framework can support, and it can get overwhelmed if too many workers are used.

A number of argument files are already provided in `args/` for the different skills. 
`train_[something]_args.txt` files are setup for `mpi_run.py` to train a policy, and 
`run_[something]_args.txt` files are setup for `DeepMimic.py` to run one of the pretrained policies.
To run your own policies, take one of the `run_[something]_args.txt` files and specify
the policy you want to run with `--model_file`. Make sure that the reference motion `--motion_file`
corresponds to the motion that your policy was trained for, otherwise the policy will not run properly.

## Running Pre-Trained Models:
Once you have installed the dependencies, you can run a bunch of pre-trained models. For example, you 
can run a pre-trained humanoid model that jumps by using the command: 

```
python3 testrl.py --arg_file run_humanoid3d_walk_args.txt
```

Note: In order for the library to work properly, you need to make sure you are using Tensorflow 1.14.

## Running Visualizer on Our Own Models:
Copy the model from the output folder to ``data/policies/humanoid3d``. 

Rename the file you just added to match that of the pre-trained model (such as ``humanoid3d_walk.ckpt.index`` 
and ``humanoid3d_walk.ckpt.data-00000-of-00001``). Now you can run the visualizer with the same command as you 
would have used for the pre-trained models:

``` 
python3 testrl.py --arg_file run_humanoid3d_walk_args.txt
```


## Interface
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
