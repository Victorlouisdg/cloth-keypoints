# Scripts for visualizing the folding primitives in Blender
Requires a recent version of Blender, e.g. 3.3.0, and our `airo-blender-toolkit` to be installed.

Also requires the `cloth_manipulation` package from this project to be installed. And the 

## Usage

For the default towel:
```
blender -P visualize_pull_primitive.py
```


For 16 random towels:
```
blender -P visualize_pull_primitive.py -- 16
```