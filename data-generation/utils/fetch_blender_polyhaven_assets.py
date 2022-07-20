import os
import subprocess

import airo_blender_toolkit as abt

output_folder = os.path.join(abt.assets_path(), "haven")

# blenderproc dir (make sure to change this if required)
home = os.path.expanduser("~")
bproc_download_script = os.path.join(home, "BlenderProc", "blenderproc", "scripts", "download_haven.py")
resolution = "1k"
types = "hdris textures"

command = f"python3 {bproc_download_script} {output_folder} --resolution {resolution} --types {types}"

subprocess.run([command], shell=True)


# Currently ignoring these textures because they are not downloaded correctly.
# https://github.com/DLR-RM/BlenderProc/blob/develop/tests/testHavenLoader.py#L9
textures_to_ignore = [
    "book_pattern",
    "church_bricks_02",
    "fabric_pattern_05",
    "fabric_pattern_07",
    "leather_red_02",
    "leather_red_03",
]  # noqa
