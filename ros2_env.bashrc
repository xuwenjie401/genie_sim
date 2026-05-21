export ROS_DISTRO=humble # or humble
export ROS_CMD_DISTRO=${ROS_DISTRO}
export CONDA_SITE_PACKAGES="/home/agxi/miniconda3/envs/issac/lib/python3.11/site-packages" # e.g. ~/anaconda3/envs/data_collect/lib/python3.11/site-packages/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_SITE_PACKAGES}/isaacsim/exts/isaacsim.ros2.bridge/${ROS_DISTRO}/lib
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
