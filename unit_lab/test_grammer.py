from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "renderer": "RealTimePathTracing",
        "extra_args": [
            "--/persistent/rtx/modes/rt2/enabled=true",
        ],
    }
)


from isaacsim.core.api import World as W_new
from omni.isaac.core import World as W_old

print("same object:", W_new is W_old)
print("new module:", W_new.__module__)
print("old module:", W_old.__module__)
print("new qualname:", W_new.__qualname__)
print("old qualname:", W_old.__qualname__)
