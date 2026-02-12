import copy
import glob
import json
import os
import pickle
import re
import time

import numpy as np
from client.robot import Robot


class BasicAgent:
    def __init__(self, robot: Robot):
        self.robot = robot

    def load(self):
        pass
