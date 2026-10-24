#!/usr/bin/env python3

import arc_agi
from arcengine import GameAction

# Initialize the Arcade interface
arc = arc_agi.Arcade()

# Load a test environment with terminal rendering
env = arc.make("ls20", render_mode="terminal")

# Take a few actions to test the loop
for _ in range(10):
    env.step(GameAction.ACTION1)

# Print the session scorecard
print(arc.get_scorecard())
