# Expose the modules under src/ as the `dqn` package so that imports like
# `from dqn.agent import Agent` work when atari/algorithms is on sys.path.
import os

__path__.append(os.path.join(os.path.dirname(__file__), "src"))
