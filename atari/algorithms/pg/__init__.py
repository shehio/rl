# Expose the modules under src/ as the `pg` package so that imports like
# `from pg.game import Game` work when atari/algorithms is on sys.path.
import os

__path__.append(os.path.join(os.path.dirname(__file__), "src"))
