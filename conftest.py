# Presence of this file puts the repo root on sys.path so tests can import `src`.

# Force a non-interactive backend before any pyplot import so plot tests run
# headless (no display) in CI. Set here AND at the top of the plots test module.
import matplotlib

matplotlib.use("Agg")
