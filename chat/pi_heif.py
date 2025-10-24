# project_root/pi_heif.py
# Small shim to expose pillow_heif as pi_heif when a library imports pi_heif
try:
    from pillow_heif import *   # re-export everything
except Exception as e:
    # Re-raise to see a clear error if pillow_heif is not installed
    raise
