import time
CONTROL_DT = 0.0025
RECORD_EVERY = 20
POLICY_DT = RECORD_EVERY * CONTROL_DT


def wait_for_next_tick(next_tick):
    """Fixed 2.5 ms schedule: sleep until the next tick and return its time (slow ticks are caught up)."""
    import time
    next_tick += CONTROL_DT
    time.sleep(max(0.0, next_tick - time.perf_counter()))
    return next_tick