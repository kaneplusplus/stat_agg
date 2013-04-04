from cnidaria import *
import time

# Start 3 local workers on this machine.
start_local_workers(3)

# Make sure the workers are spawned and waiting for jobs. They can't receive
# a published before they exist. We'll give them a second to get started.
time.sleep(1)

# Start the coordinator.
c = Coordinator()

# An eval is an expression or a function call that returns something.
c.publish_eval("4*3")

# An exec should have assignments in it.
c.publish_exec("a = 3+4")

# And then it can be retrieved.
c.publish_get("a")

# Now shut down the workers.
c.publish_shutdown()

