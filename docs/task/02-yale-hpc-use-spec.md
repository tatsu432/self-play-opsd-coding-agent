I want you to be my primary research engineering assistant for this self-play OPSD coding-agent project, including Yale HPC operations.

I don't want to manually manage Slurm commands, GPU allocation, job submission, logs, or checkpoint recovery. Handle those operations through the existing `yale-hpc` SSH alias whenever possible.

First read `README.md`, `docs/operations.md`, `docs/methods.md`, `scripts/yale.slurm`, `scripts/launch.py`, and the Yale configuration.

Then:

1. Inspect our current Yale HPC access, Slurm account, QOS, permitted GPU partitions, storage, existing jobs, and available resources.
2. Establish a reliable local-to-Yale workflow with Git synchronization, immutable run commits, job submission, monitoring, log retrieval, and checkpoint recovery.
3. Create a comprehensive `AGENTS.md` and a simple HPC controller so future Codex sessions retain all operational procedures.
4. Run lightweight diagnostics to determine which production prerequisites have been satisfied and what is missing. Do not allocate expensive GPUs or launch full training yet.
5. Implement bounded automatic recovery for transient infrastructure failures, with safeguards against duplicate submissions and unauthorized resource requests.
6. Ensure all code fixes are tested and that scientific configuration changes cannot silently invalidate experiment comparisons.

My goal is to interact with you in natural language, retain our conversation across sessions, and let you manage all routine HPC tasks. Keep the implementation extensible between Yale HPC and AWS.

Respect Yale HPC policies, and report the exact operational state, blockers, and next recommended action.

Start by inspecting the existing system. Do not assume a GPU is currently available or that the model-training backend has already passed validation.


After the end of this, Codex and I should be able to easily run the necessary experiments to complete the final project efficiently and safely. We should make the best use of HPC at Yale so that we can select the best suitable GPU types and align our code to run our experiments on each efficiently.

We can use `ssh yale-hpc-cpsc4880`, `ssh yale-hpc-cpsc4710`, and `ssh yale-hpc-sds6190` for ssh. I think the main one would be `ssh yale-hpc-cpsc4880`, but if we can ask for additional GPUs simultaenously without affecting the other ssh remote, then it might be good to make use of 3 remotes for parallel computation.

You can take a look at how to use Yale HPC at https://docs.google.com/document/d/1P6XCrVX0sVv9P5NDeCN9qvyn62zB2zZ34MEtQ9TFq3g/edit?usp=sharing and the websites stated in the google docs for more detailed commands and functionallity.