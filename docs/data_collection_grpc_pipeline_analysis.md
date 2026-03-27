# Data Collection gRPC Pipeline Analysis

Date: 2026-03-26

Scope:
- This note is based on the current workspace code state under `source/data_collection`.
- The question analyzed here is whether the current server-client gRPC pipeline is dragging collection efficiency down, and whether a one-process implementation would be better if the target scope is automated collection only.
- This note is intentionally narrower than a general architecture review. It focuses on control flow, latency structure, throughput constraints, and migration tradeoffs.

Question:
- For `source/data_collection`, is the current server-client gRPC pipeline hurting efficiency enough that it should be removed?
- If teleoperation in simulation is not needed, should the system move to a one-process design?

## Executive Summary

Short answer:

- If the target scope is automated collection only, moving away from the current gRPC split is a reasonable direction.
- However, the main issue is not "localhost gRPC is inherently too slow".
- The bigger issue is that the current architecture turns many ordinary control and state reads into synchronous round trips that are gated by the Isaac Sim main loop.
- In practice, the system is logically two-service, but execution is still serialized through a single shared command slot in the server, so the transport boundary adds complexity without buying much execution parallelism.
- A one-process design is likely the better shape for your narrowed use case, but the meaningful gains will come from reducing round trips and batching state access, not merely from deleting protobuf definitions.

My recommendation:

1. Do not keep the current gRPC split as the default path for automated-only collection.
2. Do not throw away the command abstraction entirely.
3. Replace the gRPC transport with an in-process backend first.
4. After that, reduce chatty state polling and metadata RPCs, because that is where the current design amplifies latency.

## High-Level Conclusion

The current system has three different layers that matter:

1. A launch boundary.
2. A transport boundary.
3. A simulation-thread boundary.

The launch and transport boundaries can be removed in an automated-only setup.

The simulation-thread boundary cannot be ignored. Isaac Sim work still needs to happen in the correct runtime context. So the best target is not "everything becomes direct function calls from arbitrary code". The best target is "keep a local command/controller interface, but stop paying network/protobuf/process costs for every control or state query".

## What Exists Today

### 1. Logical architecture

The current runtime is a client-server system.

- The collection app creates an RPC-backed robot in [run_data_collection.py](/home/agxi/RealityLab/genie_sim/source/data_collection/scripts/run_data_collection.py#L68).
- That robot uses `RpcClient`, which opens a localhost gRPC channel in [client.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/robot/client.py#L83).
- The server side starts a gRPC service in [grpc_server.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/grpc_server.py#L778).
- The gRPC server is backed by `CommandController`, which owns the actual Isaac Sim interaction state in [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L63).

That means the current control path is:

`DataCollectionAgent -> IsaacSimRpcRobot -> RpcClient -> gRPC -> GrpcServer -> CommandController -> UIBuilder / Isaac Sim / cuRobo`

### 2. Operational launch model

The docs and scripts show two slightly different views of startup.

The README documents a manual two-terminal workflow:

- Start `data_collector_server.py` in one terminal.
- Start `run_data_collection.py` in a second terminal.

Reference:
- [README.md](/home/agxi/RealityLab/genie_sim/source/data_collection/README.md#L132)

But the current containerized startup script already launches both programs for you:

- `run_data_collection.sh` starts a container with `data_collection_entrypoint.sh` as its entrypoint in [run_data_collection.sh](/home/agxi/RealityLab/genie_sim/source/data_collection/scripts/run_data_collection.sh#L184).
- That entrypoint starts `data_collector_server.py` in the background in [data_collection_entrypoint.sh](/home/agxi/RealityLab/genie_sim/source/data_collection/scripts/data_collection_entrypoint.sh#L261).
- Then it starts `run_data_collection.py` in the background in [data_collection_entrypoint.sh](/home/agxi/RealityLab/genie_sim/source/data_collection/scripts/data_collection_entrypoint.sh#L290).

This distinction matters:

- The system is still logically two-process and client-server.
- But from the normal automated workflow, it already behaves like a coordinated bundle rather than two independently managed tools.

That reduces the operational value of keeping gRPC as a hard architecture boundary for the automated path.

## End-To-End Command Flow

### 1. A motion request from the client

The client sends high-level moves, not per-frame joint streaming.

Example:

- `IsaacSimRpcRobot.move(...)` eventually calls `self.client.moveto(...)` in [omni_robot.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/robot/omni_robot.py#L291).
- `RpcClient.moveto(...)` builds a `LinearMoveReq` and performs a synchronous RPC in [client.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/robot/client.py#L160).

That is important because it means the gRPC layer is not being used for high-frequency servo traffic. A single RPC often represents a large amount of downstream work.

### 2. What the server actually does with that request

The gRPC service handler does not directly execute the simulation logic.

For linear moves:

- `armService.linear_move(...)` forwards the request to `blocking_start_server(...)` in [grpc_server.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/grpc_server.py#L118).
- The actual handoff happens here: [grpc_server.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/grpc_server.py#L137).

Inside `CommandController`:

- The request is stored in shared fields `self.data` and `self.Command` in [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L1740).
- The caller then waits on a condition variable until a result is produced in [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L1743).

The real execution only happens later, from the simulation loop:

- `on_physics_step()` is called each step in [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L643).
- That calls `on_command_step()` in [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L665).
- `on_command_step()` switches on `self.Command` and executes exactly one handler path in [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L1660).
- Once the command completes, waiting threads are notified in [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L1721).

So the real path is:

1. Client issues RPC.
2. gRPC thread receives request.
3. Request is copied into shared controller state.
4. Caller blocks.
5. Isaac Sim loop reaches the next `on_physics_step()`.
6. `on_command_step()` dispatches the command.
7. Result is written back.
8. Waiting RPC thread returns the response.

### 3. Physics-step gating adds fixed latency

The server defaults to `physics_step = 30` in [data_collector_server.py](/home/agxi/RealityLab/genie_sim/source/data_collection/scripts/data_collector_server.py#L42), which becomes `physics_dt = 1 / args.physics_step` in [data_collector_server.py](/home/agxi/RealityLab/genie_sim/source/data_collection/scripts/data_collector_server.py#L73).

Inference:

- A synchronous command cannot complete until the simulation loop processes it.
- At the default 30 Hz physics rate, one command has an irreducible latency component on the order of one physics tick, roughly `33 ms`, even before considering protobuf serialization, gRPC scheduling, or any actual simulation work.

That fixed latency is small for a long motion plan and large for many small control/state requests.

## The Current System Is More Serialized Than It First Appears

### 1. The gRPC layer advertises concurrency

The server is created with `ThreadPoolExecutor(max_workers=10)` in [grpc_server.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/grpc_server.py#L788).

That suggests the transport layer can accept multiple concurrent requests.

### 2. The command controller effectively serializes execution

But the controller stores only one active command in shared fields:

- `self.data` in [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L76)
- `self.Command` in [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L77)
- `self.data_to_send` in [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L78)

This is not a true multi-command execution queue. It is closer to a single in-flight command slot plus a wait/notify mechanism.

Practical implication:

- Even though gRPC can receive requests concurrently, the useful work path is effectively serialized through one controller state machine.
- So the architecture pays for concurrency machinery at the RPC layer, but the actual simulation command path remains mostly single-threaded.

This does not automatically mean the code is wrong. It does mean that the gRPC split is not buying real throughput proportional to its complexity.

## Where The Overhead Actually Comes From

### 1. Pure gRPC and protobuf overhead

This part exists, but it is not the dominant problem.

Costs include:

- protobuf object construction
- serialization and deserialization
- local socket transport
- gRPC worker scheduling

There is also some repeated stub construction on the client side:

- `set_frame_state()` creates a new stub in [client.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/robot/client.py#L124)
- `moveto()` creates a new stub in [client.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/robot/client.py#L160)
- `get_object_pose()` creates a new stub in [client.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/robot/client.py#L278)
- `get_part_dof_joint()` creates a new stub in [client.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/robot/client.py#L285)
- `get_ee_pose()` creates a new stub in [client.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/robot/client.py#L379)

This is not ideal, but it is still probably a second-order effect relative to the per-command simulation-loop synchronization.

### 2. Thread handoff and blocking synchronization

This is more important than raw gRPC overhead.

Every command does the following:

- enter gRPC worker thread
- write to controller shared state
- wait on a condition variable
- resume after the sim loop writes back a result

That introduces:

- thread scheduling overhead
- context switching
- lock/condition overhead
- result marshaling overhead

Again, for a long motion this is minor. For many small state reads, it accumulates.

### 3. Physics-loop service latency

This is likely the main fixed latency source for synchronous commands.

Because `on_command_step()` is invoked from `on_physics_step()`:

- commands are not serviced immediately on arrival
- commands wait for the simulation loop
- commands are therefore sensitive to physics rate, rendering load, ROS publishing load, and any heavy work already inside the server frame

This is why the transport discussion alone is incomplete. The system is not "RPC request enters and instantly mutates sim state". It is "RPC request enters a wait-until-next-sim-step pipeline".

### 4. Chatty automated workflow

This is where the current design becomes expensive.

The automated agent performs many RPCs around each action.

Before motion:

- `set_frame_state()` is called in [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L743)
- `remove_objs_from_obstacle()` may be called in [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L763)

Motion:

- `move_pose()` is called in [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L801)

After motion:

- `set_frame_state()` again in [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L821)
- gripper action may issue `set_gripper_state()` and possibly `detach_obj()` through [omni_robot.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/robot/omni_robot.py#L171)
- another `set_frame_state()` in [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L838)

State refresh:

- end-effector pose is fetched through `get_ee_pose()` in [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L851)
- then `update_objects()` refreshes object state in [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L222)
- each object refresh calls `get_object_pose()` via [omni_robot.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/robot/omni_robot.py#L337)
- articulated parts also call `get_part_dof_joint()` in [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L238)

More metadata:

- `set_frame_state()` again in [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L855)
- `attach_obj()` may be called in [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L884)
- final `set_frame_state()` in [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L888)

This means a single logical action can easily involve:

- 1 high-level motion RPC
- several metadata RPCs
- several state query RPCs
- plus one object-pose RPC per object
- plus one articulated-part RPC per articulated part

That is the strongest argument against the current split for automated-only collection.

### 5. Recording and ROS publishing

The gRPC split is not the only runtime cost.

When recording is enabled:

- the entrypoint adds `--publish_ros` for the server in [data_collection_entrypoint.sh](/home/agxi/RealityLab/genie_sim/source/data_collection/scripts/data_collection_entrypoint.sh#L234)
- `on_physics_step()` ticks ROS publishers in [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L666)

So if you observe poor throughput during recording, some of that slowdown is expected to come from:

- ROS bridge work
- sensor publication
- rendering
- recording serialization

not just from the command transport layer.

### 6. Motion planning and simulation work

The heavy work remains server-side:

- Isaac Sim stepping in [data_collector_server.py](/home/agxi/RealityLab/genie_sim/source/data_collection/scripts/data_collector_server.py#L104)
- world rendering in [data_collector_server.py](/home/agxi/RealityLab/genie_sim/source/data_collection/scripts/data_collector_server.py#L110)
- cuRobo stepping inside `on_physics_step()` in [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L648)

So it would be incorrect to claim that gRPC is the dominant runtime cost in all cases.

For long motions, collision-aware planning, scene loading, or recording-heavy episodes, the main costs are elsewhere.

## Why gRPC Is Not The Main Bottleneck In Every Case

There are real reasons the current system can still be acceptable:

- Motion RPCs are coarse-grained, not per-frame servo commands.
- The client does not stream large trajectories at control rate for ordinary operation.
- The server still spends most of its time in simulation, planning, rendering, and recording.
- Localhost gRPC is usually fast enough when requests are infrequent and high-level.

So if the workload were:

- one initialization
- one large move
- one recording session
- one exit

then the gRPC layer would probably not be worth optimizing first.

That is not the actual pattern of automated collection, though. Automated collection is chatty enough that the fixed per-request costs start to matter.

## Why One Process Makes Sense For Automated-Only Collection

If teleoperation, remote control, and multi-client access are out of scope, the one-process case becomes much stronger.

Benefits:

- simpler startup and failure model
- easier debugging because stack traces stay local
- no protobuf or RPC boundary for ordinary internal control
- lower latency for metadata/state traffic
- easier batching of state access
- easier profiling across the whole control path

The current launch scripts already treat both pieces as a coordinated bundle in one container. That weakens the practical argument for keeping them as separate processes for the automated path.

## Important Caveat: One Process Alone Is Not Enough

This is the key nuance.

If you simply do this:

- keep the same client logic
- keep the same blocking command handoff
- keep the same chatty call pattern
- just replace gRPC with local Python calls

then you will remove some overhead, but not all of the structural latency.

You will still have:

- per-command synchronization
- simulation-step gating
- repeated state fetches
- repeated metadata writes

So the real hierarchy of likely benefit is:

1. Reduce round trips.
2. Batch state access.
3. Avoid transport/process overhead.

not:

1. Delete gRPC and assume the problem is solved.

## Recommended Target Architecture

### 1. Keep the command/controller abstraction

I would keep a central simulation-side controller, because the current code already makes it clear that a controlled bridge into Isaac Sim is useful.

The current `CommandController` is the correct conceptual seam. The problem is the transport and call granularity around it, not the existence of a controller.

### 2. Replace `RpcClient` with a transport-agnostic backend

The clean migration path is to turn the client robot side into a backend interface.

For example:

- `GrpcBackend`: current behavior
- `LocalBackend`: direct in-process calls into the controller

That lets you:

- preserve the high-level client/agent logic initially
- compare performance without rewriting everything at once
- keep gRPC available only for optional future workflows

### 3. Add batch state APIs

This is likely the highest-value architectural improvement for automated collection.

The current `update_objects()` pattern is expensive because it performs one query per state item.

Instead, add something like:

- `get_world_snapshot()`
- `get_object_poses(object_ids)`
- `get_runtime_state(gripper_pose, object_poses, articulated_joint_states, attachments)`

Then one action can refresh state with one local call instead of many synchronous round trips.

### 4. Reduce `set_frame_state()` chatter

The agent currently writes frame metadata multiple times per action.

Some of this is probably useful for recording semantics, but it should be reviewed as data design, not only as control design.

Possible improvements:

- only write state on actual semantic transitions
- accumulate local stage metadata and flush once
- separate lightweight in-memory state updates from persistent event logging

### 5. Keep high-level motion commands coarse

This part of the current system is actually good for performance.

`move_pose()` forwards a high-level motion request rather than manually stepping joints from the client side in [omni_robot.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/robot/omni_robot.py#L291).

That design should be preserved in a local-backend version.

## Migration Options

### Option A: Local transport adapter, same control pattern

Description:

- keep `DataCollectionAgent`
- keep `IsaacSimRpcRobot`-style interface
- replace `RpcClient` with `LocalClient`
- `LocalClient` calls into an in-process controller adapter

Pros:

- lowest migration risk
- easiest A/B comparison against current pipeline
- removes protobuf and process boundary

Cons:

- still keeps much of the current call chatter
- still likely keeps synchronization boundary
- speedup will be real but limited

Expected outcome:

- good first step
- probably worthwhile
- not the maximum achievable gain

### Option B: One process plus batched state snapshot

Description:

- do everything in Option A
- add batch state refresh APIs
- replace object-by-object polling after each action

Pros:

- better real throughput improvement
- reduces the most obviously inefficient automated path
- keeps architecture understandable

Cons:

- moderate refactor effort
- requires changing agent-side assumptions

Expected outcome:

- this is the highest-value medium-risk option

### Option C: In-process scheduler integrated with simulation loop

Description:

- run the automated agent as a local task tightly integrated with the sim loop
- avoid per-command wait/notify where practical
- turn action execution into a simulation-driven state machine or coroutine

Pros:

- best long-term architecture for automated-only collection
- minimal transport overhead
- clear ownership of sim-thread interactions

Cons:

- largest rewrite
- easiest place to introduce regressions if done too quickly

Expected outcome:

- highest payoff
- not the best first move unless you want a larger architectural rewrite now

## When Keeping gRPC Still Makes Sense

There are still legitimate reasons to keep the current split available:

- future teleoperation
- remote planner or GUI client
- multi-client access
- process isolation from Isaac Sim crashes
- external tool integration

If those are future goals, the best compromise is:

- keep gRPC as an optional adapter
- do not keep it as the only default path for automated collection

## What I Would Do In Practice

If I were optimizing this codebase for your stated scope, I would do the following in order:

1. Introduce a local backend that matches the existing client interface.
2. Switch automated collection to the local backend by default.
3. Keep gRPC behind a flag for debugging or future remote use.
4. Add a batched world-state snapshot API.
5. Replace repeated per-object state polling in `update_objects()`.
6. Re-evaluate how many `set_frame_state()` writes are actually needed.
7. Profile again before attempting a deeper scheduler rewrite.

This sequence gives most of the likely benefit without taking on the full risk of an immediate architecture rewrite.

## Measurement Plan Before And After Refactor

The code already has timing infrastructure in `CommandController`:

- timing storage in [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L146)
- timing context in [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L153)
- timing report in [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L193)

That is a good start, but it is not enough to answer the architecture question completely.

I would measure:

- end-to-end episode runtime
- per-stage runtime
- per-action runtime
- count of RPC-style calls per action
- average and p95 command latency from request send to result return
- time spent in `update_objects()`
- time spent in object pose refresh
- time spent in `on_physics_step()`
- time spent in ROS publish path when recording is enabled
- time spent in motion planning vs time spent waiting for service

Minimum benchmark matrix:

- current gRPC split, recording off
- current gRPC split, recording on
- local backend, recording off
- local backend, recording on
- local backend plus batched state snapshot, recording off
- local backend plus batched state snapshot, recording on

This will tell you whether:

- the main pain is transport
- the main pain is state polling
- the main pain is recording/rendering
- or some combination

## Risks In A One-Process Refactor

There are real risks, and they should be acknowledged explicitly.

### 1. Isaac Sim thread-safety assumptions

The current controller exists partly because sim operations are not ordinary pure-Python business logic. A careless one-process refactor could accidentally call sim code from the wrong execution context.

### 2. Hidden coupling

The current split forces some API discipline. A direct one-process rewrite can devolve into uncontrolled cross-layer access if not designed carefully.

### 3. Loss of optional remote workflows

If gRPC is removed entirely, future teleoperation or remote tooling becomes harder to reintroduce.

### 4. Refactor scope creep

If transport removal gets mixed with planner changes, recording changes, and controller rewrites, the migration will become harder to validate.

That is why a transport-agnostic backend interface is the right first step.

## Final Recommendation

For automated collection only, yes: the current server-client gRPC pipeline is very likely dragging efficiency down enough that it is no longer the best default architecture.

But the precise statement should be:

- raw gRPC is not the main villain
- synchronous request/response control across a transport boundary, combined with simulation-step gating and chatty state polling, is the real issue

Therefore:

- moving to one process is a good idea
- keeping a controller boundary is also a good idea
- the best near-term target is a local backend plus batched state access
- deleting gRPC without reducing round trips will help, but only partially

If the goal is maximum practical payoff with controlled risk, I would choose:

`one-process automated path + optional gRPC adapter + batched state snapshot`

That is the architecture I believe best matches the current code and your stated requirements.

## Appendix: Concrete Reasons The Current Design Feels Slower Than It Needs To

This is the shortest plain-language summary of the analysis:

- The normal automated path is already bundled together in one container, so process separation is not buying much operational flexibility.
- Each command goes through RPC, then waits for the sim loop, so the request is not serviced immediately.
- The gRPC server can accept multiple requests, but useful execution still funnels through a single controller command slot.
- The automated agent performs enough small control and state requests that fixed latency starts to matter.
- The most valuable optimization is not "faster network". It is "fewer sync boundaries".
