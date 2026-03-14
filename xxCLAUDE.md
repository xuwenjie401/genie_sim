# Done
1. `unit_lab/free_worker_gripper.py` - gripper keyboard-interactive unit test, done.

---

# Current Status: main pipeline, need to modify about "place"-action-stage because of galbot gripper test result

## What Was Implemented
in unit_lab/free_worker_gripper.py, we have test result:
galbot gripper, due to its closed-loop-links, can't be perfect supported by isaac sim PhysX, so we settle for special-designed grasp-force-control pipeline, i.e. velocity control to reach grasp-state, then stop velocity-command, and change to persistant position control to hold gripper at this state,

but this makes move(pick)-->gripper close-->move(place) workflow failed at "place", we have to remove target obstacle from curobo_motion-gen_world to resume motion_gen planning (for other grippers, we don't have to do this)

so for now, the pipeline can basically work, but since we remove the object in stage-"place",in standard data-collection workflow,(i.e. source/data_collection/ server client), it often make collisions with place-target(box), (in free_worker_gripper.py unit-test, we lift 20cm higher and place at a height not ideal--20cm, too high).

what's your opinion for solve this?

------------------------

actually, i suddenly find that, maybe that's because of my (now-commented) solve_ik()--type    
chosen to be "Simple"&"Normal" instead of "AvoidObs"; now i changed them all, so i got "Unable   
to find valid target_obj_pose for place action, try next active/passive element combination." at 
 place.py, but it's obvious that from my eye-judge it's a planable situation, so find a way to   
debug precisely


## Config Rule
Do NOT modify original config files. Create new files or add config in code with NOTE comments.
