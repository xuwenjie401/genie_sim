# Grasp-Pose Labeling System Implementation Plan

## Overview
Build an interactive Isaac Sim tool to label grasp and place poses for GenieSimAssets objects using GraspGen for automated grasp generation and manual UI interaction for place poses.

## Reference Architecture
- **Base visualizer**: `unit_lab/grasp_vis/interaction_pose_browser.py` (read-only browser)
- **GraspGen tool**: `/home/agxi/ManipLab/GraspGen/`
  - Server-client pipeline: `client-server/graspgen_server.py` + `graspgen_client.py`
  - Standalone script: `scripts/demo_object_mesh.py`
- **Assets**: `/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/objects/benchmark/`
- **Labels**: `/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/interaction/`

## Data Format Understanding
### interaction.json structure:
```json
{
  "interaction": {
    "passive": {
      "grasp": {
        "default": ["grasp_pose/grasp_pose.pkl"]
      },
      "place": {
        "正放": [{"xyz": [x,y,z], "direction": [dx,dy,dz]}]
      }
    },
    "active": {
      "place": {
        "侧放": [{"xyz": [x,y,z], "direction": [dx,dy,dz]}]
      }
    }
  }
}
```

### grasp_pose.pkl format:
- `grasp_pose`: (N, 4, 4) numpy array of SE(3) transforms
- `width`: (N,) numpy array of gripper widths
- Convention: +x=approach, +y=width, +z=top

## Implementation Plan

### Phase 1: Setup and Validation Tool
**Goal**: Create a tool to compare GraspGen output with existing labels

**Files to create**:
1. `unit_lab/grasp_vis/grasp_label_validator.py`
   - Load existing labeled object
   - Generate grasps using GraspGen
   - Visualize both side-by-side
   - Compare axis alignment and offsets
   - Output validation report

**Key features**:
- Use GraspGen standalone mode (not server-client initially)
- Load mesh from object's `Aligned.usda`
- Generate 50-100 grasps per object
- Visual comparison in Isaac Sim
- Statistical comparison (pose distribution, coverage)

**Validation criteria**:
- Axis alignment check (rotation matrix comparison)
- Position offset analysis
- Width distribution comparison
- Coverage similarity (spatial distribution)

### Phase 2: Interactive Grasp Editor
**Goal**: Build interactive tool to generate, edit, and save grasp poses

**Files to create**:
2. `unit_lab/grasp_vis/grasp_label_editor.py`
   - Based on `interaction_pose_browser.py` architecture
   - Add GraspGen integration
   - Add grasp editing capabilities
   - Save to interaction.json format

**UI Controls**:
- Object selector (unlabeled objects only)
- "Generate Grasps" button → calls GraspGen
- Grasp filtering (by score, position, orientation)
- Individual grasp selection and deletion
- Grasp pose adjustment (translate/rotate)
- "Save Labels" button → writes interaction.json + pkl

**GraspGen Integration**:
```python
# Pseudo-code
def generate_grasps(mesh_path, gripper_config):
    # Call GraspGen demo_object_mesh.py
    # Parse output YAML
    # Convert to pkl format
    # Return grasp_pose array + width array
```

### Phase 3: Place Pose Editor
**Goal**: Add manual place pose labeling via UI interaction

**Enhancement to grasp_label_editor.py**:
- Mode switcher: "Grasp Mode" / "Place Mode"
- In Place Mode:
  - Click on object surface to add place pose
  - Drag to adjust position
  - Rotate handles to adjust direction
  - Label input (e.g., "正放", "侧放")
  - Role selector (active/passive)
- Save place poses to interaction.json

**UI Implementation**:
- Use Isaac Sim's viewport picking
- Visual manipulator gizmos (translate/rotate)
- Bracket-style pose markers (reuse from browser)

### Phase 4: Batch Processing Pipeline
**Goal**: Process multiple unlabeled objects efficiently

**Files to create**:
3. `unit_lab/grasp_vis/batch_label_generator.py`
   - Scan for unlabeled objects
   - Generate grasps for all using GraspGen
   - Save preliminary labels
   - Generate report of coverage

**Features**:
- Progress tracking
- Error handling (mesh loading failures)
- Quality metrics per object
- Output summary CSV

### Phase 5: GraspGen Server-Client Integration (Optional)
**Goal**: Use server-client mode for better performance

**Enhancement**:
- Start GraspGen server once
- Client calls for each object
- Faster iteration during editing

## File Structure
```
unit_lab/grasp_vis/
├── interaction_pose_browser.py  (existing, read-only)
├── grasp_label_validator.py     (Phase 1)
├── grasp_label_editor.py        (Phase 2-3, main tool)
├── batch_label_generator.py     (Phase 4)
└── utils/
    ├── graspgen_wrapper.py      (GraspGen integration)
    ├── interaction_io.py        (read/write interaction.json)
    └── pose_editor_ui.py        (UI components)
```

## Dependencies
- Isaac Sim (already available)
- GraspGen (already installed at `/home/agxi/ManipLab/GraspGen/`)
- trimesh (for mesh loading)
- numpy, pickle, json, yaml

## Testing Strategy
1. **Validation Phase**: Test on 3-5 already-labeled objects
   - Compare GraspGen output vs existing labels
   - Verify axis alignment
   - Check if offsets are reasonable

2. **Editor Phase**: Label 2-3 new objects manually
   - Verify save/load cycle
   - Check interaction.json format
   - Validate pkl file structure

3. **Batch Phase**: Process 10-20 unlabeled objects
   - Monitor for failures
   - Check quality metrics
   - Verify all files created correctly

## Success Criteria
- [ ] GraspGen generates axis-aligned grasps matching existing label conventions
- [ ] Editor can load, modify, and save grasp poses
- [ ] Place poses can be added manually via UI
- [ ] Batch processing completes without errors
- [ ] All output files match expected format
- [ ] Labels are usable by existing `interaction_pose_browser.py`

## Next Steps
1. Start with Phase 1: Create validator to ensure GraspGen output is compatible
2. If validation passes, proceed to Phase 2: Build interactive editor
3. Add place pose editing in Phase 3
4. Implement batch processing in Phase 4
5. Optionally optimize with server-client mode in Phase 5
