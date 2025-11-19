
import bpy
import bmesh
import numpy as np
import os

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

output_dir = "/Users/preethi/Documents/research/3dsutureplanning/data/vision_blender_ground_truth"

# Create curved surface
bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
plane = bpy.context.active_object
plane.name = "WoundSurface"

# Edit mode - subdivide first
bpy.context.view_layer.objects.active = plane
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.subdivide(number_cuts=100)
bpy.ops.object.mode_set(mode='OBJECT')

# Get bmesh and apply curvature
bm = bmesh.new()
bm.from_mesh(plane.data)
for vert in bm.verts:
    x, y = vert.co.x, vert.co.y
    r = np.sqrt((x/3)**2 + (y/2)**2)
    z = -2 * np.exp(-r**2 / 2)
    z += 0.3 * np.sin(x) * np.cos(y)
    vert.co.z = z

bm.to_mesh(plane.data)
plane.data.update()
bm.free()

# Lighting
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
bpy.context.active_object.data.energy = 3.0

# Stereo cameras
baseline = 0.1
bpy.ops.object.camera_add(location=(0, -5, 2))
left_cam = bpy.context.active_object
left_cam.name = "LeftCamera"
left_cam.rotation_euler = (1.1, 0, 0)
left_cam.data.lens = 50

bpy.ops.object.camera_add(location=(baseline, -5, 2))
right_cam = bpy.context.active_object
right_cam.name = "RightCamera"
right_cam.rotation_euler = (1.1, 0, 0)
right_cam.data.lens = 50

# Render settings
bpy.context.scene.render.resolution_x = 1000
bpy.context.scene.render.resolution_y = 1000
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.samples = 64

# Enable depth pass in view layer
bpy.context.view_layer.use_pass_z = True

# Enable compositor nodes for depth export
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
for node in tree.nodes:
    tree.nodes.remove(node)

render_layers = tree.nodes.new('CompositorNodeRLayers')
file_output = tree.nodes.new('CompositorNodeOutputFile')
file_output.base_path = output_dir
file_output.file_slots[0].path = "left_depth_"

# Try different depth output names depending on Blender version
try:
    tree.links.new(render_layers.outputs['Depth'], file_output.inputs[0])
except KeyError:
    try:
        tree.links.new(render_layers.outputs['Z'], file_output.inputs[0])
    except KeyError:
        print("Warning: Could not find depth output, skipping depth export")

# Render left
bpy.context.scene.camera = left_cam
bpy.context.scene.render.filepath = os.path.join(output_dir, "left_image")
bpy.ops.render.render(write_still=True)

# Render right (update file output)
file_output.file_slots[0].path = "right_depth_"
bpy.context.scene.camera = right_cam
bpy.context.scene.render.filepath = os.path.join(output_dir, "right_image")
bpy.ops.render.render(write_still=True)

# Save scene
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(output_dir, "scene.blend"))

print("Blender scene created and rendered!")
