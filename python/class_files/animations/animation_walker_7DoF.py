import vtk
import numpy as np
import cv2
import time
from vtk.util import numpy_support

class AnimationWalking7DoF:

    def __init__(self, sys, X_data, tspan, dt):
        self.sys = sys
        # Retrieve physical parameters from the system object
        self.l_upper = sys.l_upper
        self.l_lower = sys.l_lower
        
        # Optional: Visualization dimensions
        self.link_width = 0.05
        self.base_radius = 0.1
        self.joint_radius = 0.06 
        self.depth = 0.1

        self.X_data = X_data.T # Shape (14, N) -> Transpose to access by index
        # 7-DoF q structure: [x_base, y_base, theta_base, q_u1_rel, q_l1_rel, q_u2_rel, q_l2_rel]
        self.q = self.X_data[0:7, :] 
        self.tspan = tspan
        self.dt = dt
        self.N = len(tspan)
        self.total_duration = self.N * self.dt

        # --- State variables ---
        self.timestep_index = 0
        self.recording = False
        self.video_writer = None
        self.window_to_image_filter = None
        self.text_actor = None
        self.start_time_wall_clock = None

        # --- Pre-compute trajectories ---
        print("Pre-computing trajectories...")
        self.trajectories = self.compute_all_positions(self.q)
        print("Computation complete.")

        # --- VTK object placeholders ---
        self.renderer = None
        self.render_window = None
        self.interactor = None
        self.cam_widget = None

        # Transformation matrices
        self.H_base = vtk.vtkMatrix4x4()
        
        self.H_u1 = vtk.vtkMatrix4x4()
        self.H_l1 = vtk.vtkMatrix4x4()
        self.H_knee1 = vtk.vtkMatrix4x4()
        
        self.H_u2 = vtk.vtkMatrix4x4()
        self.H_l2 = vtk.vtkMatrix4x4()
        self.H_knee2 = vtk.vtkMatrix4x4()

    def compute_all_positions(self, q):
        """
        Computes the center position and rotation angle for every link at every timestep.
        Uses 7-DoF Kinematics: Absolute Angle = Base_Theta + Relative_Angle
        """
        x_base = q[0, :]
        y_base = q[1, :]
        theta_base = q[2, :] # NEW: Base Pitch
        
        # Relative angles
        q_u1_rel = q[3, :]
        q_l1_rel = q[4, :]
        q_u2_rel = q[5, :]
        q_l2_rel = q[6, :]

        # --- Compute Absolute Angles ---
        # Leg 1
        abs_u1 = theta_base + q_u1_rel
        abs_l1 = theta_base + q_u1_rel + q_l1_rel
        
        # Leg 2
        abs_u2 = theta_base + q_u2_rel
        abs_l2 = theta_base + q_u2_rel + q_l2_rel

        # --- Base ---
        # Center is just (x, y). Rotation is theta_base.
        
        # --- Leg 1 (Left/Front) ---
        vec_u1_x = self.l_upper * np.sin(abs_u1)
        vec_u1_y = -self.l_upper * np.cos(abs_u1)
        
        cx_u1 = x_base + 0.5 * vec_u1_x
        cy_u1 = y_base + 0.5 * vec_u1_y
        
        # Knee 1
        kx_1 = x_base + vec_u1_x
        ky_1 = y_base + vec_u1_y
        
        # Lower Leg 1
        vec_l1_x = self.l_lower * np.sin(abs_l1)
        vec_l1_y = -self.l_lower * np.cos(abs_l1)
        
        cx_l1 = kx_1 + 0.5 * vec_l1_x
        cy_l1 = ky_1 + 0.5 * vec_l1_y

        # --- Leg 2 (Right/Back) ---
        vec_u2_x = self.l_upper * np.sin(abs_u2)
        vec_u2_y = -self.l_upper * np.cos(abs_u2)
        
        cx_u2 = x_base + 0.5 * vec_u2_x
        cy_u2 = y_base + 0.5 * vec_u2_y
        
        # Knee 2
        kx_2 = x_base + vec_u2_x
        ky_2 = y_base + vec_u2_y
        
        # Lower Leg 2
        vec_l2_x = self.l_lower * np.sin(abs_l2)
        vec_l2_y = -self.l_lower * np.cos(abs_l2)
        
        cx_l2 = kx_2 + 0.5 * vec_l2_x
        cy_l2 = ky_2 + 0.5 * vec_l2_y

        return {
            "base_pos": np.vstack([x_base, y_base, np.zeros(self.N)]),
            "base_theta": theta_base, # Store base rotation
            
            "u1_pos": np.vstack([cx_u1, cy_u1, np.zeros(self.N)]),
            "u1_theta": abs_u1,
            "knee1_pos": np.vstack([kx_1, ky_1, np.zeros(self.N)]),
            "l1_pos": np.vstack([cx_l1, cy_l1, np.zeros(self.N)]),
            "l1_theta": abs_l1,
            
            "u2_pos": np.vstack([cx_u2, cy_u2, np.zeros(self.N)]),
            "u2_theta": abs_u2,
            "knee2_pos": np.vstack([kx_2, ky_2, np.zeros(self.N)]),
            "l2_pos": np.vstack([cx_l2, cy_l2, np.zeros(self.N)]),
            "l2_theta": abs_l2
        }

    def _update_matrix(self, matrix, pos, theta):
        """Helper to update a VTK matrix with 2D pos and Z-rotation theta (rad)"""
        c = -np.cos(theta) # Visual fix: VTK cube defaults to Y-align, physics is Y-down
        s = -np.sin(theta)

        # In standard 2D physics: 0 is Up/Down depending on gravity. 
        # Here we assume 0 is aligned with gravity (Down).
        # VTK Y is Up. 
        # A rotation matrix for Z axis: [c -s; s c]
        
        # Adjust for visualization: We want the link to point along the computed vector.
        # The computed vector assumes 0 is "Down".
        # We need to render the CubeSource rotated to match.
        
        matrix.SetElement(0, 0, c)
        matrix.SetElement(0, 1, -s)
        matrix.SetElement(1, 0, s)
        matrix.SetElement(1, 1, c)
        
        matrix.SetElement(0, 3, pos[0])
        matrix.SetElement(1, 3, pos[1])
        matrix.SetElement(2, 3, pos[2])

    def create_link_actor(self, matrix, length, color, width=0.05):
        geom = vtk.vtkCubeSource()
        geom.SetXLength(width)
        geom.SetYLength(length)
        geom.SetZLength(width)

        _H = vtk.vtkMatrixToLinearTransform()
        _H.SetInput(matrix)
        tf_filter = vtk.vtkTransformPolyDataFilter()
        tf_filter.SetInputConnection(geom.GetOutputPort())
        tf_filter.SetTransform(_H)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tf_filter.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        return actor
    
    def create_sphere_actor(self, matrix, radius, color):
        """Helper to create a joint sphere"""
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(radius)
        sphere.SetThetaResolution(16)
        sphere.SetPhiResolution(16)

        _H = vtk.vtkMatrixToLinearTransform()
        _H.SetInput(matrix)
        tf_filter = vtk.vtkTransformPolyDataFilter()
        tf_filter.SetInputConnection(sphere.GetOutputPort())
        tf_filter.SetTransform(_H)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tf_filter.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        return actor
    
    def create_base_actor(self, matrix, radius, color):
        """Helper to create the Base actor (Cube to show rotation)"""
        # Using a Cube instead of Sphere for Base so we can see the Pitch rotation
        cube = vtk.vtkCubeSource()
        cube.SetXLength(radius*2)
        cube.SetYLength(radius*1.5)
        cube.SetZLength(radius*1.5)

        _H = vtk.vtkMatrixToLinearTransform()
        _H.SetInput(matrix)
        tf_filter = vtk.vtkTransformPolyDataFilter()
        tf_filter.SetInputConnection(cube.GetOutputPort())
        tf_filter.SetTransform(_H)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tf_filter.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        return actor

    def create_environment(self):
        # 1. Base (Hip) - Green
        # Changed to Cube to visualize rotation
        self.actor_base = self.create_base_actor(self.H_base, self.base_radius, [0.2, 0.8, 0.2])

        # 2. Leg 1 (Red)
        self.actor_u1 = self.create_link_actor(self.H_u1, self.l_upper, [0.8, 0.2, 0.2])
        self.actor_l1 = self.create_link_actor(self.H_l1, self.l_lower, [0.8, 0.2, 0.2])
        # Knee 1
        self.actor_knee1 = self.create_sphere_actor(self.H_knee1, self.joint_radius, [0.8, 0.2, 0.2])

        # 3. Leg 2 (Blue)
        self.actor_u2 = self.create_link_actor(self.H_u2, self.l_upper, [0.2, 0.2, 0.8])
        self.actor_l2 = self.create_link_actor(self.H_l2, self.l_lower, [0.2, 0.2, 0.8])
        # Knee 2
        self.actor_knee2 = self.create_sphere_actor(self.H_knee2, self.joint_radius, [0.2, 0.2, 0.8])

        # 4. Floor
        floor = vtk.vtkCubeSource()
        floor.SetXLength(10.0)
        floor.SetYLength(0.02)
        floor.SetZLength(2.0)
        
        floor_tf = vtk.vtkTransform()
        floor_tf.Translate(0, -0.01, 0)
        
        tf_filter_floor = vtk.vtkTransformPolyDataFilter()
        tf_filter_floor.SetInputConnection(floor.GetOutputPort())
        tf_filter_floor.SetTransform(floor_tf)
        
        mapper_floor = vtk.vtkPolyDataMapper()
        mapper_floor.SetInputConnection(tf_filter_floor.GetOutputPort())
        actor_floor = vtk.vtkActor()
        actor_floor.SetMapper(mapper_floor)
        actor_floor.GetProperty().SetColor([0.7, 0.7, 0.7])

        # 5. Text
        self.text_actor = vtk.vtkTextActor()
        self.text_actor.SetInput("t = 0.00 s")
        self.text_actor.GetTextProperty().SetFontSize(30)
        self.text_actor.GetTextProperty().SetColor(0, 0, 0)
        self.text_actor.SetPosition(30, 30)

        # Renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.AddActor(self.actor_base)
        self.renderer.AddActor(self.actor_u1)
        self.renderer.AddActor(self.actor_l1)
        self.renderer.AddActor(self.actor_knee1)
        
        self.renderer.AddActor(self.actor_u2)
        self.renderer.AddActor(self.actor_l2)
        self.renderer.AddActor(self.actor_knee2)
        
        self.renderer.AddActor(actor_floor)
        self.renderer.AddActor(self.text_actor)
        self.renderer.SetBackground(1, 1, 1)

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetWindowName("Walking 7DoF Animation")
        
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        
        self.cam_widget = vtk.vtkCameraOrientationWidget()
        self.cam_widget.SetParentRenderer(self.renderer)
        self.cam_widget.On()
        
        # Camera
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(0, 1.0, 8.0)
        camera.SetFocalPoint(0, 0.5, 0)

    def set_scene_to_timestep(self, i):
        # Base (Now includes rotation)
        p_base = self.trajectories["base_pos"][:, i]
        theta_base = self.trajectories["base_theta"][i]
        
        # For base, we update both pos and orientation
        # (Assuming the CubeSource is axis aligned initially)
        # Visual rotation needs to match physics frame
        c = np.cos(theta_base) # VTK coords check
        s = np.sin(theta_base)
        
        # Matrix Update for Base
        # Note: If theta=0 means upright, we map it directly.
        self.H_base.SetElement(0, 0, c)
        self.H_base.SetElement(0, 1, -s)
        self.H_base.SetElement(1, 0, s)
        self.H_base.SetElement(1, 1, c)
        self.H_base.SetElement(0, 3, p_base[0])
        self.H_base.SetElement(1, 3, p_base[1])
        self.H_base.SetElement(2, 3, p_base[2])

        # Leg 1
        self._update_matrix(self.H_u1, 
                            self.trajectories["u1_pos"][:, i], 
                            self.trajectories["u1_theta"][i])
        
        p_k1 = self.trajectories["knee1_pos"][:, i]
        self.H_knee1.SetElement(0, 3, p_k1[0])
        self.H_knee1.SetElement(1, 3, p_k1[1])
        self.H_knee1.SetElement(2, 3, p_k1[2])

        self._update_matrix(self.H_l1, 
                            self.trajectories["l1_pos"][:, i], 
                            self.trajectories["l1_theta"][i])

        # Leg 2
        self._update_matrix(self.H_u2, 
                            self.trajectories["u2_pos"][:, i], 
                            self.trajectories["u2_theta"][i])
        
        p_k2 = self.trajectories["knee2_pos"][:, i]
        self.H_knee2.SetElement(0, 3, p_k2[0])
        self.H_knee2.SetElement(1, 3, p_k2[1])
        self.H_knee2.SetElement(2, 3, p_k2[2])

        self._update_matrix(self.H_l2, 
                            self.trajectories["l2_pos"][:, i], 
                            self.trajectories["l2_theta"][i])

        self.H_base.Modified()
        self.H_u1.Modified()
        self.H_l1.Modified()
        self.H_knee1.Modified()
        self.H_u2.Modified()
        self.H_l2.Modified()
        self.H_knee2.Modified()

    def update_scene_callback(self, interactor, event):
        now = time.time()
        elapsed = now - self.start_time_wall_clock
        time_in_sim = elapsed % self.total_duration
        idx = int(time_in_sim / self.dt)
        if idx >= self.N: idx = self.N - 1
        
        self.timestep_index = idx
        self.text_actor.SetInput(f"t = {time_in_sim:.2f} s")
        self.set_scene_to_timestep(idx)
        self.render_window.Render()

    def write_video_frame(self):
        if self.recording and self.video_writer is not None:
            self.window_to_image_filter.Modified()
            self.window_to_image_filter.Update()
            vtk_image = self.window_to_image_filter.GetOutput()
            width, height, _ = vtk_image.GetDimensions()
            vtk_array = vtk_image.GetPointData().GetScalars()
            components = vtk_array.GetNumberOfComponents()
            arr = numpy_support.vtk_to_numpy(vtk_array).reshape(height, width, components)
            arr = np.flip(arr, 0)
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            self.video_writer.write(arr)

    def animate(self, save_video=False, filename="walker_7dof.mp4", resolution=(1280, 720), fullscreen=False):
        self.create_environment()

        if fullscreen:
            self.render_window.FullScreenOn()
        else:
            self.render_window.SetSize(resolution[0], resolution[1])

        self.recording = save_video

        if self.recording:
            # Offline render logic (Smooth)
            self.window_to_image_filter = vtk.vtkWindowToImageFilter()
            self.window_to_image_filter.SetInput(self.render_window)
            self.window_to_image_filter.SetInputBufferTypeToRGB()
            self.window_to_image_filter.ReadFrontBufferOff()
            
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(filename, fourcc, 60, self.render_window.GetSize(), True)
            
            total_frames = int(self.N * self.dt * 60)
            print(f"Rendering {total_frames} frames...")
            
            for k in range(total_frames):
                t = k / 60.0
                idx = int(t / self.dt)
                if idx >= self.N: idx = self.N - 1
                self.set_scene_to_timestep(idx)
                self.text_actor.SetInput(f"t = {t:.2f} s")
                self.render_window.Render()
                self.write_video_frame()
                if k % 60 == 0: print(f"Rendering {k}/{total_frames}")
            
            self.video_writer.release()
            self.render_window.Finalize()
            self.interactor.TerminateApp()
        else:
            # Live logic
            print("Starting live preview...")
            
            self.set_scene_to_timestep(0)
            self.render_window.Render()
            
            self.interactor.Initialize()
            self.interactor.AddObserver(vtk.vtkCommand.TimerEvent, self.update_scene_callback)
            self.interactor.CreateRepeatingTimer(16)
            
            self.start_time_wall_clock = time.time()
            self.interactor.Start()

if __name__ == "__main__":
    # Test stub for 7-DoF
    class DummySys:
        l_upper = 0.5
        l_lower = 0.5
    
    t = np.linspace(0, 2, 200)
    q = np.zeros((7, 200))
    q[1, :] = 1.1 + 0.1*np.sin(5*t)  # y
    q[2, :] = 0.1 * np.sin(2*t)      # theta_B (Pitch)
    q[3, :] = 0.5 * np.sin(5*t)      # Hip
    q[4, :] = 0.5 * np.cos(5*t)      # Knee
    
    X = np.vstack([q, np.zeros_like(q)]).T
    anim = AnimationWalking7DoF(DummySys(), X, t, 0.01)
    anim.animate()