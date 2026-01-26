import vtk
import numpy as np
import cv2
import time
from vtk.util import numpy_support

class AnimationDualArmBox:

    def __init__(self, sys, X_data, tspan, dt):
        self.sys = sys
        # Dimensions
        self.w_box = sys.w_box
        self.h_box = sys.h_box
        self.w_EE = sys.w_EE
        self.h_EE = sys.h_EE
        self.l1 = sys.l1
        self.l2 = sys.l2
        self.link_width = 0.04

        self.X_data = X_data
        # q indices: 
        # 0-2: Left Arm (q1, q2, q3)
        # 3-5: Right Arm (q4, q5, q6)
        # 6-8: Box (x, y, phi)
        self.q = self.X_data[0:9, :]
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

        # --- VTK object placeholders ---
        self.renderer = None
        self.render_window = None
        self.interactor = None

        # Transformation matrices (4x4)
        self.H_L1 = vtk.vtkMatrix4x4()
        self.H_L2 = vtk.vtkMatrix4x4()
        self.H_EE1 = vtk.vtkMatrix4x4()
        
        self.H_R1 = vtk.vtkMatrix4x4()
        self.H_R2 = vtk.vtkMatrix4x4()
        self.H_EE2 = vtk.vtkMatrix4x4()
        
        self.H_box = vtk.vtkMatrix4x4()

    def create_geometry_actor(self, length, height, depth, color, shift_x=0.0):
        """
        Creates a cube actor. 
        shift_x: Shifts the geometry relative to its local origin. 
                 Useful for links pivoting at one end (shift by length/2).
        """
        cube = vtk.vtkCubeSource()
        cube.SetXLength(length)
        cube.SetYLength(height)
        cube.SetZLength(depth)
        
        # Apply local shift if needed (e.g. for pivoting)
        tf = vtk.vtkTransform()
        tf.Translate(shift_x, 0, 0)
        
        tf_filter = vtk.vtkTransformPolyDataFilter()
        tf_filter.SetInputConnection(cube.GetOutputPort())
        tf_filter.SetTransform(tf)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tf_filter.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        
        return actor

    def create_sphere_actor(self, radius, color):
        """
        Creates a high-resolution sphere actor (smooth, no visible edges).
        """
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(radius)
        # Increase resolution so it looks round, not faceted
        sphere.SetThetaResolution(50) 
        sphere.SetPhiResolution(50)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        return actor

    def create_environment(self):
        # --- Geometry Setup ---
        
        # 1. Left Arm Links
        # Link 1: Pivots at Base. Shift geometry by +l1/2 so origin is at the start.
        self.actor_L1 = self.create_geometry_actor(self.l1, self.link_width, self.link_width, 
                                                   [0.1, 0.1, 0.6], shift_x=self.l1/2)
        
        # Link 2: Pivots at Elbow. Shift geometry by +l2/2.
        self.actor_L2 = self.create_geometry_actor(self.l2, self.link_width, self.link_width, 
                                                   [0.2, 0.2, 0.7], shift_x=self.l2/2)
        
        # EE1: Attached at Wrist. Centered geometry.
        self.actor_EE1 = self.create_geometry_actor(self.w_EE, self.h_EE, self.w_EE, 
                                                    [0.3, 0.3, 0.9], shift_x=0.0)

        # 2. Right Arm Links
        self.actor_R1 = self.create_geometry_actor(self.l1, self.link_width, self.link_width, 
                                                   [0.6, 0.1, 0.1], shift_x=self.l1/2)
        self.actor_R2 = self.create_geometry_actor(self.l2, self.link_width, self.link_width, 
                                                   [0.7, 0.2, 0.2], shift_x=self.l2/2)
        self.actor_EE2 = self.create_geometry_actor(self.w_EE, self.h_EE, self.w_EE, 
                                                     [0.9, 0.3, 0.3], shift_x=0.0)

        # 3. Box
        self.actor_box = self.create_geometry_actor(self.w_box, self.h_box, self.w_box, 
                                                    [144 / 255, 213 / 255, 255 / 255])
        

        # 4. Floor
        self.actor_floor = self.create_geometry_actor(4.0, 0.02, 2.0, [0.8, 0.8, 0.8])
        floor_tf = vtk.vtkTransform()
        floor_tf.Translate(0, -0.01, 0)
        self.actor_floor.SetUserTransform(floor_tf)

        # 5. Base Spheres (Static)
        # Made slightly larger and smoother
        self.actor_BaseL = self.create_sphere_actor(0.06, [0.1, 0.1, 0.1]) # 
        self.actor_BaseL.SetPosition(self.sys.x_base_L, self.sys.y_base_L, 0)

        self.actor_BaseR = self.create_sphere_actor(0.06, [0.1, 0.1, 0.1])
        self.actor_BaseR.SetPosition(self.sys.x_base_R, self.sys.y_base_R, 0)

        # 6. Joint Spheres (Dynamic)
        # These will represent the rotational joints at Elbows and Wrists
        joint_radius = 0.04
        # joint_color_L = [0.2, 0.2, 0.7] # Match Left arm theme
        # joint_color_R = [0.7, 0.2, 0.2] # Match Right arm theme
        joint_color_L = [0.0, 0.0, 0.0] # Black for visibility
        joint_color_R = [0.0, 0.0, 0.0] # Black for visibility 
        # Left Elbow (Located at origin of Link 2)
        self.actor_joint_L2 = self.create_sphere_actor(joint_radius, joint_color_L)
        # Left Wrist (Located at origin of EE)
        self.actor_joint_EE1 = self.create_sphere_actor(joint_radius/2, joint_color_L)

        # Right Elbow
        self.actor_joint_R2 = self.create_sphere_actor(joint_radius, joint_color_R)
        # Right Wrist
        self.actor_joint_EE2 = self.create_sphere_actor(joint_radius/2, joint_color_R)

        # --- Link Matrices to Actors ---
        def link_matrix(actor, matrix):
            t = vtk.vtkMatrixToLinearTransform()
            t.SetInput(matrix)
            
            # For spheres, we didn't use a TransformPolyDataFilter in creation,
            # so we must check if the actor already has a filter chain or just a mapper.
            # Simpler approach for VTK dynamic updates: Just set the UserMatrix.
            # However, to keep consistent with your previous code structure (Filter pipeline):
            
            # 1. Get current input connection
            prev_out = actor.GetMapper().GetInputConnection(0, 0)
            
            # 2. Create Dynamic Transform Filter
            dyn_filter = vtk.vtkTransformPolyDataFilter()
            dyn_filter.SetInputConnection(prev_out)
            dyn_filter.SetTransform(t)
            
            # 3. Re-connect Mapper
            actor.GetMapper().SetInputConnection(dyn_filter.GetOutputPort())

        # Link Geometry to Kinematic Matrices
        link_matrix(self.actor_L1, self.H_L1)
        link_matrix(self.actor_L2, self.H_L2)
        link_matrix(self.actor_EE1, self.H_EE1)
        
        link_matrix(self.actor_R1, self.H_R1)
        link_matrix(self.actor_R2, self.H_R2)
        link_matrix(self.actor_EE2, self.H_EE2)
        
        link_matrix(self.actor_box, self.H_box)

        # Link the new Joint Spheres to the same matrices
        # Elbow Sphere tracks Link 2's origin (which IS the elbow)
        link_matrix(self.actor_joint_L2, self.H_L2) 
        link_matrix(self.actor_joint_R2, self.H_R2)

        # Wrist Sphere tracks EE's origin (which IS the wrist)
        link_matrix(self.actor_joint_EE1, self.H_EE1)
        link_matrix(self.actor_joint_EE2, self.H_EE2)

        # 7. Text
        self.text_actor = vtk.vtkTextActor()
        self.text_actor.SetInput("t = 0.00 s")
        txt_prop = self.text_actor.GetTextProperty()
        txt_prop.SetFontSize(30)
        txt_prop.SetColor(0, 0, 0)
        txt_prop.SetFontFamilyToArial()
        self.text_actor.SetPosition(30, 30)

        # --- Renderer & Window ---
        self.renderer = vtk.vtkRenderer()
        
        actors_list = [
            self.actor_L1, self.actor_L2, self.actor_EE1,
            self.actor_R1, self.actor_R2, self.actor_EE2,
            self.actor_box, self.actor_floor, self.text_actor,
            self.actor_BaseL, self.actor_BaseR,
            self.actor_joint_L2, self.actor_joint_EE1,
            self.actor_joint_R2, self.actor_joint_EE2
        ]
        
        for a in actors_list:
            self.renderer.AddActor(a)
            
        self.renderer.SetBackground(1, 1, 1)

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetWindowName("Dual Arm Manipulation")

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

        self.cam_widget = vtk.vtkCameraOrientationWidget()
        self.cam_widget.SetParentRenderer(self.renderer)
        self.cam_widget.On()

        # Camera
        camera = self.renderer.GetActiveCamera()
        # camera.SetPosition(0, 1.0, 3.5)
        # camera.SetFocalPoint(0, 0.5, 0.0)
        # camera.SetViewUp(0, 1, 0)
        # Camera
        
        # 1. Position: Center X and Y at 0, move Z far out (e.g., 5.0)
        camera.SetPosition(0, 0, 5.0) 
        
        # 2. Focal Point: Look at the origin (center of your workspace)
        camera.SetFocalPoint(0, 0, 0) 
        
        # 3. View Up: Ensure the Y-axis points "up" on the screen
        camera.SetViewUp(0, 1, 0)
    def update_matrix(self, vtk_matrix, x, y, phi):
        """
        Updates the 4x4 VTK matrix for a 2D rigid body transformation
        (Rotation about Z, Translation in X, Y).
        """
        c = np.cos(phi)
        s = np.sin(phi)

        # Col 0 (X-axis)
        vtk_matrix.SetElement(0, 0, c)
        vtk_matrix.SetElement(1, 0, s)
        vtk_matrix.SetElement(2, 0, 0)
        vtk_matrix.SetElement(3, 0, 0)

        # Col 1 (Y-axis)
        vtk_matrix.SetElement(0, 1, -s)
        vtk_matrix.SetElement(1, 1, c)
        vtk_matrix.SetElement(2, 1, 0)
        vtk_matrix.SetElement(3, 1, 0)

        # Col 2 (Z-axis Identity)
        vtk_matrix.SetElement(0, 2, 0)
        vtk_matrix.SetElement(1, 2, 0)
        vtk_matrix.SetElement(2, 2, 1)
        vtk_matrix.SetElement(3, 2, 0)

        # Col 3 (Translation)
        vtk_matrix.SetElement(0, 3, x)
        vtk_matrix.SetElement(1, 3, y)
        vtk_matrix.SetElement(2, 3, 0) 
        vtk_matrix.SetElement(3, 3, 1)

    def set_scene_to_timestep(self, i):
        q_t = self.q[:, i]
        
        # Calculate FK using the system class (returns Dict of JAX arrays)
        fk_jax = self.sys.get_forward_kinematics(q_t)
        
        # --- Left Arm ---
        # Link 1: Base -> Elbow
        self.update_matrix(self.H_L1, 
                           self.sys.x_base_L, self.sys.y_base_L, 
                           q_t[0])
        
        # Link 2: Elbow -> Wrist
        # Pos: Elbow Location. Angle: q[0] + q[1]
        self.update_matrix(self.H_L2,
                           fk_jax['joint_L2'][0], fk_jax['joint_L2'][1],
                           q_t[0] + q_t[1])
                            
        # EE1: Wrist
        # Pos: Wrist Location. Angle: q[0] + q[1] + q[2]
        self.update_matrix(self.H_EE1,
                           fk_jax['EE1'][0], fk_jax['EE1'][1],
                           q_t[0] + q_t[1] + q_t[2])

        # --- Right Arm ---
        # Link 1
        self.update_matrix(self.H_R1, 
                           self.sys.x_base_R, self.sys.y_base_R, 
                           q_t[3])
        
        # Link 2
        self.update_matrix(self.H_R2,
                           fk_jax['joint_R2'][0], fk_jax['joint_R2'][1],
                           q_t[3] + q_t[4])
        
        # EE2
        self.update_matrix(self.H_EE2,
                           fk_jax['EE2'][0], fk_jax['EE2'][1],
                           q_t[3] + q_t[4] + q_t[5])

        # --- Box ---
        self.update_matrix(self.H_box,
                           fk_jax['box'][0], fk_jax['box'][1],
                           q_t[8])

        self.H_L1.Modified()
        self.H_L2.Modified()
        self.H_EE1.Modified()
        self.H_R1.Modified()
        self.H_R2.Modified()
        self.H_EE2.Modified()
        self.H_box.Modified()

    def write_video_frame(self):
        if self.recording and self.video_writer is not None:
            self.window_to_image_filter.Modified()
            self.window_to_image_filter.Update()
            vtk_image = self.window_to_image_filter.GetOutput()
            width, height, _ = vtk_image.GetDimensions()
            vtk_array = vtk_image.GetPointData().GetScalars()
            components = vtk_array.GetNumberOfComponents()
            arr = numpy_support.vtk_to_numpy(vtk_array).reshape(
                height, width, components
            )
            arr = np.flip(arr, 0)
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            self.video_writer.write(arr)

    def update_scene_callback(self, interactor, event):
        now = time.time()
        elapsed = now - self.start_time_wall_clock
        time_in_sim = elapsed % self.total_duration
        idx = int(time_in_sim / self.dt)
        if idx >= self.N:
            idx = self.N - 1

        self.timestep_index = idx
        self.text_actor.SetInput(f"t = {time_in_sim:.2f} s")
        self.set_scene_to_timestep(self.timestep_index)
        self.render_window.Render()

    def animate(
        self,
        save_video=False,
        filename="dual_arm_manipulation.mp4",
        resolution=(1920, 1080),
        bitrate=4000000,
        fullscreen=False,
    ):

        self.create_environment()

        if fullscreen:
            self.render_window.FullScreenOn()
        else:
            self.render_window.SetSize(resolution[0], resolution[1])
            self.render_window.SetPosition(0, 0)

        self.recording = save_video

        # -----------------------------------------------------------
        # BRANCH A: OFFLINE RENDER
        # -----------------------------------------------------------
        if self.recording:
            current_size = self.render_window.GetSize()
            print(f"--- STARTING SMOOTH OFFLINE RENDER ---")
            
            self.window_to_image_filter = vtk.vtkWindowToImageFilter()
            self.window_to_image_filter.SetInput(self.render_window)
            self.window_to_image_filter.SetInputBufferTypeToRGB()
            self.window_to_image_filter.ReadFrontBufferOff()

            video_fps = 60
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                filename, fourcc, video_fps, current_size, isColor=True
            )
            self.video_writer.set(cv2.CAP_PROP_BITRATE, bitrate)

            total_sim_time = self.N * self.dt
            total_video_frames = int(total_sim_time * video_fps)

            for k in range(total_video_frames):
                t_target = k / video_fps
                idx = int(round(t_target / self.dt))
                if idx >= self.N:
                    idx = self.N - 1

                self.timestep_index = idx
                self.text_actor.SetInput(f"t = {t_target:.2f} s")
                self.set_scene_to_timestep(idx)

                self.render_window.Render()
                self.write_video_frame()

                if k % max(1, (total_video_frames // 10)) == 0:
                    percent = (k / total_video_frames) * 100
                    print(f"Rendering: {percent:.0f}% complete")

            self.video_writer.release()
            print("--- VIDEO SAVED SUCCESSFULLY ---")
            self.render_window.Finalize()
            self.interactor.TerminateApp()

        # -----------------------------------------------------------
        # BRANCH B: LIVE INTERACTIVE PREVIEW
        # -----------------------------------------------------------
        else:
            print("--- STARTING LIVE PREVIEW (REAL-TIME) ---")
            print("Press 'q' to quit.")
            self.set_scene_to_timestep(0)
            self.render_window.Render()

            self.start_time_wall_clock = time.time()
            self.interactor.AddObserver(
                vtk.vtkCommand.TimerEvent, self.update_scene_callback
            )
            self.interactor.CreateRepeatingTimer(16)

            self.interactor.Initialize()
            self.interactor.Start()

if __name__ == "__main__":
    pass