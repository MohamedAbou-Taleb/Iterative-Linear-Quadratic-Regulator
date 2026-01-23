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

    def create_environment(self):
        # --- Geometry Setup ---
        
        # 1. Left Arm
        # Link 1: Pivots at Base. Shift geometry by +l1/2 so origin is at the start.
        self.actor_L1 = self.create_geometry_actor(self.l1, self.link_width, self.link_width, 
                                                   [0.1, 0.1, 0.6], shift_x=self.l1/2)
        
        # Link 2: Pivots at Elbow. Shift geometry by +l2/2.
        self.actor_L2 = self.create_geometry_actor(self.l2, self.link_width, self.link_width, 
                                                   [0.2, 0.2, 0.7], shift_x=self.l2/2)
        
        # EE1: Attached at Wrist. Centered geometry (shift=0) or adjust as needed.
        self.actor_EE1 = self.create_geometry_actor(self.w_EE, self.h_EE, self.w_EE, 
                                                    [0.3, 0.3, 0.9], shift_x=0.0)

        # 2. Right Arm
        self.actor_R1 = self.create_geometry_actor(self.l1, self.link_width, self.link_width, 
                                                   [0.6, 0.1, 0.1], shift_x=self.l1/2)
        self.actor_R2 = self.create_geometry_actor(self.l2, self.link_width, self.link_width, 
                                                   [0.7, 0.2, 0.2], shift_x=self.l2/2)
        self.actor_EE2 = self.create_geometry_actor(self.w_EE, self.h_EE, self.w_EE, 
                                                    [0.9, 0.3, 0.3], shift_x=0.0)

        # 3. Box
        self.actor_box = self.create_geometry_actor(self.w_box, self.h_box, self.w_box, 
                                                    [0.4, 0.8, 0.4])

        # 4. Floor
        self.actor_floor = self.create_geometry_actor(4.0, 0.02, 2.0, [0.8, 0.8, 0.8])
        # Static shift for floor down
        floor_tf = vtk.vtkTransform()
        floor_tf.Translate(0, -0.01, 0)
        self.actor_floor.SetUserTransform(floor_tf)

        # 5. Base Markers
        sphere_L = vtk.vtkSphereSource()
        sphere_L.SetRadius(0.05)
        mapper_SL = vtk.vtkPolyDataMapper()
        mapper_SL.SetInputConnection(sphere_L.GetOutputPort())
        actor_SL = vtk.vtkActor()
        actor_SL.SetMapper(mapper_SL)
        actor_SL.SetPosition(self.sys.x_base_L, self.sys.y_base_L, 0)
        actor_SL.GetProperty().SetColor([0,0,0])

        sphere_R = vtk.vtkSphereSource()
        sphere_R.SetRadius(0.05)
        mapper_SR = vtk.vtkPolyDataMapper()
        mapper_SR.SetInputConnection(sphere_R.GetOutputPort())
        actor_SR = vtk.vtkActor()
        actor_SR.SetMapper(mapper_SR)
        actor_SR.SetPosition(self.sys.x_base_R, self.sys.y_base_R, 0)
        actor_SR.GetProperty().SetColor([0,0,0])

        # --- Link Matrices to Actors ---
        def link_matrix(actor, matrix):
            t = vtk.vtkMatrixToLinearTransform()
            t.SetInput(matrix)
            tf = vtk.vtkTransformPolyDataFilter()
            # Chain the previous filter (geometry) to this new transform
            # Note: The actor already has a mapper connected to 'create_geometry_actor' output.
            # To update the matrix dynamically, simpler to set UserMatrix on the actor directly.
            # But 'vtkMatrixToLinearTransform' is good for filter chains.
            # Here we follow the style of AnimationSurfaceBox:
            # Using vtkTransformPolyDataFilter for the matrix update
            
            # Re-build pipeline for dynamic update:
            # Source -> Shift(static) -> MatrixTransform(dynamic) -> Mapper -> Actor
            
            # 1. Get the shift filter output (from create_geometry_actor)
            prev_out = actor.GetMapper().GetInputConnection(0, 0)
            
            # 2. Create Dynamic Transform Filter
            dyn_filter = vtk.vtkTransformPolyDataFilter()
            dyn_filter.SetInputConnection(prev_out)
            dyn_filter.SetTransform(t)
            
            # 3. Re-connect Mapper
            actor.GetMapper().SetInputConnection(dyn_filter.GetOutputPort())

        link_matrix(self.actor_L1, self.H_L1)
        link_matrix(self.actor_L2, self.H_L2)
        link_matrix(self.actor_EE1, self.H_EE1)
        link_matrix(self.actor_R1, self.H_R1)
        link_matrix(self.actor_R2, self.H_R2)
        link_matrix(self.actor_EE2, self.H_EE2)
        link_matrix(self.actor_box, self.H_box)

        # 6. Text
        self.text_actor = vtk.vtkTextActor()
        self.text_actor.SetInput("t = 0.00 s")
        txt_prop = self.text_actor.GetTextProperty()
        txt_prop.SetFontSize(30)
        txt_prop.SetColor(0, 0, 0)
        txt_prop.SetFontFamilyToArial()
        self.text_actor.SetPosition(30, 30)

        # --- Renderer & Window ---
        self.renderer = vtk.vtkRenderer()
        actors = [self.actor_L1, self.actor_L2, self.actor_EE1,
                  self.actor_R1, self.actor_R2, self.actor_EE2,
                  self.actor_box, self.actor_floor, self.text_actor,
                  actor_SL, actor_SR]
        
        for a in actors:
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
        camera.SetPosition(0, 1.0, 3.5)
        camera.SetFocalPoint(0, 0.5, 0.0)

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
        # Pos: Base Location. Angle: q[0]
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