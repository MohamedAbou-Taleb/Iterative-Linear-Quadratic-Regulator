import vtk
import numpy as np
import cv2
import time
from vtk.util import numpy_support


class AnimationSurfaceBox:

    def __init__(self, sys, X_data, tspan, dt):
        self.sys = sys
        # Dimensions from the system object (consistent with your Matlab script variables)
        self.w_box = sys.w_box
        self.h_box = sys.h_box
        self.w_EE = sys.w_EE
        self.h_EE = sys.h_EE

        self.X_data = X_data
        # q indices: 
        # 0-2: EE1 (x, y, phi)
        # 3-5: EE2 (x, y, phi)
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
        self.H_EE1 = vtk.vtkMatrix4x4()
        self.H_EE2 = vtk.vtkMatrix4x4()
        self.H_box = vtk.vtkMatrix4x4()

    def create_environment(self):
        # --- Geometry Setup ---

        # 1. Left End Effector (EE1) - Blue
        cube_ee1 = vtk.vtkCubeSource()
        cube_ee1.SetXLength(self.w_EE)
        cube_ee1.SetYLength(self.h_EE)
        cube_ee1.SetZLength(self.w_EE) # Depth

        _H_EE1 = vtk.vtkMatrixToLinearTransform()
        _H_EE1.SetInput(self.H_EE1)
        tf_filter_ee1 = vtk.vtkTransformPolyDataFilter()
        tf_filter_ee1.SetInputConnection(cube_ee1.GetOutputPort())
        tf_filter_ee1.SetTransform(_H_EE1)

        mapper_ee1 = vtk.vtkPolyDataMapper()
        mapper_ee1.SetInputConnection(tf_filter_ee1.GetOutputPort())
        actor_ee1 = vtk.vtkActor()
        actor_ee1.SetMapper(mapper_ee1)
        actor_ee1.GetProperty().SetColor([57 / 255, 49 / 255, 133 / 255])  # Deep Blue

        # 2. Right End Effector (EE2) - Red
        cube_ee2 = vtk.vtkCubeSource()
        cube_ee2.SetXLength(self.w_EE)
        cube_ee2.SetYLength(self.h_EE)
        cube_ee2.SetZLength(self.w_EE)

        _H_EE2 = vtk.vtkMatrixToLinearTransform()
        _H_EE2.SetInput(self.H_EE2)
        tf_filter_ee2 = vtk.vtkTransformPolyDataFilter()
        tf_filter_ee2.SetInputConnection(cube_ee2.GetOutputPort())
        tf_filter_ee2.SetTransform(_H_EE2)

        mapper_ee2 = vtk.vtkPolyDataMapper()
        mapper_ee2.SetInputConnection(tf_filter_ee2.GetOutputPort())
        actor_ee2 = vtk.vtkActor()
        actor_ee2.SetMapper(mapper_ee2)
        actor_ee2.GetProperty().SetColor([199 / 255, 33 / 255, 37 / 255])  # Red

        # 3. Box - Green/Light Blue
        box_geom = vtk.vtkCubeSource()
        box_geom.SetXLength(self.w_box)
        box_geom.SetYLength(self.h_box)
        box_geom.SetZLength(self.w_box) 

        _H_box = vtk.vtkMatrixToLinearTransform()
        _H_box.SetInput(self.H_box)
        tf_filter_box = vtk.vtkTransformPolyDataFilter()
        tf_filter_box.SetInputConnection(box_geom.GetOutputPort())
        tf_filter_box.SetTransform(_H_box)

        mapper_box = vtk.vtkPolyDataMapper()
        mapper_box.SetInputConnection(tf_filter_box.GetOutputPort())
        actor_box = vtk.vtkActor()
        actor_box.SetMapper(mapper_box)
        actor_box.GetProperty().SetColor(
            [144 / 255, 213 / 255, 255 / 255]
        )

        # 4. Floor
        floor = vtk.vtkCubeSource()
        floor.SetXLength(4.0)
        floor.SetYLength(0.02)
        floor.SetZLength(2.0)

        # Static transform for floor (shift down so top surface is at y=0)
        floor_tf = vtk.vtkTransform()
        floor_tf.Translate(0, -0.01, 0)

        tf_filter_floor = vtk.vtkTransformPolyDataFilter()
        tf_filter_floor.SetInputConnection(floor.GetOutputPort())
        tf_filter_floor.SetTransform(floor_tf)

        mapper_floor = vtk.vtkPolyDataMapper()
        mapper_floor.SetInputConnection(tf_filter_floor.GetOutputPort())
        actor_floor = vtk.vtkActor()
        actor_floor.SetMapper(mapper_floor)
        actor_floor.GetProperty().SetColor([0.8, 0.8, 0.8])

        # 5. Text
        self.text_actor = vtk.vtkTextActor()
        self.text_actor.SetInput("t = 0.00 s")
        txt_prop = self.text_actor.GetTextProperty()
        txt_prop.SetFontSize(30)
        txt_prop.SetColor(0, 0, 0)
        txt_prop.SetFontFamilyToArial()
        self.text_actor.SetPosition(30, 30)

        # --- Renderer & Window ---
        self.renderer = vtk.vtkRenderer()
        self.renderer.AddActor(actor_ee1)
        self.renderer.AddActor(actor_ee2)
        self.renderer.AddActor(actor_box)
        self.renderer.AddActor(actor_floor)
        self.renderer.AddActor(self.text_actor)
        self.renderer.SetBackground(1, 1, 1)

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetWindowName("Surface Box Manipulation")

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

        self.cam_widget = vtk.vtkCameraOrientationWidget()
        self.cam_widget.SetParentRenderer(self.renderer)
        self.cam_widget.On()

        # Initial camera setup
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(0, 1.5, 4.0)
        camera.SetFocalPoint(0, 0.2, 0.0)

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
        # Extract states
        # EE1
        x_ee1 = self.q[0, i]
        y_ee1 = self.q[1, i]
        phi_ee1 = self.q[2, i]
        
        # EE2
        x_ee2 = self.q[3, i]
        y_ee2 = self.q[4, i]
        phi_ee2 = self.q[5, i]

        # Box
        x_box = self.q[6, i]
        y_box = self.q[7, i]
        phi_box = self.q[8, i]

        # Update Matrices
        self.update_matrix(self.H_EE1, x_ee1, y_ee1, phi_ee1)
        self.update_matrix(self.H_EE2, x_ee2, y_ee2, phi_ee2)
        self.update_matrix(self.H_box, x_box, y_box, phi_box)

        self.H_EE1.Modified()
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
        # 1. Calculate elapsed real-world time
        now = time.time()
        elapsed = now - self.start_time_wall_clock

        # 2. Wrap around logic (looping)
        time_in_sim = elapsed % self.total_duration

        # 3. Convert time to index
        idx = int(time_in_sim / self.dt)

        # 4. Safety clamp
        if idx >= self.N:
            idx = self.N - 1

        self.timestep_index = idx

        # 5. Update Scene
        self.text_actor.SetInput(f"t = {time_in_sim:.2f} s")
        self.set_scene_to_timestep(self.timestep_index)
        self.render_window.Render()

    def animate(
        self,
        save_video=False,
        filename="surface_box_manipulation.mp4",
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
        # BRANCH A: OFFLINE RENDER (Smooth 60fps Video File)
        # -----------------------------------------------------------
        if self.recording:
            current_size = self.render_window.GetSize()
            print(f"--- STARTING SMOOTH OFFLINE RENDER ---")
            print(
                f"Output: {filename} | Resolution: {current_size[0]}x{current_size[1]}"
            )

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

            print(
                f"Resampling: {self.N} sim steps -> {total_video_frames} video frames."
            )

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
        # BRANCH B: LIVE INTERACTIVE PREVIEW (Real-Time)
        # -----------------------------------------------------------
        else:
            print("--- STARTING LIVE PREVIEW (REAL-TIME) ---")
            print("Press 'q' to quit.")
            self.set_scene_to_timestep(0)
            self.render_window.Render()

            # 1. Set the start time
            self.start_time_wall_clock = time.time()

            # 2. Attach the Real-Time Callback
            self.interactor.AddObserver(
                vtk.vtkCommand.TimerEvent, self.update_scene_callback
            )

            # 3. Set Timer to ~60 FPS (16ms)
            self.interactor.CreateRepeatingTimer(16)

            self.interactor.Initialize()
            self.interactor.Start()


if __name__ == "__main__":
    # --- TEST STUB ---
    class MySystem:
        def __init__(self):
            # Example dimensions
            self.w_EE = 0.1
            self.h_EE = 0.3
            self.w_box = 0.5
            self.h_box = 0.5

    sys_dummy = MySystem()

    T = 5.0
    dt = 0.01
    tspan = np.arange(0, T + dt, dt)
    N = len(tspan)

    # Create dummy trajectory
    # Box rotates and moves up/down
    phi_box = 0.5 * np.sin(2 * tspan)
    x_box = np.zeros(N)
    y_box = 0.5 + 0.2 * np.sin(tspan)

    # EE1 (Left) - Approaches box
    x_ee1 = -0.6 + 0.1 * np.sin(tspan)
    y_ee1 = y_box
    phi_ee1 = np.zeros(N)

    # EE2 (Right) - Approaches box
    x_ee2 = 0.6 - 0.1 * np.sin(tspan)
    y_ee2 = y_box
    phi_ee2 = np.zeros(N)

    # Construct state 
    # [x1, y1, phi1, x2, y2, phi2, xbox, ybox, phibox]
    q = np.vstack([x_ee1, y_ee1, phi_ee1, 
                   x_ee2, y_ee2, phi_ee2, 
                   x_box, y_box, phi_box])
    
    # Pad velocity
    v = np.zeros_like(q)
    X_data = np.vstack([q, v])

    anim = AnimationSurfaceBox(sys_dummy, X_data, tspan, dt)

    # Run Live Preview
    anim.animate(save_video=False, filename="surface_box_test.mp4", fullscreen=False)