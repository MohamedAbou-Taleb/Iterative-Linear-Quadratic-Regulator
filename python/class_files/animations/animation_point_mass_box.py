import vtk
import numpy as np
import cv2
import time
from vtk.util import numpy_support


class AnimationPointMassBox:

    def __init__(self, sys, X_data, tspan, dt):
        self.sys = sys
        self.ball_radius = sys.ball_radius
        self.box_width = sys.box_width
        self.box_height = sys.box_height

        # Visualization depth (Z-axis thickness for 2D objects)
        self.depth = 0.1

        self.X_data = X_data
        # q indices: 0,1 (Ball1); 2,3 (Ball2); 4,5 (Box)
        self.q = self.X_data[0:6, :]
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
        self.r_ball1, self.r_ball2, self.r_box = self.compute_all_positions(self.q)
        print("Computation complete.")

        # --- VTK object placeholders ---
        self.renderer = None
        self.render_window = None
        self.interactor = None

        # Transformation matrices
        self.H_ball1 = vtk.vtkMatrix4x4()
        self.H_ball2 = vtk.vtkMatrix4x4()
        self.H_box = vtk.vtkMatrix4x4()
        self.H_floor = vtk.vtkMatrix4x4()

    def compute_all_positions(self, q):
        """
        Extracts position data from state vector q.
        q structure: [x_b1, y_b1, x_b2, y_b2, x_box, y_box]
        """
        # Ball 1 (Indices 0, 1)
        x_b1 = q[0, :]
        y_b1 = q[1, :]
        r_ball1 = np.vstack([x_b1, y_b1, np.zeros(self.N)])

        # Ball 2 (Indices 2, 3)
        x_b2 = q[2, :]
        y_b2 = q[3, :]
        r_ball2 = np.vstack([x_b2, y_b2, np.zeros(self.N)])

        # Box (Indices 4, 5)
        x_box = q[4, :]
        y_box = q[5, :]
        r_box = np.vstack([x_box, y_box, np.zeros(self.N)])

        return r_ball1, r_ball2, r_box

    def create_environment(self):
        # --- Geometry Setup ---

        # 1. Ball 1 (Left) - Blue
        sphere_1 = vtk.vtkSphereSource()
        sphere_1.SetRadius(self.ball_radius)
        sphere_1.SetThetaResolution(32)
        sphere_1.SetPhiResolution(32)

        _H_ball1 = vtk.vtkMatrixToLinearTransform()
        _H_ball1.SetInput(self.H_ball1)
        tf_filter_1 = vtk.vtkTransformPolyDataFilter()
        tf_filter_1.SetInputConnection(sphere_1.GetOutputPort())
        tf_filter_1.SetTransform(_H_ball1)

        mapper_1 = vtk.vtkPolyDataMapper()
        mapper_1.SetInputConnection(tf_filter_1.GetOutputPort())
        actor_ball1 = vtk.vtkActor()
        actor_ball1.SetMapper(mapper_1)
        actor_ball1.GetProperty().SetColor([57 / 255, 49 / 255, 133 / 255])  # Deep Blue

        # 2. Ball 2 (Right) - Red
        sphere_2 = vtk.vtkSphereSource()
        sphere_2.SetRadius(self.ball_radius)
        sphere_2.SetThetaResolution(32)
        sphere_2.SetPhiResolution(32)

        _H_ball2 = vtk.vtkMatrixToLinearTransform()
        _H_ball2.SetInput(self.H_ball2)
        tf_filter_2 = vtk.vtkTransformPolyDataFilter()
        tf_filter_2.SetInputConnection(sphere_2.GetOutputPort())
        tf_filter_2.SetTransform(_H_ball2)

        mapper_2 = vtk.vtkPolyDataMapper()
        mapper_2.SetInputConnection(tf_filter_2.GetOutputPort())
        actor_ball2 = vtk.vtkActor()
        actor_ball2.SetMapper(mapper_2)
        actor_ball2.GetProperty().SetColor([199 / 255, 33 / 255, 37 / 255])  # Red

        # 3. Box - Green/Grey
        box_geom = vtk.vtkCubeSource()
        box_geom.SetXLength(self.box_width)
        box_geom.SetYLength(self.box_height)
        box_geom.SetZLength(self.box_width)  # Make it a cube depth-wise

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
        )  # Dark Green

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
        self.renderer.AddActor(actor_ball1)
        self.renderer.AddActor(actor_ball2)
        self.renderer.AddActor(actor_box)
        self.renderer.AddActor(actor_floor)
        self.renderer.AddActor(self.text_actor)
        self.renderer.SetBackground(1, 1, 1)

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetWindowName("Box and Point Mass Manipulation")

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

        self.cam_widget = vtk.vtkCameraOrientationWidget()
        self.cam_widget.SetParentRenderer(self.renderer)
        self.cam_widget.On()

        # Initial camera setup
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(0, 1.5, 5.0)
        camera.SetFocalPoint(0, 0.1, 0.0)

    def set_scene_to_timestep(self, i):
        # Helper to update matrices (Only Translation for point masses/box in this model)
        p_b1 = self.r_ball1[:, i]
        p_b2 = self.r_ball2[:, i]
        p_box = self.r_box[:, i]

        # Update Ball 1 Matrix
        self.H_ball1.SetElement(0, 3, p_b1[0])
        self.H_ball1.SetElement(1, 3, p_b1[1])
        self.H_ball1.SetElement(2, 3, p_b1[2])

        # Update Ball 2 Matrix
        self.H_ball2.SetElement(0, 3, p_b2[0])
        self.H_ball2.SetElement(1, 3, p_b2[1])
        self.H_ball2.SetElement(2, 3, p_b2[2])

        # Update Box Matrix
        self.H_box.SetElement(0, 3, p_box[0])
        self.H_box.SetElement(1, 3, p_box[1])
        self.H_box.SetElement(2, 3, p_box[2])

        self.H_ball1.Modified()
        self.H_ball2.Modified()
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
        filename="box_manipulation.mp4",
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
    class MyBoxSystem:
        def __init__(self):
            self.ball_radius = 0.05
            self.box_width = 0.5
            self.box_height = 0.3

    sys_dummy = MyBoxSystem()

    T = 5.0
    dt = 0.01
    tspan = np.arange(0, T + dt, dt)
    N = len(tspan)

    # Create dummy trajectory
    # Box slides back and forth
    x_box = 0.5 * np.sin(tspan)
    y_box = np.ones(N) * (sys_dummy.box_height / 2)  # Sliding on floor

    # Ball 1 (Left) bounces
    x_b1 = x_box - sys_dummy.box_width / 2 - sys_dummy.ball_radius - 0.1
    y_b1 = sys_dummy.ball_radius + 0.2 * np.abs(np.cos(2 * tspan))

    # Ball 2 (Right) orbits slightly
    x_b2 = x_box + sys_dummy.box_width / 2 + sys_dummy.ball_radius + 0.1
    y_b2 = sys_dummy.ball_radius + 0.1 * np.sin(3 * tspan)

    # Construct state [q1x, q1y, q2x, q2y, bx, by]
    q = np.vstack([x_b1, y_b1, x_b2, y_b2, x_box, y_box])
    # Pad velocity (zeros) to make it look like full state X
    v = np.zeros_like(q)
    X_data = np.vstack([q, v])

    anim = AnimationPointMassBox(sys_dummy, X_data, tspan, dt)

    # Run Live Preview
    anim.animate(save_video=False, filename="box_test.mp4", fullscreen=True)
