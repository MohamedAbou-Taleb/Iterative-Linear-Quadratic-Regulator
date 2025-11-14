import vtk
import numpy as np
import cv2
import time  # <--- Added for real-time tracking
from vtk.util import numpy_support

class AnimationDoublePendulum:
    
    def __init__(self, double_pendulum_sys, X_data, tspan, dt):
        self.double_pendulum_sys = double_pendulum_sys
        self.l1 = double_pendulum_sys.l1
        self.l2 = double_pendulum_sys.l2
        self.width1 = self.l1 * 0.05
        self.width2 = self.l2 * 0.05
        self.height1 = self.l1 * 0.05
        self.height2 = self.l2 * 0.05
        
        self.X_data = X_data
        self.q = self.X_data[0:2, :]
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
        self.start_time_wall_clock = None # <--- Stores when animation started

        # --- Pre-compute trajectories ---
        print("Pre-computing trajectories...")
        self.I_r_os_1, self.I_r_os_2, self.I_r_oj_1, self.I_r_oj_2, self.A_IB_1, self.A_IB_2 = self.compute_all_positions_and_orientations(self.q)
        print("Computation complete.")

        # --- VTK object placeholders ---
        self.renderer = None
        self.render_window = None
        self.interactor = None
        self.H_IB_1 = vtk.vtkMatrix4x4()
        self.H_IB_2 = vtk.vtkMatrix4x4()
        self.H_IB_joint_1 = vtk.vtkMatrix4x4()
        self.H_IB_joint_2 = vtk.vtkMatrix4x4()
        self.H_IB_floor = vtk.vtkMatrix4x4()

    def compute_positions(self, q):
        q1 = q[0, :]
        q2 = q[1, :]
        x1 = 0.5*self.l1 * np.sin(q1)
        y1 = -0.5*self.l1 * np.cos(q1)
        x2 = 2*x1 + 0.5*self.l2 * np.sin(q1 + q2)
        y2 = 2*y1 - 0.5*self.l2 * np.cos(q1+ q2)
        return np.array([x1, y1, np.zeros(len(x1))]), np.array([x2, y2, np.zeros(len(x2))])
    
    def compute_joint_positions(self, q):
        q1 = q[0, :]
        q2 = q[1, :]
        x1 = self.l1 * np.sin(q1)
        y1 = -self.l1 * np.cos(q1)
        x2 = x1 + self.l2 * np.sin(q1 + q2)
        y2 = y1 - self.l2 * np.cos(q1+ q2)
        return np.array([x1, y1, np.zeros(len(x1))]), np.array([x2, y2, np.zeros(len(x2))])
    
    def compute_transformation_matrix(self, q_i):
        return np.array([[np.cos(q_i), -np.sin(q_i), 0.0],
                         [np.sin(q_i),  np.cos(q_i), 0.0],
                         [0.0,          0.0,         1.0]])
    
    def compute_all_positions_and_orientations(self, q):
        I_r_os_1, I_r_os_2 = self.compute_positions(q)
        I_r_oj_1, I_r_oj_2 = self.compute_joint_positions(q)
        A_IB_1 = np.array([self.compute_transformation_matrix(q_i) for q_i in q[0, :]])
        A_IB_2 = np.array([self.compute_transformation_matrix(q_i[0] + q_i[1]) for q_i in q.T])
        return I_r_os_1, I_r_os_2, I_r_oj_1, I_r_oj_2, A_IB_1, A_IB_2
    
    def create_environment(self):
            # --- Geometry Setup ---
            # Pendulum 1
            pend_1 = vtk.vtkCubeSource()
            pend_1.SetXLength(self.width1); pend_1.SetYLength(self.l1); pend_1.SetZLength(self.height1)
            _H_IB_1 = vtk.vtkMatrixToLinearTransform(); _H_IB_1.SetInput(self.H_IB_1)
            tf_filter_1 = vtk.vtkTransformPolyDataFilter()
            tf_filter_1.SetInputConnection(pend_1.GetOutputPort()); tf_filter_1.SetTransform(_H_IB_1)
            mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(tf_filter_1.GetOutputPort())
            actor_1 = vtk.vtkActor(); actor_1.SetMapper(mapper); actor_1.GetProperty().SetColor([57/255, 49/255, 133/255]) 

            # Floor
            floor = vtk.vtkCubeSource()
            floor.SetXLength(0.2); floor.SetYLength(0.05); floor.SetZLength(0.2)
            _H_IB_floor = vtk.vtkMatrixToLinearTransform(); _H_IB_floor.SetInput(self.H_IB_floor)
            tf_filter_floor = vtk.vtkTransformPolyDataFilter()
            tf_filter_floor.SetInputConnection(floor.GetOutputPort()); tf_filter_floor.SetTransform(_H_IB_floor)
            mapper_floor = vtk.vtkPolyDataMapper(); mapper_floor.SetInputConnection(tf_filter_floor.GetOutputPort())
            actor_floor = vtk.vtkActor(); actor_floor.SetMapper(mapper_floor); actor_floor.GetProperty().SetColor([0,0,0])  

            # Pendulum 2
            pend_2 = vtk.vtkCubeSource()
            pend_2.SetXLength(self.width2); pend_2.SetYLength(self.l2); pend_2.SetZLength(self.height2)
            _H_IB_2 = vtk.vtkMatrixToLinearTransform(); _H_IB_2.SetInput(self.H_IB_2)
            tf_filter_2 = vtk.vtkTransformPolyDataFilter()
            tf_filter_2.SetInputConnection(pend_2.GetOutputPort()); tf_filter_2.SetTransform(_H_IB_2)
            mapper2 = vtk.vtkPolyDataMapper(); mapper2.SetInputConnection(tf_filter_2.GetOutputPort())
            actor_2 = vtk.vtkActor(); actor_2.SetMapper(mapper2); actor_2.GetProperty().SetColor([199/255, 33/255, 37/255]) 

            # Joint 1
            joint_sphere_1 = vtk.vtkSphereSource()
            joint_sphere_1.SetRadius(self.width1 * 1.2); joint_sphere_1.SetThetaResolution(16); joint_sphere_1.SetPhiResolution(16)
            _H_IB_joint_1 = vtk.vtkMatrixToLinearTransform(); _H_IB_joint_1.SetInput(self.H_IB_joint_1)
            tf_filter_joint_1 = vtk.vtkTransformPolyDataFilter()
            tf_filter_joint_1.SetInputConnection(joint_sphere_1.GetOutputPort()); tf_filter_joint_1.SetTransform(_H_IB_joint_1)
            joint_mapper_1 = vtk.vtkPolyDataMapper(); joint_mapper_1.SetInputConnection(tf_filter_joint_1.GetOutputPort())
            joint_actor_1 = vtk.vtkActor(); joint_actor_1.SetMapper(joint_mapper_1); joint_actor_1.GetProperty().SetColor([0, 0, 0]) 

            # Joint 2
            joint_sphere_2 = vtk.vtkSphereSource()
            joint_sphere_2.SetRadius(self.width2 * 1.2); joint_sphere_2.SetThetaResolution(16); joint_sphere_2.SetPhiResolution(16)
            _H_IB_joint_2 = vtk.vtkMatrixToLinearTransform(); _H_IB_joint_2.SetInput(self.H_IB_joint_2)
            tf_filter_joint_2 = vtk.vtkTransformPolyDataFilter()
            tf_filter_joint_2.SetInputConnection(joint_sphere_2.GetOutputPort()); tf_filter_joint_2.SetTransform(_H_IB_joint_2)
            joint_mapper_2 = vtk.vtkPolyDataMapper(); joint_mapper_2.SetInputConnection(tf_filter_joint_2.GetOutputPort())
            joint_actor_2 = vtk.vtkActor(); joint_actor_2.SetMapper(joint_mapper_2); joint_actor_2.GetProperty().SetColor([0, 0, 0]) 
            
            # Text
            self.text_actor = vtk.vtkTextActor()
            self.text_actor.SetInput("t = 0.00 s")
            txt_prop = self.text_actor.GetTextProperty()
            txt_prop.SetFontSize(30); txt_prop.SetColor(0, 0, 0); txt_prop.SetFontFamilyToArial()
            self.text_actor.SetPosition(30, 30) 

            # --- Renderer & Window ---
            self.renderer = vtk.vtkRenderer()
            self.renderer.AddActor(actor_1); self.renderer.AddActor(actor_2); self.renderer.AddActor(actor_floor)
            self.renderer.AddActor(joint_actor_1); self.renderer.AddActor(joint_actor_2); self.renderer.AddActor(self.text_actor)
            self.renderer.SetBackground(1,1,1) 
            
            self.render_window = vtk.vtkRenderWindow()
            self.render_window.AddRenderer(self.renderer)
            
            self.interactor = vtk.vtkRenderWindowInteractor()
            self.interactor.SetRenderWindow(self.render_window)
            
            self.cam_widget = vtk.vtkCameraOrientationWidget()
            self.cam_widget.SetParentRenderer(self.renderer)
            self.cam_widget.On()
            camera = self.renderer.GetActiveCamera()
            camera.SetPosition(0, 1, 8)

    def set_scene_to_timestep(self, i):
        # Helper to update matrices
        r1, rj1, A1 = self.I_r_os_1[:, i], self.I_r_oj_1[:, i], self.A_IB_1[i, :, :]
        r2, rj2, A2 = self.I_r_os_2[:, i], self.I_r_oj_2[:, i], self.A_IB_2[i, :, :]

        for k in range(3):
            self.H_IB_1.SetElement(k, 3, r1[k]); self.H_IB_joint_1.SetElement(k, 3, rj1[k])
            self.H_IB_2.SetElement(k, 3, r2[k]); self.H_IB_joint_2.SetElement(k, 3, rj2[k])
            for j in range(3):
                self.H_IB_1.SetElement(k, j, A1[k, j]); self.H_IB_joint_1.SetElement(k, j, A1[k, j])
                self.H_IB_2.SetElement(k, j, A2[k, j]); self.H_IB_joint_2.SetElement(k, j, A2[k, j])
        
        self.H_IB_1.Modified(); self.H_IB_joint_1.Modified()
        self.H_IB_2.Modified(); self.H_IB_joint_2.Modified()
    
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

    def update_scene_callback(self, interactor, event):
        """
        REAL-TIME CALLBACK
        Checks the wall-clock time and jumps to the correct simulation index.
        This skips frames if rendering is slow, ensuring real-time speed.
        """
        # 1. Calculate elapsed real-world time
        now = time.time()
        elapsed = now - self.start_time_wall_clock
        
        # 2. Wrap around logic (looping)
        # "elapsed % self.total_duration" keeps the time between 0 and T
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

    def animate(self, save_video=False, filename="double_pendulum.mp4", 
                resolution=(1920, 1080), bitrate=4000000, fullscreen=False):
        
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
            print(f"Output: {filename} | Resolution: {current_size[0]}x{current_size[1]}")
            
            self.window_to_image_filter = vtk.vtkWindowToImageFilter()
            self.window_to_image_filter.SetInput(self.render_window)
            self.window_to_image_filter.SetInputBufferTypeToRGB()
            self.window_to_image_filter.ReadFrontBufferOff()
            
            video_fps = 60 
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            self.video_writer = cv2.VideoWriter(filename, fourcc, video_fps, 
                                              current_size, isColor=True)
            self.video_writer.set(cv2.CAP_PROP_BITRATE, bitrate) 

            total_sim_time = self.N * self.dt
            total_video_frames = int(total_sim_time * video_fps)
            
            print(f"Resampling: {self.N} sim steps -> {total_video_frames} video frames.")
            
            for k in range(total_video_frames):
                t_target = k / video_fps
                idx = int(round(t_target / self.dt))
                if idx >= self.N: idx = self.N - 1

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
            self.interactor.AddObserver(vtk.vtkCommand.TimerEvent, self.update_scene_callback)
            
            # 3. Set Timer to ~60 FPS (16ms)
            # We don't use 'dt' here. We update the screen at 60Hz, 
            # and the callback figures out the physics index mathematically.
            self.interactor.CreateRepeatingTimer(16) 
            
            self.interactor.Initialize()
            self.interactor.Start()

if __name__ == "__main__":
    class MyDoublePendulum:
        def __init__(self, dt, g):
            self.l1 = 1.0; self.l2 = 1.0; self.dt = dt; self.g = g

    import numpy as jnp 
    pend = MyDoublePendulum(dt=0.01, g=9.81)
    
    T = 5.0
    dt = 0.01
    tspan = jnp.arange(0, T + dt, dt)
    N = len(tspan)
    q1 = jnp.pi * jnp.sin(0.5 * tspan)
    q2 = jnp.pi * jnp.cos(0.5 * tspan)
    X_data = np.array([q1, q2, np.zeros(N), np.zeros(N)]) 

    anim = AnimationDoublePendulum(pend, X_data, tspan, dt)
    
    # --- CONFIGURATION ---
    # save_video=False -> Full Real-Time Live View
    # save_video=True  -> High Quality Smooth Rendering (Slow process, smooth file)
    anim.animate(save_video=True, 
                 filename="final_pendulum.mp4", 
                 fullscreen=True)