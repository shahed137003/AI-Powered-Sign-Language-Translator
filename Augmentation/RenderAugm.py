import os
import numpy as np
import cv2
from pathlib import Path

def render_skeleton(data, canvas_size=(480,640), dynamic=True):
    h, w = canvas_size
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # ---- RESHAPE LANDMARKS ----
    pose = data[0:132].reshape(33, 4)
    face = data[132:312].reshape(-1, 3)
    lh   = data[312:375].reshape(21, 3)
    rh   = data[375:438].reshape(21, 3)

    # ---- COLLECT VALID POINTS ----
    points = []
    for p in pose:
        if p[3] > 0.5 and not np.any(np.isnan(p)):
            points.append(p[:2])
    for hand in [lh, rh]:
        for p in hand:
            if not np.any(np.isnan(p)):
                points.append(p[:2])
    for p in face:
        if not np.any(np.isnan(p)):
            points.append(p[:2])
    points = np.array(points)

    # ---- DYNAMIC SCALING ----
    if dynamic and len(points) > 0:
        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)
        center = (min_xy + max_xy) / 2
        size = (max_xy - min_xy).max()
        if size < 1e-6: size = 1.0
        scale = 300 / size
    else:
        center = np.array([0,0])
        scale = 200

    def to_pixel(coord):
        x = int((coord[0] - center[0]) * scale + w//2)
        y = int((coord[1] - center[1]) * scale + h//2)
        x = max(0, min(w-1, x))
        y = max(0, min(h-1, y))
        return x, y

    # ---- POSE ----
    pose_connections = [
        (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
        (9,10),(11,12),(11,13),(13,15),(15,17),(12,14),(14,16),
        (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
        (27,29),(28,30),(29,31),(30,32)
    ]
    for p1, p2 in pose_connections:
        if pose[p1][3] > 0.5 and pose[p2][3] > 0.5:
            x1, y1 = to_pixel(pose[p1])
            x2, y2 = to_pixel(pose[p2])
            cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)
    for p in pose:
        if p[3] > 0.5 and not np.any(np.isnan(p)):
            x, y = to_pixel(p)
            cv2.circle(img, (x,y), 3, (0,255,0), -1)

    # ---- HANDS ----
    hand_connections = [
        (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20)
    ]
    def draw_hand(hand, color):
        for p1,p2 in hand_connections:
            if p1 >= len(hand) or p2 >= len(hand):
                continue
            if np.any(np.isnan(hand[p1])) or np.any(np.isnan(hand[p2])):
                continue
            x1,y1 = to_pixel(hand[p1])
            x2,y2 = to_pixel(hand[p2])
            cv2.line(img,(x1,y1),(x2,y2),color,2)
        for p in hand:
            if not np.any(np.isnan(p)):
                x,y = to_pixel(p)
                cv2.circle(img,(x,y),2,color,-1)
    draw_hand(lh,(255,0,0))
    draw_hand(rh,(0,0,255))

    # ---- FACE ----
    for p in face:
        if not np.any(np.isnan(p)):
            x,y = to_pixel(p)
            cv2.circle(img,(x,y),1,(255,255,255),-1)

    return img


def display_video(sequence, delay=30, loop=False, title="Skeleton Video"):
    """
    Display a single video from sequence
    
    Args:
        sequence: numpy array of shape (num_frames, 438)
        delay: time between frames in ms
        loop: whether to loop the video
        title: window title
    """
    num_frames = len(sequence)
    
    while True:
        for i, frame in enumerate(sequence):
            img = render_skeleton(frame, dynamic=True)
            
            # Add frame counter
            cv2.putText(img, f"Frame: {i}/{num_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(img, "Controls: q=quit, p=pause, s=save", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow(title, img)
            
            key = cv2.waitKey(delay) & 0xFF
            
            if key == ord('q'):  # Quit
                cv2.destroyAllWindows()
                return
            elif key == ord('p'):  # Pause
                print("Paused. Press any key to resume...")
                cv2.waitKey(0)
            elif key == ord('s'):  # Save current frame
                cv2.imwrite(f"frame_{i}.png", img)
                print(f"Saved frame {i}")
        
        if not loop:
            break
    
    cv2.destroyAllWindows()


def display_single_video(npy_path, delay=30):
    """
    Display a single .npy file
    
    Args:
        npy_path: path to the .npy file
        delay: time between frames in ms
    """
    print(f"Loading: {npy_path}")
    sequence = np.load(npy_path)
    print(f"Shape: {sequence.shape}")
    print(f"Number of frames: {len(sequence)}")
    print("Controls: 'q' to quit, 'p' to pause, 's' to save frame")
    
    display_video(sequence, delay=delay)


def display_video_from_array(sequence, label="", delay=30):
    """
    Display a video from a numpy array (e.g., from processed_data)
    
    Args:
        sequence: numpy array of shape (num_frames, 438)
        label: label for the video
        delay: time between frames in ms
    """
    print(f"Displaying video: {label}")
    print(f"Number of frames: {len(sequence)}")
    print("Controls: 'q' to quit, 'p' to pause, 's' to save frame")
    
    display_video(sequence, delay=delay, title=f"Skeleton: {label}")


# ============================================================
# NEW FUNCTION: RENDER BOTH ORIGINAL VIDEO AND SKELETON SIDE BY SIDE
# ============================================================

def render_original_and_skeleton(
    video_path: str,
    skeleton_sequence: np.ndarray,
    start_frame: int = 0,
    end_frame: int = None,
    delay: int = 30,
    skeleton_only: bool = False,
    original_only: bool = False,
    save_video: str = None,
    fps: int = 30
):
    """
    Render original video and skeleton side by side for comparison
    
    Args:
        video_path: Path to the original video file
        skeleton_sequence: numpy array of shape (num_frames, 438) - skeleton landmarks
        start_frame: Starting frame index (default: 0)
        end_frame: Ending frame index (default: None = all frames)
        delay: Time between frames in ms (default: 30)
        skeleton_only: Show only skeleton (default: False)
        original_only: Show only original video (default: False)
        save_video: Path to save the output video (optional)
        fps: Frames per second for saved video (default: 30)
    """
    
    # Open original video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return
    
    # Get video properties
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Determine frame range
    if end_frame is None:
        end_frame = min(total_video_frames, len(skeleton_sequence))
    end_frame = min(end_frame, total_video_frames, len(skeleton_sequence))
    
    # Calculate canvas size for side-by-side display
    if skeleton_only:
        canvas_width = 640
        canvas_height = 480
    elif original_only:
        canvas_width = video_width
        canvas_height = video_height
    else:
        canvas_width = video_width + 640
        canvas_height = max(video_height, 480)
    
    # Video writer for saving
    video_writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(save_video, fourcc, fps, (canvas_width, canvas_height))
    
    print(f"\n🎬 Rendering Comparison Video")
    print(f"   Original video: {video_path}")
    print(f"   Original frames: {total_video_frames}")
    print(f"   Skeleton frames: {len(skeleton_sequence)}")
    print(f"   Displaying frames: {start_frame} to {end_frame-1}")
    print(f"   Controls: 'q' to quit, 'p' to pause, 's' to save frame, 'a' to toggle original/skeleton")
    print("="*60)
    
    # Jump to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_idx = start_frame
    show_original = True
    show_skeleton = True
    
    while frame_idx < end_frame:
        ret, original_frame = cap.read()
        if not ret:
            break
        
        # Get skeleton frame (align indices)
        skeleton_idx = frame_idx
        if skeleton_idx >= len(skeleton_sequence):
            skeleton_idx = len(skeleton_sequence) - 1
        
        skeleton_frame = render_skeleton(skeleton_sequence[skeleton_idx], dynamic=True)
        
        # Create combined display based on mode
        if skeleton_only:
            display_frame = skeleton_frame
            mode_text = "SKELETON ONLY"
        elif original_only:
            display_frame = original_frame
            mode_text = "ORIGINAL ONLY"
        else:
            # Resize frames to have same height for side-by-side
            target_height = canvas_height
            
            # Resize original frame
            orig_resized = cv2.resize(original_frame, (video_width, target_height))
            
            # Resize skeleton frame
            skel_resized = cv2.resize(skeleton_frame, (640, target_height))
            
            # Concatenate horizontally
            display_frame = np.hstack([orig_resized, skel_resized])
            mode_text = "ORIGINAL (LEFT) | SKELETON (RIGHT)"
        
        # Add overlay information
        # Frame counter
        cv2.putText(display_frame, f"Frame: {frame_idx}/{end_frame-1}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Mode indicator
        cv2.putText(display_frame, mode_text, 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls help
        controls = "q:quit | p:pause | s:save | a:toggle view"
        cv2.putText(display_frame, controls, 
                   (10, display_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Display or save
        if video_writer:
            video_writer.write(display_frame)
        
        cv2.imshow("Original vs Skeleton Comparison", display_frame)
        
        key = cv2.waitKey(delay) & 0xFF
        
        if key == ord('q'):  # Quit
            break
        elif key == ord('p'):  # Pause
            print("Paused. Press any key to resume...")
            cv2.waitKey(0)
        elif key == ord('s'):  # Save current frame
            cv2.imwrite(f"comparison_frame_{frame_idx}.png", display_frame)
            print(f"Saved comparison frame {frame_idx}")
        elif key == ord('a'):  # Toggle view mode
            if skeleton_only:
                skeleton_only = False
                original_only = False
                print("Switched to: SIDE-BY-SIDE mode")
            elif original_only:
                original_only = False
                skeleton_only = True
                print("Switched to: SKELETON ONLY mode")
            else:
                original_only = True
                skeleton_only = False
                print("Switched to: ORIGINAL ONLY mode")
            
            # Recalculate canvas size
            if skeleton_only:
                canvas_width = 640
                canvas_height = 480
            elif original_only:
                canvas_width = video_width
                canvas_height = video_height
            else:
                canvas_width = video_width + 640
                canvas_height = max(video_height, 480)
            
            if video_writer:
                video_writer.release()
                video_writer = cv2.VideoWriter(save_video, fourcc, fps, (canvas_width, canvas_height))
        
        frame_idx += 1
    
    # Cleanup
    cap.release()
    if video_writer:
        video_writer.release()
        print(f"\n💾 Video saved to: {save_video}")
    
    cv2.destroyAllWindows()
    print("\n✅ Rendering complete!")


def display_video_with_skeleton(npy_path, video_path, delay=30):
    """
    Convenience function to display skeleton and original video together
    
    Args:
        npy_path: Path to the .npy skeleton file
        video_path: Path to the original video file
        delay: Time between frames in ms
    """
    print(f"Loading skeleton: {npy_path}")
    skeleton_sequence = np.load(npy_path)
    
    render_original_and_skeleton(
        video_path=video_path,
        skeleton_sequence=skeleton_sequence,
        delay=delay
    )


def display_array_with_video(skeleton_sequence, video_path, label="", delay=30):
    """
    Display skeleton array (from processed_data) with original video
    
    Args:
        skeleton_sequence: numpy array of skeleton landmarks
        video_path: Path to the original video file
        label: Label for the video
        delay: Time between frames in ms
    """
    print(f"Displaying video: {label}")
    print(f"Skeleton frames: {len(skeleton_sequence)}")
    
    render_original_and_skeleton(
        video_path=video_path,
        skeleton_sequence=skeleton_sequence,
        delay=delay
    )



def display_two_sequences(sequence1, sequence2, 
                         label1="Sequence 1", 
                         label2="Sequence 2",
                         dynamic=True,
                         delay=30,
                         title="Sequence Comparison",
                         save_video=None,
                         fps=30):
    """
    Display two skeleton sequences side by side for comparison
    
    Args:
        sequence1: First numpy array of shape (num_frames, 438)
        sequence2: Second numpy array of shape (num_frames, 438)
        label1: Label for first sequence (default: "Sequence 1")
        label2: Label for second sequence (default: "Sequence 2")
        dynamic: Whether to use dynamic scaling (default: True)
        delay: Time between frames in ms (default: 30)
        title: Window title (default: "Sequence Comparison")
        save_video: Path to save the output video (optional)
        fps: Frames per second for saved video (default: 30)
    """
    
    # Get number of frames for both sequences
    frames1 = len(sequence1)
    frames2 = len(sequence2)
    max_frames = max(frames1, frames2)
    
    # Canvas dimensions
    canvas_width = 640 * 2  # 640 for each sequence
    canvas_height = 480
    
    # Video writer for saving
    video_writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(save_video, fourcc, fps, (canvas_width, canvas_height))
    
    print(f"\n🎬 Rendering Side-by-Side Comparison")
    print(f"   {label1}: {frames1} frames")
    print(f"   {label2}: {frames2} frames")
    print(f"   Displaying {max_frames} frames total")
    print(f"   Controls: 'q' to quit, 'p' to pause, 's' to save frame")
    print("="*60)
    
    # Determine if we need to pad sequences
    def get_frame(sequence, idx):
        if idx < len(sequence):
            return render_skeleton(sequence[idx], canvas_size=(640, 480), dynamic=dynamic)
        else:
            # Return blank frame with message
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "End of Sequence", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            return blank
    
    for frame_idx in range(max_frames):
        # Get frames from both sequences
        frame1 = get_frame(sequence1, frame_idx)
        frame2 = get_frame(sequence2, frame_idx)
        
        # Add labels
        # Label for first sequence
        cv2.putText(frame1, f"{label1}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame1, f"Frame: {min(frame_idx, frames1-1)}/{frames1-1}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Label for second sequence
        cv2.putText(frame2, f"{label2}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame2, f"Frame: {min(frame_idx, frames2-1)}/{frames2-1}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Concatenate horizontally
        combined = np.hstack([frame1, frame2])
        
        # Add overall frame counter
        cv2.putText(combined, f"Comparison Frame: {frame_idx}/{max_frames-1}", 
                   (10, canvas_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Add controls help
        cv2.putText(combined, "Controls: q=quit | p=pause | s=save", 
                   (canvas_width - 250, canvas_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Save or display
        if video_writer:
            video_writer.write(combined)
        
        cv2.imshow(title, combined)
        
        key = cv2.waitKey(delay) & 0xFF
        
        if key == ord('q'):  # Quit
            break
        elif key == ord('p'):  # Pause
            print("Paused. Press any key to resume...")
            cv2.waitKey(0)
        elif key == ord('s'):  # Save current frame
            cv2.imwrite(f"comparison_{label1}_vs_{label2}_frame_{frame_idx}.png", combined)
            print(f"Saved comparison frame {frame_idx}")
    
    # Cleanup
    if video_writer:
        video_writer.release()
        print(f"\n💾 Video saved to: {save_video}")
    
    cv2.destroyAllWindows()
    print("\n✅ Comparison complete!")


def display_original_vs_augmented(original_sequence, augmented_sequence, 
                                   label="Sample", delay=30, save_video=None):
    """
    Display original vs augmented sequence (handles different frame lengths)
    """
    # Get lengths
    len_orig = len(original_sequence)
    len_aug = len(augmented_sequence)
    
    print(f"Original frames: {len_orig}, Augmented frames: {len_aug}")
    
    if len_orig == len_aug:
        # Same length - normal display
        display_two_sequences(
            original_sequence, 
            augmented_sequence,
            label1=f"Original {label}",
            label2=f"Augmented {label}",
            delay=delay,
            title="Original vs Augmented Comparison",
            save_video=save_video
        )
    else:
        # Different lengths - pad or show message
        print(f"⚠️ Different frame counts ({len_orig} vs {len_aug})")
        print(f"   Speed augmentation changes temporal dimension")
        print(f"   Showing side-by-side with padding...")
        
        # Pad shorter sequence to match longer one
        max_len = max(len_orig, len_aug)
        
        def pad_sequence(seq, target_len):
            if len(seq) == target_len:
                return seq
            # Pad with last frame
            last_frame = seq[-1]
            pad = np.tile(last_frame, (target_len - len(seq), 1))
            return np.vstack([seq, pad])
        
        orig_padded = pad_sequence(original_sequence, max_len)
        aug_padded = pad_sequence(augmented_sequence, max_len)
        
        display_two_sequences(
            orig_padded, 
            aug_padded,
            label1=f"Original {label} ({len_orig} frames)",
            label2=f"Augmented {label} ({len_aug} frames)",
            delay=delay,
            title="Original vs Augmented Comparison (Padded)",
            save_video=save_video
        )

def display_before_after(before_sequence, after_sequence, 
                         before_label="Before Preprocessing",
                         after_label="After Preprocessing",
                         delay=30, save_video=None):
    """
    Display before and after preprocessing comparison
    
    Args:
        before_sequence: Sequence before preprocessing
        after_sequence: Sequence after preprocessing
        before_label: Label for before sequence
        after_label: Label for after sequence
        delay: Time between frames in ms
        save_video: Path to save the output video
    """
    display_two_sequences(
        before_sequence, 
        after_sequence,
        label1=before_label,
        label2=after_label,
        delay=delay,
        title="Before vs After Preprocessing",
        save_video=save_video
    )


def display_multiple_sequences(sequences, labels, 
                               cols=2, dynamic=True, delay=30, 
                               title="Multiple Sequence Comparison",
                               save_video=None, fps=30):
    """
    Display multiple sequences in a grid layout
    
    Args:
        sequences: List of numpy arrays (each shape: frames, 438)
        labels: List of labels for each sequence
        cols: Number of columns in the grid (default: 2)
        dynamic: Whether to use dynamic scaling
        delay: Time between frames in ms
        title: Window title
        save_video: Path to save the output video
        fps: Frames per second for saved video
    """
    num_sequences = len(sequences)
    rows = (num_sequences + cols - 1) // cols
    
    # Get max frames
    max_frames = max(len(seq) for seq in sequences)
    
    # Canvas dimensions per cell
    cell_width = 640
    cell_height = 480
    canvas_width = cell_width * cols
    canvas_height = cell_height * rows
    
    # Video writer
    video_writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(save_video, fourcc, fps, (canvas_width, canvas_height))
    
    print(f"\n🎬 Rendering Grid Comparison ({rows}x{cols})")
    print(f"   Sequences: {num_sequences}")
    print(f"   Max frames: {max_frames}")
    print(f"   Controls: 'q' to quit, 'p' to pause, 's' to save frame")
    print("="*60)
    
    def get_frame(sequence, idx, label):
        if idx < len(sequence):
            frame = render_skeleton(sequence[idx], canvas_size=(cell_width, cell_height), dynamic=dynamic)
        else:
            frame = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
            cv2.putText(frame, "End of Sequence", (cell_width//2-100, cell_height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
        
        # Add label
        cv2.putText(frame, label, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Frame: {idx}/{len(sequence)-1}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    for frame_idx in range(max_frames):
        # Create grid
        grid = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        for i, (seq, label) in enumerate(zip(sequences, labels)):
            row = i // cols
            col = i % cols
            
            y_start = row * cell_height
            x_start = col * cell_width
            
            frame = get_frame(seq, frame_idx, label)
            grid[y_start:y_start+cell_height, x_start:x_start+cell_width] = frame
        
        # Add overall frame counter
        cv2.putText(grid, f"Frame: {frame_idx}/{max_frames-1}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add controls help
        cv2.putText(grid, "q:quit | p:pause | s:save", 
                   (canvas_width - 200, canvas_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        if video_writer:
            video_writer.write(grid)
        
        cv2.imshow(title, grid)
        
        key = cv2.waitKey(delay) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('p'):
            print("Paused. Press any key to resume...")
            cv2.waitKey(0)
        elif key == ord('s'):
            cv2.imwrite(f"grid_comparison_frame_{frame_idx}.png", grid)
            print(f"Saved grid frame {frame_idx}")
    
    if video_writer:
        video_writer.release()
        print(f"\n💾 Video saved to: {save_video}")
    
    cv2.destroyAllWindows()
    print("\n✅ Grid comparison complete!")
