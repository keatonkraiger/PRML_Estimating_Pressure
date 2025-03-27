import cv2
import subprocess
from pathlib import Path

def prepare_images_for_stacking(images, stack_vertical=True):
    """
    Prepare images for stacking by resizing them appropriately.
    
    Args:
        images: List of numpy arrays (images to stack)
        stack_vertical: If True, stack vertically (match widths), else stack horizontally (match heights)
    
    Returns:
        List of resized images ready for stacking
    """
    if not images:
        return images
        
    if stack_vertical:
        # For vertical stacking, all images need same width
        target_width = max(img.shape[1] for img in images)
        resized_images = []
        
        for img in images:
            if img.shape[1] != target_width:
                # Scale height proportionally to maintain aspect ratio
                scale_factor = target_width / img.shape[1]
                new_height = int(img.shape[0] * scale_factor)
                resized = cv2.resize(img, (target_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                resized_images.append(resized)
            else:
                resized_images.append(img)
    else:
        # For horizontal stacking, all images need same height
        target_height = max(img.shape[0] for img in images)
        resized_images = []
        
        for img in images:
            if img.shape[0] != target_height:
                # Scale width proportionally to maintain aspect ratio
                scale_factor = target_height / img.shape[0]
                new_width = int(img.shape[1] * scale_factor)
                resized = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_LANCZOS4)
                resized_images.append(resized)
            else:
                resized_images.append(img)
    
    return resized_images

class VideoCreator:
    """Handles video creation and combination"""
    def __init__(self, default_fps=30):
        self.default_fps = default_fps
        
    def _ensure_even_dimensions(self, first_frame_path):
        """Check dimensions and return scaling filter if needed"""
        img = cv2.imread(str(first_frame_path))
        height, width = img.shape[:2]
        
        new_width = width - (width % 2)
        new_height = height - (height % 2)
        
        if new_width != width or new_height != height:
            print(f"Adjusting dimensions from {width}x{height} to {new_width}x{new_height}")
            return f'scale={new_width}:{new_height}'
        return None

    def create_video_from_frames(self, frame_dir, output_path):
        """Create video from frame directory using ffmpeg"""
        frame_dir = Path(frame_dir)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get and sort all frame paths
        frame_files = sorted(frame_dir.glob('frame_*.png'))
        if not frame_files:
            raise FileNotFoundError(f"No frames found in {frame_dir}")
            
        # Check dimensions from first frame
        first_frame = cv2.imread(str(frame_files[0]))
        height, width = first_frame.shape[:2]
        print(f"Original dimensions: {width}x{height}")
        
        # Calculate dimensions divisible by 2
        new_width = width - (width % 2)
        new_height = height - (height % 2)
        print(f"Adjusted dimensions: {new_width}x{new_height}")

        # Create a concat file listing all frames in order
        concat_file = output_path.parent / 'concat.txt'
        try:
            with open(concat_file, 'w') as f:
                for frame in frame_files:
                    f.write(f"file '{frame.absolute()}'\n")
                    f.write(f"duration {1/self.default_fps}\n")

            # Use concat demuxer with scaling filter
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',  # Allow absolute paths
                '-i', str(concat_file),
                '-vf', f'scale={new_width}:{new_height}',  # Force even dimensions
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                str(output_path)
            ]
            
            print("Running ffmpeg command:", ' '.join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print("FFmpeg error output:", result.stderr)
                result.check_returncode()

            # Verify the output video
            if output_path.exists():
                cap = cv2.VideoCapture(str(output_path))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                print(f"Created video with {frame_count} frames")
                
        finally:
            # Clean up concat file
            if concat_file.exists():
                concat_file.unlink()

    def combine_videos(self, video_paths, output_path):
        """Combine multiple videos vertically"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Verify and check all videos
        max_width = 0
        for video_path in video_paths:
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video not found: {video_path}")
            
            cap = cv2.VideoCapture(str(video_path))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            max_width = max(max_width, width)
            cap.release()

        # Ensure max_width is even
        max_width = max_width - (max_width % 2)

        # Build filter complex with scaling
        filter_parts = []
        input_labels = []
        for i, video_path in enumerate(video_paths):
            cap = cv2.VideoCapture(str(video_path))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            
            if width != max_width:
                filter_parts.append(f'[{i}:v]scale={max_width}:-2[v{i}];')  # -2 maintains aspect ratio with even height
                input_labels.append(f'[v{i}]')
            else:
                input_labels.append(f'[{i}:v]')

        filter_complex = ''.join(filter_parts) + \
                        ''.join(input_labels) + \
                        f'vstack=inputs={len(video_paths)}[out]'
        
        cmd = ['ffmpeg', '-y']
        for video_path in video_paths:
            cmd.extend(['-i', str(video_path)])
        cmd.extend([
            '-filter_complex', filter_complex,
            '-map', '[out]',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            str(output_path)
        ])
        
        print("Running ffmpeg command:", ' '.join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("FFmpeg error output:", result.stderr)
            result.check_returncode()
            
def process_modality_videos(viz_dir, modalities, fps=30, stack_vertical=True, modality_order=None):
    """
    Process and combine videos for each modality with configurable ordering and stacking.
    
    Args:
        viz_dir: Base visualization directory
        modalities: List of modality names (e.g., ['pressure', 'contact', 'com'])
        fps: Frames per second for the videos
        stack_vertical: If True, stack videos vertically; if False, horizontally
        modality_order: Optional list specifying order of modalities in combined video
    """
    viz_dir = Path(viz_dir)
    creator = VideoCreator()
    video_paths = {}
    
    # Create individual videos
    for modality in modalities:
        frame_dir = viz_dir / modality / 'combined'
        if not frame_dir.exists():
            print(f"Skipping {modality} - no frames directory found")
            continue
            
        output_path = viz_dir / f'{modality}_visualization.mp4'
        try:
            creator.create_video_from_frames(frame_dir, output_path, fps=fps)
            video_paths[modality] = output_path
            print(f"Successfully created video for {modality}")
        except Exception as e:
            print(f"Error processing {modality}: {str(e)}")
            continue
    
    # Combine videos if we have multiple
    if len(video_paths) > 1:
        try:
            # Determine video order
            if modality_order is None:
                # Default to order in modalities list
                ordered_paths = [video_paths[m] for m in modalities if m in video_paths]
            else:
                # Use specified order
                ordered_paths = [video_paths[m] for m in modality_order if m in video_paths]
            
            print(f"\nCombining videos in order: {[p.stem for p in ordered_paths]}")
            combined_path = viz_dir / 'combined_visualization.mp4'
            creator.combine_videos(ordered_paths, combined_path, stack_vertical=stack_vertical)
            print("Successfully combined videos")
            
        except Exception as e:
            print(f"Error combining videos: {str(e)}")