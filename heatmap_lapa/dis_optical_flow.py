import cv2 as cv
import numpy as np
import os
import glob

PATCH_GRID = (8, 8)  # Define the grid size for the DIS optical flow

def disOpticalFlowNuScenes():
    # read the image
    root = os.getcwd()
    
    image_dir = os.path.join(root,'/home/andy/Dataset/Nuscenes/v1.0-mini/sweeps/CAM_FRONT')
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')) + glob.glob(os.path.join(image_dir, '*.png')))
    
    if not image_files:
        raise FileNotFoundError(f'No images found in directory: {image_dir}')

    # read the first frame
    frame1 = cv.imread(image_files[0])
    if frame1 is None:
        raise RuntimeError('Failed to read the first frame from the image directory.')

    # convert the first frame to grayscale
    img1Gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    
    # create an HSV image for flow visualization
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255  # Set saturation to maximum
    
    # Create the DIS Optical flow instance
    #Presets: PRESET_ULTRAFAST, PRESET_FAST, PRESET_MEDIUM, PRESET_SLOW
    dis = cv.DISOpticalFlow_create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM)
    
    output_dir = os.path.join(root, 'optical_flow/output_dis')
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, 'dis_optical_flow.mp4')
    fps = 20.0  # Set a default FPS value
    writer = cv.VideoWriter(
        video_path,
        cv.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame1.shape[1], frame1.shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError(f'Unable to create video writer: {video_path}')

    frame_idx = 0

    while True:
        # read the next frame
        if frame_idx + 1 >= len(image_files):
            break
        frame2 = cv.imread(image_files[frame_idx + 1])
        if frame2 is None:
            break
        
        # convert the next frame to grayscale
        img2Gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        
        # calculate dense optical flow using DIS method
        flow = dis.calc(img1Gray, img2Gray, None)
        
        # convert the flow to polar coordinates (magnitude and angle)
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        
        # set the hue according to the optical flow direction
        hsv[..., 0] = angle * 180 / np.pi / 2
        
        # set the value according to the optical flow magnitude (normalized)
        hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        
        # convert HSV to BGR for visualization
        bgrFlow = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        
        # save the result to video and optionally display when a GUI is available
        writer.write(bgrFlow)
        output_path = os.path.join(output_dir, f'dis_flow_{frame_idx:04d}.png')
        cv.imwrite(output_path, bgrFlow)
        frame_idx += 1

        if os.environ.get('DISPLAY') is not None:
            try:
                cv.imshow('DIS Optical Flow', bgrFlow)
                key = cv.waitKey(15) & 0xFF
                if key == ord('q'):
                    break
            except cv.error:
                pass

        # update the previous frame and previous points
        img1Gray = img2Gray.copy()

    writer.release()
    if os.environ.get('DISPLAY') is not None:
        cv.destroyAllWindows()
    
if __name__ == "__main__":
    disOpticalFlowNuScenes()
    