import BEN2
from PIL import Image
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

video_path = "/root/gpufree-data/samplevideo/111.mp4"# input video

model = BEN2.BEN_Base().to(device).eval() #init pipeline

model.loadcheckpoints("/root/gpufree-data/models/BEN2_Base.pth")



model.segment_video(
    video_path= video_path,
    output_path="./", # Outputs will be saved as foreground.webm or foreground.mp4. The default value is "./"
    fps=0, # If this is set to 0 CV2 will detect the fps in the original video. The default value is 0.
    refine_foreground=False,  #refine foreground is an extract postprocessing step that increases inference time but can improve matting edges. The default value is False.
    batch=8,  # We recommended that batch size not exceed 3 for consumer GPUs as there are minimal inference gains. The default value is 1.
    print_frames_processed=False,  #Informs you what frame is being processed. The default value is True.
    webm = False, # This will output an alpha layer video but this defaults to mp4 when webm is false. The default value is False.
    rgb_value= (0, 255, 0) # If you do not use webm this will be the RGB value of the resulting background only when webm is False. The default value is a green background (0,255,0).
 )