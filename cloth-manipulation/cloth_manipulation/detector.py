import wandb
from pathlib import Path
from keypoint_detection.models.detector import KeypointDetector
from keypoint_detection.models.backbones.convnext_unet import ConvNeXtUnet
from keypoint_detection.utils.visualization import overlay_image_with_heatmap
from keypoint_detection.utils.heatmap import get_keypoints_from_heatmap


def get_wandb_model(checkpoint_reference, backbone=ConvNeXtUnet()):
    """
    checkpoint_reference: str e.g. 'airo-box-manipulation/iros2022_0/model-17tyvqfk:v3'
    """
    # download checkpoint locally (if not already cached)
    run = wandb.init(project="inference", entity="airo-box-manipulation")
    artifact = run.use_artifact(checkpoint_reference, type="model")
    artifact_dir = artifact.download()
    model_path = Path(artifact_dir) / "model.ckpt"
    model = KeypointDetector.load_from_checkpoint(model_path, backbone=backbone)
    return model
