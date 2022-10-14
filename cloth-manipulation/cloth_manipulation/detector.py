import wandb
from pathlib import Path
from keypoint_detection.models.detector import KeypointDetector
from keypoint_detection.models.backbones.maxvit_unet import MaxVitUnet

default_reference = "airo-box-manipulation/iros2022/model-14zb70au:v1"
default_backbone_class  = MaxVitUnet

# TODO figure out whether we can remove the requirement to pass backbone
def get_wandb_model(checkpoint_reference=default_reference, backbone=default_backbone_class()):
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
