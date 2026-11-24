import torch
import torch.nn as nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors

IMAGE_KEY = "observation.images.crown"
STATE_KEY = "observation.state"
ACTION_KEY = "action"

class IL_Model(nn.Module):
    """ACT policy (ImageNet-pretrained ResNet18 + transformer) with its input/output normalisation."""
    def __init__(self, action_dim, obs_dim, action_chunk, img_dim):

        super().__init__()
        img_h, img_w = img_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = ACTConfig(
            input_features={
                IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, img_h, img_w)),
                STATE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(obs_dim,)),
            },
            output_features={ACTION_KEY: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
            chunk_size=action_chunk,
            n_action_steps=action_chunk,
            vision_backbone="resnet18",
            pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
            device=self.device,
        )
        self.policy = ACTPolicy(self.config).to(self.device)
        self.preprocessor, self.postprocessor = None, None

    def set_stats(self, stats):
        self.preprocessor, self.postprocessor = make_act_pre_post_processors(self.config, dataset_stats=stats)

    def forward(self, batch):
        return self.policy.forward(self.preprocessor(batch))

    @torch.no_grad()
    def act(self, obs):
        return self.postprocessor(self.policy.select_action(self.preprocessor(obs)))

    def reset(self):
        self.policy.reset()

    def save(self, path):
        self.policy.save_pretrained(path)
        self.preprocessor.save_pretrained(path)
        self.postprocessor.save_pretrained(path)

    @classmethod
    def load(cls, path):
        policy = ACTPolicy.from_pretrained(path)
        cfg = policy.config
        model = cls(cfg.output_features[ACTION_KEY].shape[0], cfg.input_features[STATE_KEY].shape[0],
                    cfg.chunk_size, cfg.input_features[IMAGE_KEY].shape[1:])
        model.policy = policy.to(model.device)
        model.preprocessor, model.postprocessor = make_pre_post_processors(cfg, pretrained_path=str(path))
        model.eval()
        return model