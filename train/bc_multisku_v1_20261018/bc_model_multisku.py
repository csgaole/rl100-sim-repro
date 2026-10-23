"""[MULTI-SKU copy of bc_model_novib.py: adds low_dim obs key sku_cond=[L,W,H,mass]/[.30,.25,.15,.60] (4D),
concatenated by MultiImageObsEncoder into the global cond of BOTH obs steps (UNet global_cond_dim += 8). Nothing else changed.]
[COPY of bc_model.py with use_vib=False (P3 v3)]
Official RL-100 modules and unchanged compute_loss, with a task-only host.

The upstream policy imports 3D/RL dependencies even for RGB BC. Load only its
compute_loss and RGB augmentation method via AST; do not stub those dependencies.
The extracted methods are compiled without modifying their AST or source.
"""
import ast,sys
from pathlib import Path
import torch,torch.nn as nn,torch.nn.functional as F
import torchvision
from einops import reduce,rearrange
ROOT=Path('/data02/kemove/sku09-bc-migration/sku09-flip-demonstrations')
sys.path.insert(0,str(ROOT/'vendor/rl100'))
from rl_100.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from rl_100.model.diffusion.conditional_unet1d import ConditionalUnet1D
from rl_100.model.diffusion.mask_generator import LowdimMaskGenerator
from rl_100.model.common.normalizer import LinearNormalizer
from rl_100.model.common.aug import RandomShiftsAug
from rl_100.common.pytorch_util import dict_apply
from rl_100.unidpg.diffusion_policy.diffusers_patch.ddim_with_logprob_dpok import DDIMSchedulerExtended

SOURCE=ROOT/'vendor/rl100/rl_100/policy/rl100_2d.py'
tree=ast.parse(SOURCE.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RL1002D')
methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['compute_loss','_augment_rgb_obs']]
scope=dict(torch=torch,F=F,reduce=reduce,rearrange=rearrange,dict_apply=dict_apply)
exec(compile(ast.Module(body=methods,type_ignores=[]),str(SOURCE),'exec'),scope)

class BCPolicy(nn.Module):
    compute_loss=scope['compute_loss']
    _augment_rgb_obs=scope['_augment_rgb_obs']
    def __init__(self,normalizer_data,pretrained=True):
        super().__init__()
        self.rgb_obs_keys=('global','left_wrist','right_wrist')
        shape=dict(action={'shape':[14]},obs={k:dict(shape=[3,240,320],type='rgb') for k in self.rgb_obs_keys})
        shape['obs']['agent_pos']=dict(shape=[28],type='low_dim')
        shape['obs']['sku_cond']=dict(shape=[4],type='low_dim')
        resnet=torchvision.models.resnet18(weights=None);resnet.fc=nn.Identity()
        if pretrained:
            raw=torch.load(ROOT/'assets/r3m_resnet18.pt',map_location='cpu',weights_only=False)['r3m']
            prefix='module.convnet.';weights={k[len(prefix):]:v for k,v in raw.items() if k.startswith(prefix)}
            resnet.load_state_dict(weights,strict=True)
        self.obs_encoder=MultiImageObsEncoder(shape_meta=shape,rgb_model=resnet,resize_shape=(224,224),crop_shape=None,
            use_group_norm=True,share_rgb_model=False,imagenet_norm=True,use_agent_pos=True,
            use_recon=True,use_vib=False,recon_loss_weight=.05,kl_loss_weight=1.,kl_beta=5e-4)
        self.model=ConditionalUnet1D(input_dim=14,global_cond_dim=self.obs_encoder.output_shape()*2,
            diffusion_step_embed_dim=256,down_dims=[256,512,1024],kernel_size=5,n_groups=8,
            condition_type='film',use_down_condition=True,use_mid_condition=True,use_up_condition=True)
        self.normalizer=LinearNormalizer();self.normalizer.fit(normalizer_data,mode='limits')
        self.noise_scheduler=DDIMSchedulerExtended(num_train_timesteps=100,beta_start=.0001,beta_end=.02,
            beta_schedule='squaredcos_cap_v2',clip_sample=True,set_alpha_to_one=True,steps_offset=1,prediction_type='epsilon',clip_std_min=.0067,clip_std_max=None)
        self.mask_generator=LowdimMaskGenerator(action_dim=14,obs_dim=0,max_n_obs_steps=2,fix_obs_steps=True,action_visible=False)
        self.n_obs_steps=2;self.n_action_steps=8;self.horizon=9;self.no_pre_action=True
        self.action_norm=True;self.obs_as_global_cond=True;self.w_pc=False;self.use_pc_color=False
        self.use_aug=True;self.aug=RandomShiftsAug(pad=4);self.use_recon=True;self.encoder_type='resnet'
        self.condition_type='film';self.is_flow=False
    @property
    def device(self):return next(self.parameters()).device

