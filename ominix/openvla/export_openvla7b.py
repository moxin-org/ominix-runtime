# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
import gguf
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import re

dtype = torch.bfloat16

repo_id = "openvla/openvla-7b"

# Load Processor & VLA
processor = AutoProcessor.from_pretrained(
    repo_id, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    repo_id,
    # [Optional] Requires `flash_attn`
    attn_implementation="flash_attention_2",
    dtype=dtype,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to("cuda:0")

# save large language model
llm = vla.language_model.to(dtype)
llm.save_pretrained("language_model")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    repo_id, trust_remote_code=True)
tokenizer.save_pretrained("language_model")


featurizer = vla.vision_backbone.featurizer.to(dtype)
fused_featurizer = vla.vision_backbone.fused_featurizer.to(dtype)
projector = vla.projector.to(dtype)





# export vision models and projector to gguf
export = True
model_names = ["siglip", "dinov2" , "proj"]
models = [fused_featurizer, featurizer, projector]
def add_config(gguf_writer: gguf.GGUFWriter, model_cfg):
    for k, v in model_cfg.items():
        if isinstance(v, bool):
            gguf_writer.add_bool(k, v)
        elif isinstance(v, float):
            gguf_writer.add_float32(k, v)
        elif isinstance(v, int):
            gguf_writer.add_uint32(k, v)
        elif isinstance(v, str):
            gguf_writer.add_string(k, v)
        elif isinstance(v, list):
            gguf_writer.add_array(k, v)
        else:
            raise ValueError(f"Unsupported type: {type(v)}")


if export:
    cfgs = {
        "proj": {},
        "dinov2": {
            "clip.projector_type": "openvla",
            "clip.has_vision_encoder": True,
            "clip.vision.embedding_length": 1024,
            "clip.vision.attention.head_count": 16,
            "clip.vision.feed_forward_length": 4096,
            "clip.vision.block_count": 24,
            "clip.vision.projection_dim": 2560,
            "clip.vision.attention.layer_norm_epsilon": 1e-6,
            "clip.vision.image_size": 224,
            "clip.vision.patch_size": 14,
            "clip.use_gelu": True,
            "clip.vision.image_mean": [0.484375, 0.455078125, 0.40625],
            "clip.vision.image_std": [0.228515625, 0.2236328125, 0.224609375],
            "clip.vision.feature_layer": [24-2],
        },
        "siglip": {
            "clip.projector_type": "openvla",
            "clip.has_vision_encoder": True,
            "clip.vision.embedding_length": 1152,
            "clip.vision.attention.head_count": 16,
            "clip.vision.feed_forward_length": 4304,
            "clip.vision.block_count": 27,
            "clip.vision.projection_dim": 2560,
            "clip.vision.attention.layer_norm_epsilon": 1e-6,
            "clip.vision.image_size": 224,
            "clip.vision.patch_size": 14,
            "clip.use_gelu": True,
            "clip.vision.image_mean": [0.5, 0.5, 0.5],
            "clip.vision.image_std": [0.5, 0.5, 0.5],
            "clip.vision.feature_layer": [27-2],
        }
    }
    # siglip
    model_params = {
        "cls_token": "v.class_embd",
        "reg_token": "v.reg_embd",
        # "norm_pre.weight": "v.pre_ln.weight",
        # "norm_pre.bias": "v.pre_ln.bias",
        # "": "v.post_ln.weight",
        # "": "v.post_ln.bias",
        "patch_embed.proj.weight": "v.patch_embd.weight",
        "patch_embed.proj.bias": "v.patch_embd.bias",
        # "": "vision.patch_embd.weight.1",
        "pos_embed": "v.position_embd.weight",  # 1152x4096
        # "": "mm.input_projection.weight",
        # "": "mm.soft_emb_norm.weight",
        "fc1.weight": "fc1.weight",
        "fc1.bias": "fc1.bias",
        "fc2.weight": "fc2.weight",
        "fc2.bias": "fc2.bias",
        "fc3.weight": "fc3.weight",
        "fc3.bias": "fc3.bias",
    }
    blk_params = {
        "blocks.%d.attn.qkv.weight": "v.blk.%d.attn_q.weight",
        # "blocks.%d.attn.qkv.weight": "v.blk.%d.attn_k.weight",
        # "blocks.%d.attn.qkv.weight": "v.blk.%d.attn_vision.weight",
        "blocks.%d.attn.proj.weight": "v.blk.%d.attn_out.weight",
        # "": "v.blk.%d.attn_k_norm.weight",
        # "": "v.blk.%d.attn_q_norm.weight",
        "blocks.%d.norm1.weight": "v.blk.%d.ln1.weight",
        "blocks.%d.norm2.weight": "v.blk.%d.ln2.weight",
        "blocks.%d.ls1.scale_factor": "v.blk.%d.ls1.weight",
        "blocks.%d.ls2.scale_factor": "v.blk.%d.ls2.weight",

        "blocks.%d.attn.qkv.bias": "v.blk.%d.attn_q.bias",
        # "blocks.%d.attn.qkv.bias": "v.blk.%d.attn_k.bias",
        # "blocks.%d.attn.qkv.weight": "v.blk.%d.attn_vision.bias",
        "blocks.%d.attn.proj.bias": "v.blk.%d.attn_out.bias",
        "blocks.%d.norm1.bias": "v.blk.%d.ln1.bias",
        "blocks.%d.norm2.bias": "v.blk.%d.ln2.bias",

        "blocks.%d.mlp.fc1.weight": "v.blk.%d.ffn_up.weight",
        # "": "v.blk.%d.ffn_gate.weight",
        "blocks.%d.mlp.fc2.weight": "v.blk.%d.ffn_down.weight",

        "blocks.%d.mlp.fc1.bias": "v.blk.%d.ffn_up.bias",
        # "": "v.blk.%d.ffn_gate.bias",
        "blocks.%d.mlp.fc2.bias": "v.blk.%d.ffn_down.bias",
    }

    for model_name, model in zip(model_names, models):
        gguf_writer = gguf.GGUFWriter(f"{model_name}.gguf", model_name)

        model_cfg = cfgs[model_name]
        add_config(gguf_writer, model_cfg)
        cur_params = {}
        cur_params.update(model_params)

        n_blocks = model_cfg.get("clip.vision.block_count", 0)
        for b_id in range(n_blocks):
            for k, v in blk_params.items():
                new_k = k % b_id
                if new_k in cur_params:
                    raise ValueError(f"{new_k} already exists")
                cur_params[new_k] = v % b_id

        for tensor_name, new_tensor_name in cur_params.items():
            if tensor_name not in model.state_dict():
                continue
            param = model.state_dict()[tensor_name]
            # bias和norm层
            if param.dim() <= 1:
                param = param.to(torch.float32)
            elif tensor_name.endswith(("pos_embed", "_norm.weight", "cls_token", "reg_token")):
                param = param.to(torch.float32)
            if "qkv" in tensor_name:
                new_q_name = new_tensor_name
                new_k_name = new_tensor_name.replace("attn_q", "attn_k")
                new_v_name = new_tensor_name.replace("attn_q", "attn_v")
                split_length = param.shape[0] // 3
                q, k, v = torch.split(param, split_length, dim=0)
                gguf_writer.add_tensor(new_q_name, q.squeeze().cpu().numpy())
                gguf_writer.add_tensor(new_k_name, k.squeeze().cpu().numpy())
                gguf_writer.add_tensor(new_v_name, v.squeeze().cpu().numpy())
            else:
                gguf_writer.add_tensor(
                    new_tensor_name, param.squeeze().cpu().numpy())

        gguf_writer.write_header_to_file()
        gguf_writer.write_kv_data_to_file()
        gguf_writer.write_tensors_to_file()
        gguf_writer.close()


# # Grab image input & format prompt
# # image: Image.Image = get_from_camera(...)
# image = Image.open("pokemon.jpeg").convert("RGB")
# # instruction = "Pick up the red mug and place it on the tray."
# instruction = "Move the mouse next to the stapler".lower()
# instruction = "move pokman to the left".lower()
# prompt = f"In: What action should the robot take to {instruction}?\nOut:"

# # Predict Action (7-DoF; un-normalize for BridgeV2)
# inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
# action = vla.predict_action(
#     **inputs, unnorm_key="bridge_orig", do_sample=False)

# print(action)
# # Execute...
# # robot.act(action, ...)
