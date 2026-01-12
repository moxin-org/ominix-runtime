#pragma once

#include "build_graph.h"
#include "ctx_manager.h"
#include "ggml.h"
#include "model_loader.h"
#include <unordered_set>
#include <vector>
#include <functional>

class BaseModel {
public:
  bool load_tensors(ModelLoader &model_loader, ContextManager &ctx_manager);

  virtual bool load_hparams(const ModelLoader &model_loader) { return true; }
  virtual std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) {
    return {};
  }
  virtual ~BaseModel() = default;
};

class FakeModel: public BaseModel {
  public:
  std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
  std::vector<ggml_tensor *> build_graph(ggml_context *ctx0);

  ggml_tensor *embed_tokens = nullptr;
  ggml_tensor *llm_head = nullptr;
};

struct VisionParams {
  int32_t image_size = 224;
  int32_t patch_size;
  int32_t n_embd;
  int32_t n_ff;
  int32_t n_head;
  int32_t n_layer;
  std::vector<float> image_mean;
  std::vector<float> image_std;
  ffn_op_type ffn_op = FFN_GELU;
  float eps = 1e-6;
  int32_t projection_dim;
  std::unordered_set<int32_t> vision_feature_layer;
};

struct ClipLayer {
  // attention
  ggml_tensor *k_w = nullptr;
  ggml_tensor *k_b = nullptr;
  ggml_tensor *q_w = nullptr;
  ggml_tensor *q_b = nullptr;
  ggml_tensor *v_w = nullptr;
  ggml_tensor *v_b = nullptr;

  ggml_tensor *o_w = nullptr;
  ggml_tensor *o_b = nullptr;

  ggml_tensor *k_norm = nullptr;
  ggml_tensor *q_norm = nullptr;

  // layernorm 1
  ggml_tensor *ln_1_w = nullptr;
  ggml_tensor *ln_1_b = nullptr;

  ggml_tensor *ff_up_w = nullptr;
  ggml_tensor *ff_up_b = nullptr;
  ggml_tensor *ff_gate_w = nullptr;
  ggml_tensor *ff_gate_b = nullptr;
  ggml_tensor *ff_down_w = nullptr;
  ggml_tensor *ff_down_b = nullptr;

  // layernorm 2
  ggml_tensor *ln_2_w = nullptr;
  ggml_tensor *ln_2_b = nullptr;

  // layer scale (no bias)
  ggml_tensor *ls_1_w = nullptr;
  ggml_tensor *ls_2_w = nullptr;
};

class ProjectorModel : public BaseModel {
public:
  std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
  std::vector<ggml_tensor *> build_graph(ggml_context *ctx0);

  ggml_tensor *fc1_weight = nullptr;
  ggml_tensor *fc1_bias = nullptr;
  ggml_tensor *fc2_weight = nullptr;
  ggml_tensor *fc2_bias = nullptr;
  ggml_tensor *fc3_weight = nullptr;
  ggml_tensor *fc3_bias = nullptr;
};

class VisionTransformerModel : public BaseModel {
public:
  std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
  bool load_hparams(const ModelLoader &model_loader) override;
  std::vector<ggml_tensor *> build_graph(ggml_context *ctx0);

  ggml_tensor *build_vit(
      ggml_context *ctx0, ggml_tensor *inp, int n_pos, norm_type norm_t,
      std::function<ggml_tensor *(ggml_tensor *, const ClipLayer &)> add_pos);

  VisionParams hparams;
  // embeddings
  ggml_tensor *class_embedding = nullptr;
  ggml_tensor *reg_embedding = nullptr;
  ggml_tensor *patch_embeddings_0 = nullptr;
  ggml_tensor *patch_embeddings_1 =
      nullptr; // second Conv2D kernel when we decouple Conv3D along temproal
               // dimension (Qwen2VL)
  ggml_tensor *patch_bias = nullptr;
  ggml_tensor *position_embeddings = nullptr;

  ggml_tensor *pre_ln_w = nullptr;
  ggml_tensor *pre_ln_b = nullptr;

  std::vector<ClipLayer> layers;

  ggml_tensor *post_ln_w;
  ggml_tensor *post_ln_b;
};

struct RegressionParams {
  int action_dim = 7;
  int num_actions_chunk = 8;
  int num_actions_per_token = 8;
  int num_blocks = 4;
  int input_dim = 2048;
  int hidden_dim = 512;
  int expansion = 4;
};

class MLPResNetBlockV2 {
public:
  /*
  class MLPResNetBlockV2(nn.Module):
      def __init__(self, dim, expansion=4, dropout=0.1):
          super().__init__()
          self.ffn = nn.Sequential(
              nn.LayerNorm(dim),
              nn.Linear(dim, dim * expansion),
              nn.SiLU(),
              nn.Linear(dim * expansion, dim)
          )
          self.dropout = nn.Dropout(dropout)

      def forward(self, x):
          identity = x
          x_ffn = self.ffn(x)
          x_dropped = self.dropout(x_ffn)
          x = x_dropped + identity
          return x
  */
  ggml_tensor *ffn_ln_w = nullptr;
  ggml_tensor *ffn_ln_b = nullptr;
  ggml_tensor *ffn_fc_w = nullptr;
  ggml_tensor *ffn_fc_b = nullptr;
  ggml_tensor *ffn_fc2_w = nullptr;
  ggml_tensor *ffn_fc2_b = nullptr;

  ggml_tensor *build_graph(ggml_context *ctx0, ggml_tensor* inp);
};

class L1RegressionActionHeadFunnelModel : public BaseModel {
public:
  std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
  bool load_hparams(const ModelLoader &model_loader) override;
  std::vector<ggml_tensor *> build_graph(ggml_context *ctx0);

  RegressionParams hparams;

  /*
    self.input_proj = nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
    )
  */
  ggml_tensor *input_proj_ln_w = nullptr;
  ggml_tensor *input_proj_ln_b = nullptr;
  ggml_tensor *input_proj_fc_w = nullptr;
  ggml_tensor *input_proj_fc_b = nullptr;

  std::vector<MLPResNetBlockV2> resnet_body;

  ggml_tensor *output_head_ln_w = nullptr;
  ggml_tensor *output_head_ln_b = nullptr;
  ggml_tensor *output_head_fc_w = nullptr;
  ggml_tensor *output_head_fc_b = nullptr;
};