#include "llm.h"
#include "llama.h"
#include "utils.h"
#include <algorithm>
#include <cassert>
#include <numeric>
#include <string.h>

static std::string common_token_to_piece(const llama_vocab *vocab,
                                         llama_token token, int32_t lstrip = 0,
                                         bool special = true) {
  char buf[256];
  int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), lstrip, special);
  if (n < 0) {
    GGML_ABORT("failed to convert token to piece\n");
  }
  std::string piece(buf, n);
  printf("%s", piece.c_str());
  fflush(stdout);
  return piece;
}
Llm::Llm() {
  // only print errors
  llama_log_set(
      [](enum ggml_log_level level, const char *text, void * /* user_data */) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
          fprintf(stderr, "%s", text);
        }
      },
      nullptr);

  // load dynamic backends
  ggml_backend_load_all();
}

bool Llm::load_model(const std::string &model_path, LlmParam params) {
  // initialize the model
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = params.ngl;
  model_ = llama_model_load_from_file(model_path.c_str(), model_params);
  if (!model_) {
    fprintf(stderr, "%s: error: unable to load model\n", __func__);
    return false;
  }
  vocab_ = llama_model_get_vocab(model_);

  // initialize the context
  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = params.n_ctx;
  ctx_params.n_batch = params.n_ctx;
  ctx_params.embeddings = params.embeddings;
  require_embeddings_ = params.embeddings;
  ctx_ = llama_init_from_model(model_, ctx_params);
  if (!ctx_) {
    fprintf(stderr, "%s: error: failed to create the llama_context\n",
            __func__);
    return false;
  }
  n_ctx_ = llama_n_ctx(ctx_);

  // initialize the sampler
  smpl_ = llama_sampler_chain_init(llama_sampler_chain_default_params());
  llama_sampler_chain_add(smpl_, llama_sampler_init_greedy());
  // llama_sampler_chain_add(smpl_, llama_sampler_init_min_p(0.05f, 1));
  // llama_sampler_chain_add(smpl_, llama_sampler_init_temp(0.8f));
  // llama_sampler_chain_add(smpl_,
  // llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

  if (!params.tokenizer_path.empty()) {
    init_tokenizer(params.tokenizer_path);
  }

  return true;
}

bool Llm::encode_text_by_tokenizer_cpp(const std::string &prompt,
                                       std::vector<llama_token> &prompt_tokens,
                                       bool add_special) {
  // tokenize the prompt
  prompt_tokens = tokenizer_->encode(prompt, add_special);
  return true;
}

std::string Llm::format_prompt(const std::string &prompt,
                               const std::string &system_prompt) {
  std::vector<llama_chat_message> messages;
  std::vector<char> formatted(llama_n_ctx(ctx_));
  const char *tmpl = llama_model_chat_template(model_, /* name */ nullptr);
  // add the user input to the message list and format it
  if (!system_prompt.empty()) {
    messages.push_back({"system", strdup(system_prompt.c_str())});
  }
  messages.push_back({"user", strdup(prompt.c_str())});
  int new_len =
      llama_chat_apply_template(tmpl, messages.data(), messages.size(), true,
                                formatted.data(), formatted.size());
  if (new_len > (int)formatted.size()) {
    formatted.resize(new_len);
    new_len =
        llama_chat_apply_template(tmpl, messages.data(), messages.size(), true,
                                  formatted.data(), formatted.size());
  }
  if (new_len < 0) {
    fprintf(stderr, "failed to apply the chat template\n");
    return "";
  }
  return std::string(formatted.data(), new_len);
}

bool Llm::encode_text(const std::string &prompt,
                      std::vector<llama_token> &prompt_tokens,
                      bool add_special) {
  // tokenize the prompt
  const int n_prompt_tokens = -llama_tokenize(
      vocab_, prompt.c_str(), prompt.size(), NULL, 0, add_special, true);
  prompt_tokens.resize(n_prompt_tokens);
  if (llama_tokenize(vocab_, prompt.c_str(), prompt.size(),
                     prompt_tokens.data(), prompt_tokens.size(), add_special,
                     true) < 0) {
    GGML_ABORT("failed to tokenize the prompt\n");
  }
  return true;
}

Llm::~Llm() {
  if (smpl_) {
    llama_sampler_free(smpl_);
    smpl_ = nullptr;
  }
  if (ctx_) {
    llama_free(ctx_);
    ctx_ = nullptr;
  }
  if (model_) {
    llama_model_free(model_);
    model_ = nullptr;
  }
}

bool Llm::init_tokenizer(const std::string &tokenizer_path) {
  tokenizer_ = create_tokenizer(tokenizer_path);
  return (tokenizer_ != nullptr);
}

bool Llm::eval_chunk(llama_token *tokens, float *embd, int n_tokens,
                     bool is_last) {
  // llama_pos n_past = llama_memory_seq_pos_max(llama_get_memory(ctx_), 0) + 1;
  // std::vector<llama_pos> pos(n_tokens);
  // std::iota(pos.begin(), pos.end(), n_past);
  // std::vector<llama_seq_id> tmp_seq(n_tokens, 0);
  // std::vector<llama_seq_id *> seq_id(n_tokens, tmp_seq.data());
  // std::vector<int> n_seq_id(n_tokens, 1);

  std::vector<int8_t> logits(n_tokens, 0);
  if (is_last) {
    logits[n_tokens - 1] = true;
  }
  {
    llama_batch batch = {
        .n_tokens = n_tokens,
        .token = tokens ? tokens : nullptr,
        .embd = embd ? embd : nullptr,
        .pos = nullptr,
        .n_seq_id = nullptr,
        .seq_id = nullptr,
        .logits = logits.data(),
    };

    int ret = llama_decode(ctx_, batch);
    if (ret != 0) {
      // GGML_ABORT("failed to decode, ret = %d\n", ret);
      return false;
    }
  }
  return true;
}

bool Llm::get_last_hidden_state(std::vector<float> &output) const {
  uint32_t n_outputs = llama_get_n_outputs(ctx_);
  int32_t n_embd = llama_model_n_embd(model_);
  output.resize(n_embd);
  float *all_embeddings = llama_get_embeddings(ctx_);
  memcpy(output.data(), all_embeddings + (n_outputs - 1) * n_embd,
         n_embd * sizeof(float));
  return true;
}

bool Llm::get_last_logit(std::vector<float> &output) const {
  uint32_t n_outputs = llama_get_n_outputs(ctx_);
  size_t n_logits = llama_vocab_n_tokens(vocab_);
  output.resize(n_logits);
  float *logits = llama_get_logits(ctx_);
  memcpy(output.data(), logits + (n_outputs - 1) * n_logits,
         n_logits * sizeof(float));
  return true;
}

std::string Llm::generate(const std::string &prompt,
                          const std::string &system_prompt,
                          std::vector<llama_token> &generated_tokens,
                          std::vector<float> &embeddings, bool use_history) {
  if (!use_history) {
    llama_memory_clear(llama_get_memory(ctx_), true);
    llama_synchronize(ctx_);
    llama_perf_context_reset(ctx_);
    llama_set_warmup(ctx_, false);
  }
  const bool is_first =
      llama_memory_seq_pos_max(llama_get_memory(ctx_), 0) == -1;
  // printf(" is_first: %s\n", is_first ? "true" : "false");

  // apply template
  std::string formated_prompt = format_prompt(prompt, system_prompt);
  printf("prompt: %s\n", formated_prompt.c_str());
  std::vector<llama_token> prompt_tokens;
  if (!encode_text(formated_prompt, prompt_tokens, is_first)) {
    fprintf(stderr, "%s: Failed to encode text\n", __func__);
    return "";
  };

  if (!eval_chunk(prompt_tokens.data(), nullptr, prompt_tokens.size(), true)) {
    fprintf(stderr, "%s: prefill failed.\n", __func__);
    return "";
  }

  {
    std::vector<float> v_logits;
    get_last_logit(v_logits);
    // print_vector(v_logits, 20);
  }

  if (require_embeddings_) {
    get_last_hidden_state(embeddings);
    // print_vector(embeddings, 20);
    return "";
  }

  //   llama_token new_token_id;
  std::string response = "";
  generated_tokens.clear();
  while (true) {
    // sample the next token
    llama_token new_token_id = llama_sampler_sample(smpl_, ctx_, -1);

    // is it an end of generation?
    if (llama_vocab_is_eog(vocab_, new_token_id)) {
      printf("\n");
      fflush(stdout);
      break;
    }
    generated_tokens.push_back(new_token_id);

    std::string piece = common_token_to_piece(vocab_, new_token_id, 0, true);
    response += piece;

    if (!eval_chunk(&new_token_id, nullptr, 1, true)) {
      fprintf(stderr, "%s: decode failed.\n", __func__);
      return "";
    }
  }
  return response;
}

std::string OpenvlaLlm::format_prompt(const std::string &prompt,
                                      const std::string &system_prompt) {
  std::string formated_prompt =
      "<__media__>In: What action should the robot take to " + prompt +
      "?\nOut:";
  return formated_prompt;
}

bool OpenvlaLlm::generate(const std::string &prompt, float *img_emb,
                          int32_t n_img_tokens,
                          std::vector<llama_token> &generated_tokens,
                          std::vector<float> &output, bool use_history) {
  if (!use_history) {
    llama_memory_clear(llama_get_memory(ctx_), true);
    llama_synchronize(ctx_);
    llama_perf_context_reset(ctx_);
    llama_set_warmup(ctx_, false);
  }
  // apply template
  std::string formated_prompt = format_prompt(prompt, "");
  // printf("prompt: %s\n", formated_prompt.c_str());
  std::vector<llama_token> prompt_tokens;

  std::string template_img = "<__media__>";
  std::vector<std::string> texts = split_text(formated_prompt, template_img);
  if (texts.size() > 1 && texts[0] == template_img) {
    texts.insert(texts.begin(), "");
  }

  for (size_t i = 0; i < texts.size(); i++) {
    llama_token *p_tokens = nullptr;
    int n_tokens = 0;
    float *embd = nullptr;
    bool is_last = i == texts.size() - 1;
    if (texts[i] == template_img) {
      embd = img_emb;
      n_tokens = n_img_tokens;
      p_tokens = nullptr;
    } else {
      if (tokenizer_) {
        encode_text_by_tokenizer_cpp(texts[i], prompt_tokens, i == 0);
      } else {
        encode_text(texts[i], prompt_tokens, i == 0);
        if (prompt_tokens[0] == 797) {
          prompt_tokens[0] = 512; // for debug
        }
      }
      // print_vector(prompt_tokens, prompt_tokens.size());
      if (is_last && !prompt_tokens.empty() &&
          prompt_tokens.back() != empty_token_) {
        prompt_tokens.push_back(empty_token_);
      }
      // print_vector(prompt_tokens, prompt_tokens.size());
      p_tokens = prompt_tokens.data();
      n_tokens = prompt_tokens.size();
      embd = nullptr;
    }
    // print_vector(prompt_tokens, prompt_tokens.size());
    if (!eval_chunk(p_tokens, embd, n_tokens, is_last)) {
      fprintf(stderr, "%s: prefill failed.\n", __func__);
      return false;
    }
  }
  {
    std::vector<float> v_logits;
    get_last_logit(v_logits);
    // print_vector(v_logits, 20);
  }

  if (require_embeddings_) {
    get_last_hidden_state(output);
    // print_vector(output, 20);
    return true;
  }

  generated_tokens.clear();
  while (true) {
    // sample the next token
    llama_token new_token_id = llama_sampler_sample(smpl_, ctx_, -1);

    // is it an end of generation?
    if (llama_vocab_is_eog(vocab_, new_token_id)) {
      printf("\n");
      fflush(stdout);
      break;
    }
    generated_tokens.push_back(new_token_id);

    if (!eval_chunk(&new_token_id, nullptr, 1, true)) {
      fprintf(stderr, "%s: decode failed.\n", __func__);
      return false;
    }
  }
  assert(generated_tokens.size() >= 7);
  size_t vocab_size = llama_vocab_n_tokens(vocab_) - pad_to_multiple_of_;
  std::vector<llama_token> predicted_action_token_ids(
      generated_tokens.end() - 7, generated_tokens.end());
  std::transform(predicted_action_token_ids.begin(),
                 predicted_action_token_ids.end(),
                 predicted_action_token_ids.begin(), [&](llama_token token) {
                   token = vocab_size - token - 1;
                   return std::max(token, 0);
                 });
  generated_tokens = predicted_action_token_ids;
  return true;
}