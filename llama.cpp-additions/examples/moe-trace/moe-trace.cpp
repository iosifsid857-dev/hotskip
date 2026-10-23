// moe-trace: run a MoE model on a prompt, generate N tokens, and log which experts the router picked.
// Output (one line per layer per token): <token_index> <layer> <expert ids...>
// Usage: llama-moe-trace -m model.gguf -p "prompt" -n 128 --trace-out trace.txt
#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct trace_state {
    FILE * out = nullptr;
    int token_index = 0;  // index of the first token in the current batch
    std::vector<uint8_t> buf;
};

// Called by the scheduler for every graph node. We only want the router's top-k expert ids:
// llama.cpp names that tensor "ffn_moe_topk-<layer>" (int32, shape [n_expert_used, n_tokens]).
static bool on_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (trace_state *) user_data;
    const char * name = ggml_get_name(t);
    const bool is_topk = strncmp(name, "ffn_moe_topk-", 13) == 0 && t->type == GGML_TYPE_I32;
    if (ask) {
        return is_topk;
    }
    if (!is_topk) {
        return true;
    }
    const int layer = atoi(name + 13);
    const int64_t k = t->ne[0];
    const int64_t n_tok = t->ne[1];
    st->buf.resize(ggml_nbytes(t));
    ggml_backend_tensor_get(t, st->buf.data(), 0, ggml_nbytes(t));
    const int32_t * ids = (const int32_t *) st->buf.data();
    for (int64_t j = 0; j < n_tok; j++) {
        fprintf(st->out, "%d %d", st->token_index + (int) j, layer);
        for (int64_t i = 0; i < k; i++) {
            fprintf(st->out, " %d", ids[j * (t->nb[1] / sizeof(int32_t)) + i]);
        }
        fputc('\n', st->out);
    }
    return true;
}

int main(int argc, char ** argv) {
    std::string trace_path = "moe-trace.txt";
    bool chat = false; // --chat: wrap the prompt as a user message in the model's own chat template
    // pull our own flag out before common parsing sees it
    std::vector<char *> args;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--trace-out") == 0 && i + 1 < argc) {
            trace_path = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "--chat") == 0) {
            chat = true;
            continue;
        }
        args.push_back(argv[i]);
    }
    int ac = (int) args.size();

    common_params params;
    if (!common_params_parse(ac, args.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    trace_state st;
    st.out = fopen(trace_path.c_str(), "w");
    if (!st.out) {
        fprintf(stderr, "cannot open %s\n", trace_path.c_str());
        return 1;
    }
    params.cb_eval = on_eval;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    llama_backend_init();
    common_init_result_ptr init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (!model || !ctx) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::string prompt = params.prompt;
    if (chat) { // trace what the chat server actually feeds the model (reasoning off, like hotskip run)
        common_chat_templates_ptr tmpls = common_chat_templates_init(model, "");
        common_chat_templates_inputs in;
        common_chat_msg msg;
        msg.role    = "user";
        msg.content = params.prompt;
        in.messages = { msg };
        in.enable_thinking = false;
        prompt = common_chat_templates_apply(tmpls.get(), in).prompt.str();
    }
    std::vector<llama_token> toks = common_tokenize(ctx, prompt, !chat, true);
    common_sampler * smpl = common_sampler_init(model, params.sampling);

    llama_batch batch = llama_batch_get_one(toks.data(), (int32_t) toks.size());
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }
    st.token_index = (int) toks.size();

    const int n_predict = params.n_predict > 0 ? params.n_predict : 128;
    for (int i = 0; i < n_predict; i++) {
        llama_token id = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, id, true);
        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }
        printf("%s", common_token_to_piece(ctx, id).c_str());
        fflush(stdout);
        batch = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx, batch)) {
            break;
        }
        st.token_index++;
    }
    printf("\n");
    fclose(st.out);
    common_sampler_free(smpl);
    llama_backend_free();
    return 0;
}
